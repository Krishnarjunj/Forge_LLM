"""Training loop with full resume semantics (M9).

Implements the differentiation move #1 -- a 12-hour Kaggle session can die at
any minute, and the loss curve after resume must be bitwise-indistinguishable
from the uninterrupted run on a single GPU (CLAUDE.md sec 5, sec 7, sec 8).

Checkpoint format follows CLAUDE.md sec 8: model state, optimizer state,
scheduler state, GradScaler state, RNG states (torch, cuda, numpy, python),
data-iterator state, wandb run id, config hash, and git SHA.

Mixed precision (fp16 + GradScaler) lights up only when ``dtype="fp16"`` AND
the device is CUDA -- on CPU we always run fp32 with the scaler disabled, so
the resume-safety test can exercise the resume path on a laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import (  # type: ignore[attr-defined]  # not in torch.amp __all__ but is the documented API
    GradScaler,
    autocast,
)

from forge_llm.config import PRESETS, ForgeConfig
from forge_llm.data import PackedDataset, fineweb_doc_iterator
from forge_llm.model import ForgeForCausalLM
from forge_llm.tokenizer import BPETokenizer

logger = logging.getLogger("forge_llm.train")


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters. Frozen so config hash is stable on resume."""

    lr: float = 3.0e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-8
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    warmup_steps: int = 0
    total_steps: int = 1000
    min_lr: float = 3.0e-5

    micro_bs: int = 8
    grad_accum: int = 1
    seq_len: int = 1024

    seed: int = 0
    dtype: str = "fp32"  # "fp32" or "fp16"


def _seed_all(seed: int) -> None:
    """CLAUDE.md sec 5 seeding block. Called by Trainer.__init__ and resume."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git_sha() -> str:
    """Resolve HEAD commit SHA; returns "<unknown>" if not in a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "<unknown>"


def _config_hash(model_config: ForgeConfig, train_config: TrainConfig) -> str:
    """SHA-256 of the merged JSON representation of both configs."""
    payload = json.dumps(
        {"model": asdict(model_config), "train": asdict(train_config)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup -> cosine decay schedule. Plain Python so it's bit-stable."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


@dataclass
class _State:
    """Mutable training state -- snapshotted into checkpoints."""

    step: int = 0
    wandb_run_id: str | None = None


class Trainer:
    """Causal-LM trainer with checkpoint round-tripping (CLAUDE.md sec 8)."""

    def __init__(
        self,
        model_config: ForgeConfig,
        train_config: TrainConfig,
        data_factory: Callable[[], Iterator[str]],
        tokenizer: BPETokenizer,
        device: str = "cpu",
    ) -> None:
        _seed_all(train_config.seed)

        self.model_config = model_config
        self.train_config = train_config
        self.data_factory = data_factory
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model = ForgeForCausalLM(model_config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config.lr,
            betas=train_config.betas,
            eps=train_config.eps,
            weight_decay=train_config.weight_decay,
        )

        # GradScaler is only meaningful for fp16 on CUDA; on CPU it's a no-op
        # but we keep the instance so checkpoint format is uniform.
        scaler_enabled = train_config.dtype == "fp16" and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=scaler_enabled)

        self.dataset = PackedDataset(
            doc_iter_factory=data_factory,
            tokenizer=tokenizer,
            seq_len=train_config.seq_len,
            eos_id=tokenizer.eos_id,
        )
        self._data_iter: Iterator[Tensor] | None = None

        self._state = _State()
        self._config_hash = _config_hash(model_config, train_config)
        self._git_sha = _git_sha()

    # ----- training loop -----

    def _ensure_data_iter(self) -> Iterator[Tensor]:
        if self._data_iter is None:
            self._data_iter = iter(self.dataset)
        return self._data_iter

    def _next_batch(self) -> Tensor:
        it = self._ensure_data_iter()
        seqs: list[Tensor] = []
        for _ in range(self.train_config.micro_bs):
            seqs.append(next(it))
        return torch.stack(seqs).to(self.device)

    def _set_lr(self, step: int) -> float:
        lr = _cosine_lr(
            step,
            self.train_config.warmup_steps,
            self.train_config.total_steps,
            self.train_config.lr,
            self.train_config.min_lr,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def _forward_loss(self, batch: Tensor) -> Tensor:
        # Standard next-token CE: shift inputs by one.
        inputs = batch[:, :-1].contiguous()
        targets = batch[:, 1:].contiguous()
        logits = self.model(inputs)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    def train_steps(self, n_steps: int) -> list[float]:
        """Run ``n_steps`` optimisation steps; return per-step loss values."""
        self.model.train()
        losses: list[float] = []

        amp_enabled = self.train_config.dtype == "fp16" and self.device.type == "cuda"

        for _ in range(n_steps):
            self._set_lr(self._state.step)
            self.optimizer.zero_grad(set_to_none=True)

            accumulated_loss = 0.0
            for _ in range(self.train_config.grad_accum):
                batch = self._next_batch()
                if amp_enabled:
                    with autocast(device_type=self.device.type, dtype=torch.float16):
                        loss = self._forward_loss(batch)
                else:
                    loss = self._forward_loss(batch)
                # Scale by 1/grad_accum so gradients average across micro-batches.
                loss = loss / self.train_config.grad_accum
                self.scaler.scale(loss).backward()  # type: ignore[no-untyped-call]  # GradScaler.scale returns Tensor with untyped backward stub
                accumulated_loss += loss.detach().item()

            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.train_config.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            losses.append(accumulated_loss)
            self._state.step += 1

        return losses

    # ----- checkpoint -----

    def save_checkpoint(self, path: str | Path) -> Path:
        """Write the 9-field checkpoint dict per CLAUDE.md sec 8."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        ckpt: dict[str, Any] = {
            "step": self._state.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {
                "warmup_steps": self.train_config.warmup_steps,
                "total_steps": self.train_config.total_steps,
                "min_lr": self.train_config.min_lr,
            },
            "scaler": self.scaler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "data_iterator": self.dataset.state_dict(),
            "wandb_run_id": self._state.wandb_run_id,
            "config_hash": self._config_hash,
            "git_sha": self._git_sha,
            "model_config_dict": asdict(self.model_config),
            "train_config_dict": asdict(self.train_config),
        }
        torch.save(ckpt, out)
        logger.info("checkpoint written: %s (step=%d)", out, self._state.step)
        return out

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        data_factory: Callable[[], Iterator[str]],
        tokenizer: BPETokenizer,
        device: str = "cpu",
        force: bool = False,
    ) -> Trainer:
        """Reconstitute a Trainer from a checkpoint and the runtime-only deps."""
        ckpt: dict[str, Any] = torch.load(Path(path), map_location=device, weights_only=False)
        model_config = ForgeConfig(**ckpt["model_config_dict"])
        train_cfg_dict = dict(ckpt["train_config_dict"])
        # Tuples become lists in dataclass-asdict-then-json round trips;
        # asdict preserves tuples natively so this stays a tuple, but
        # belt-and-suspenders cast.
        train_cfg_dict["betas"] = tuple(train_cfg_dict["betas"])
        train_config = TrainConfig(**train_cfg_dict)

        trainer = cls(
            model_config=model_config,
            train_config=train_config,
            data_factory=data_factory,
            tokenizer=tokenizer,
            device=device,
        )

        current_hash = _config_hash(model_config, train_config)
        if not force and current_hash != ckpt["config_hash"]:
            raise RuntimeError(
                "config_hash drift between checkpoint and current code; "
                "pass force=True to override."
            )

        trainer.model.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.scaler.load_state_dict(ckpt["scaler"])
        trainer.dataset.load_state_dict(ckpt["data_iterator"])

        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

        trainer._state.step = ckpt["step"]
        trainer._state.wandb_run_id = ckpt.get("wandb_run_id")
        # data_iter rebuilt lazily on the next train step.
        trainer._data_iter = None
        return trainer


# ---------------------------------------------------------------------------
# CLI: `python -m forge_llm.train --config configs/model_30m.yaml --steps 5`
# ---------------------------------------------------------------------------

_MOCK_DOCS: list[str] = [
    "Forge-LLM is a from-scratch decoder trained on free Kaggle T4 GPU.",
    "The quick brown fox jumps over the lazy dog repeatedly.",
    "Packed datasets concatenate documents with EOS separators between them.",
    "Training a causal language model means predicting the next token given the previous ones.",
    "Mixed precision uses fp16 forward and backward with fp32 master weights.",
] * 200


def _mock_factory() -> Iterator[str]:
    return iter(_MOCK_DOCS)


def _load_model_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]  # noqa: PLC0415 — pyyaml has no stubs in our dev set
    except ImportError as exc:
        raise RuntimeError(
            "Loading a YAML config requires `pyyaml`. Install with: pip install pyyaml"
        ) from exc
    with path.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)
    # Strip purely-informational fields the dataclass doesn't accept.
    for k in ("activation", "norm", "position", "attention"):
        data.pop(k, None)
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a model YAML (e.g. configs/model_30m.yaml). "
        "If omitted, uses PRESETS['forge-30m'].",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of training steps to run (default: 5, smoke).",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging (always disabled in the smoke gate).",
    )
    parser.add_argument(
        "--mock-data",
        action="store_true",
        default=True,
        help="Use an inline mock corpus instead of FineWeb-Edu streaming. "
        "Default true so the smoke gate runs offline.",
    )
    parser.add_argument(
        "--micro-bs",
        type=int,
        default=2,
        help="Micro batch size override for the smoke run.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Sequence length override for the smoke run.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (default: cpu).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve model config: YAML if provided, else the registered preset.
    if args.config is not None:
        model_cfg_dict = _load_model_yaml(args.config)
    else:
        model_cfg_dict = dict(PRESETS["forge-30m"])

    # Smoke override: smaller seq + small max_seq so a 5-step CPU run is fast.
    model_cfg_dict["max_seq"] = min(args.seq_len, model_cfg_dict["max_seq"])

    model_config = ForgeConfig(**model_cfg_dict)
    train_config = TrainConfig(
        lr=1e-3,
        micro_bs=args.micro_bs,
        grad_accum=1,
        seq_len=args.seq_len,
        warmup_steps=0,
        total_steps=args.steps,
        min_lr=1e-4,
        seed=0,
        dtype="fp32",
    )

    if args.mock_data:
        factory: Callable[[], Iterator[str]] = _mock_factory
        tokenizer = BPETokenizer.train(" ".join(_MOCK_DOCS), vocab_size=model_config.vocab_size)
    else:
        # Non-mock data path lands in Phase F when configs/tokenizer.json
        # exists and FineWeb-Edu streaming is wired in via fineweb_doc_iterator.
        _ = fineweb_doc_iterator  # mark imported for Phase F
        raise NotImplementedError(
            "Non-mock data path lands in Phase F (with configs/tokenizer.json)."
        )

    if args.no_wandb:
        logger.info("wandb disabled (smoke run)")

    trainer = Trainer(
        model_config=model_config,
        train_config=train_config,
        data_factory=factory,
        tokenizer=tokenizer,
        device=args.device,
    )
    logger.info("starting %d-step smoke run on %s", args.steps, args.device)
    losses = trainer.train_steps(args.steps)
    logger.info("done. losses: %s", [f"{x:.4f}" for x in losses])
    return 0


if __name__ == "__main__":
    sys.exit(main())
