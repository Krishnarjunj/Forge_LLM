"""HuggingFace Hub + Kaggle Dataset checkpoint helpers (M9).

* ``save_pretrained(trainer, out_dir)`` writes the checkpoint, the model
  config as JSON, and the tokenizer JSON into ``out_dir``.
* ``from_pretrained(out_dir, data_factory, tokenizer)`` reverses it.
* ``push_to_hub(local_dir, repo_id, token, dry_run)`` uploads the directory to
  HF Hub. With ``dry_run=True`` it returns the would-be upload manifest
  without contacting the network; the M9 smoke gate exercises this path so
  the helper has CI coverage without requiring an ``HF_TOKEN`` secret.
* The Kaggle Dataset mirror lands in Phase F when we have real checkpoints
  to push; the function is stubbed with a NotImplementedError so callers see
  a clear failure rather than a silent no-op.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge_llm.tokenizer import BPETokenizer

logger = logging.getLogger("forge_llm.hub")


def save_pretrained(trainer: Any, out_dir: str | Path) -> Path:
    """Write trainer checkpoint + config + tokenizer JSON to ``out_dir``.

    Mirrors the HuggingFace ``save_pretrained`` API but is shaped around our
    own checkpoint format (CLAUDE.md sec 8). Trainer is typed as ``Any`` to
    avoid a circular import; the only methods we call are
    ``save_checkpoint`` and the ``model_config``/``tokenizer`` attributes.
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    ckpt_path = trainer.save_checkpoint(target / "checkpoint.pt")
    (target / "config.json").write_text(
        json.dumps(asdict(trainer.model_config), indent=2, sort_keys=True)
    )
    trainer.tokenizer.save(target / "tokenizer.json")
    logger.info(
        "save_pretrained wrote %s, %s, %s",
        ckpt_path.name,
        "config.json",
        "tokenizer.json",
    )
    return target


def from_pretrained(
    out_dir: str | Path,
    data_factory: Callable[[], Iterator[str]],
    device: str = "cpu",
) -> Any:
    """Reload a Trainer from ``save_pretrained`` artefacts."""
    src = Path(out_dir)
    # Lazy import to break the train.py <-> hub.py cycle.
    from forge_llm.train import Trainer  # noqa: PLC0415

    tokenizer = BPETokenizer.load(src / "tokenizer.json")
    return Trainer.load_checkpoint(
        src / "checkpoint.pt",
        data_factory=data_factory,
        tokenizer=tokenizer,
        device=device,
    )


def push_to_hub(
    local_dir: str | Path,
    repo_id: str,
    token: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Upload ``local_dir`` to the HF Hub repo at ``repo_id``.

    Returns a manifest dict ``{"repo_id", "files", "dry_run"}``. With
    ``dry_run=True`` no network calls are made and the manifest is returned
    immediately -- this is what CI exercises (no HF token needed).
    """
    src = Path(local_dir)
    if not src.is_dir():
        raise FileNotFoundError(f"local_dir does not exist: {src}")
    files = sorted(p.name for p in src.iterdir() if p.is_file())
    manifest = {"repo_id": repo_id, "files": files, "dry_run": dry_run}

    if dry_run:
        logger.info("push_to_hub --dry-run: would upload %s -> %s", files, repo_id)
        return manifest

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "push_to_hub requires `huggingface_hub`. "
            "Install with: pip install huggingface_hub"
        ) from exc

    if token is None:
        raise RuntimeError(
            "push_to_hub requires an HF token (HF_TOKEN env var or token=...). "
            "Pass dry_run=True for a manifest-only check."
        )

    api = HfApi(token=token)
    api.upload_folder(folder_path=str(src), repo_id=repo_id)
    logger.info("push_to_hub uploaded %s -> %s", files, repo_id)
    return manifest


def pull_latest_checkpoint(
    repo_id: str,
    local_dir: str | Path,
    token: str | None = None,
) -> Path:
    """Download the latest checkpoint snapshot from HF Hub into ``local_dir``."""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "pull_latest_checkpoint requires `huggingface_hub`. "
            "Install with: pip install huggingface_hub"
        ) from exc

    target = Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    out = snapshot_download(repo_id=repo_id, local_dir=str(target), token=token)
    return Path(out)


def push_to_kaggle_dataset(
    local_dir: str | Path,
    dataset_id: str,
) -> None:
    """Mirror checkpoints to a Kaggle Dataset.

    Stubbed in M9 -- Phase F wires up ``kaggle.api.dataset_create_version``
    once we have real checkpoints to push. Raises rather than no-op'ing so a
    caller that depends on the mirror sees a loud failure (CLAUDE.md sec 11).
    """
    raise NotImplementedError(
        "Kaggle Dataset mirror is wired in Phase F; "
        "for M9 use push_to_hub() with the HF Hub primary."
    )
