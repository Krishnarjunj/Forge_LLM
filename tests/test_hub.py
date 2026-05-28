"""Tests for the HF Hub / Kaggle Dataset checkpoint helpers (M9).

The M9 exit criteria require a tested ``--dry-run`` mode on the Hub upload
helper so CI can exercise the path without an ``HF_TOKEN`` secret. We also
round-trip ``save_pretrained`` -> ``from_pretrained`` to confirm the artefact
layout is correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_llm.config import ForgeConfig
from forge_llm.hub import (
    from_pretrained,
    push_to_hub,
    push_to_kaggle_dataset,
    save_pretrained,
)
from forge_llm.tokenizer import BPETokenizer
from forge_llm.train import TrainConfig, Trainer


def _tiny_trainer() -> Trainer:
    model_config = ForgeConfig(
        name="test-tiny",
        n_layer=2,
        d_model=16,
        n_head=2,
        n_kv_head=1,
        head_dim=8,
        d_ff=32,
        vocab_size=512,
        max_seq=16,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_embeddings=True,
        init_std=0.02,
        dropout=0.0,
    )
    train_config = TrainConfig(
        lr=1e-3,
        seq_len=16,
        micro_bs=2,
        grad_accum=1,
        warmup_steps=0,
        total_steps=10,
        seed=0,
        dtype="fp32",
    )
    docs = ["hub test data " * 20] * 50
    tokenizer = BPETokenizer.train(" ".join(docs), vocab_size=model_config.vocab_size)
    factory = lambda: iter(docs)  # noqa: E731 -- closure capture is clearest as a lambda
    return Trainer(
        model_config=model_config,
        train_config=train_config,
        data_factory=factory,
        tokenizer=tokenizer,
        device="cpu",
    )


def test_push_to_hub_dry_run_returns_manifest_without_network(tmp_path: Path) -> None:
    """``dry_run=True`` returns a manifest with no network call and no token."""
    (tmp_path / "checkpoint.pt").write_bytes(b"fake-ckpt")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")

    manifest = push_to_hub(tmp_path, repo_id="user/forge-test", dry_run=True)

    assert manifest["dry_run"] is True
    assert manifest["repo_id"] == "user/forge-test"
    assert sorted(manifest["files"]) == [
        "checkpoint.pt",
        "config.json",
        "tokenizer.json",
    ]


def test_push_to_hub_without_token_or_dry_run_fails_loudly(tmp_path: Path) -> None:
    """A non-dry-run push without a token raises a clear RuntimeError (CLAUDE.md sec 11)."""
    (tmp_path / "checkpoint.pt").write_bytes(b"fake-ckpt")
    with pytest.raises(RuntimeError, match="HF token"):
        push_to_hub(tmp_path, repo_id="user/forge-test", dry_run=False)


def test_push_to_kaggle_dataset_stub_raises() -> None:
    """The Kaggle mirror is a stub in M9; calling it must fail loudly."""
    with pytest.raises(NotImplementedError, match="Phase F"):
        push_to_kaggle_dataset("/tmp/x", dataset_id="user/forge-resume")


def test_save_pretrained_roundtrip(tmp_path: Path) -> None:
    """``save_pretrained`` artefacts re-load via ``from_pretrained`` and resume training."""
    trainer = _tiny_trainer()
    losses_before = trainer.train_steps(3)
    save_pretrained(trainer, tmp_path)
    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "tokenizer.json").exists()

    docs = ["hub test data " * 20] * 50
    factory = lambda: iter(docs)  # noqa: E731
    reloaded = from_pretrained(tmp_path, data_factory=factory, device="cpu")
    losses_after = reloaded.train_steps(3)
    assert len(losses_before) == 3
    assert len(losses_after) == 3
