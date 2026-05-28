"""Tests for the ``forge-llm`` console entrypoint (M12).

Three smoke tests covering the M12 exit criteria:

* ``test_cli_help_lists_all_subcommands`` -- ``forge-llm --help`` exits 0 and
  the four subcommand names appear in the output.
* ``test_cli_generate_emits_nonempty_output`` -- save a tiny checkpoint via
  ``hub.save_pretrained``, invoke ``forge-llm generate ... --checkpoint
  <dir>`` and verify stdout is non-empty. Verifies the end-to-end checkpoint
  -> tokenizer -> generate path.
* ``test_cli_bench_cache_writes_table`` -- ``forge-llm bench-cache --small
  --warmup 0 --out ...`` runs and writes a Markdown table.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_llm.cli import main as cli_main
from forge_llm.config import ForgeConfig
from forge_llm.hub import save_pretrained
from forge_llm.tokenizer import BPETokenizer
from forge_llm.train import TrainConfig, Trainer


def test_cli_help_lists_all_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for sub in ("generate", "train", "eval", "bench-cache"):
        assert sub in out, f"--help output missing subcommand {sub!r}: {out!r}"


def test_cli_generate_emits_nonempty_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Save a tiny checkpoint, invoke `forge-llm generate`, expect text on stdout."""
    model_cfg = ForgeConfig(
        name="cli-tiny",
        n_layer=2,
        d_model=16,
        n_head=2,
        n_kv_head=1,
        head_dim=8,
        d_ff=32,
        vocab_size=512,
        max_seq=32,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_embeddings=True,
        init_std=0.02,
        dropout=0.0,
    )
    train_cfg = TrainConfig(
        lr=1e-2,
        seq_len=16,
        micro_bs=2,
        grad_accum=1,
        warmup_steps=0,
        total_steps=20,
        seed=0,
        dtype="fp32",
    )
    # Diverse corpus so the BPE actually fills most of vocab_size (otherwise
    # un-learned slots decode to empty bytes and the CLI prints nothing).
    docs = [
        "The quick brown fox jumps over the lazy dog repeatedly across the meadow.",
        "Packed datasets concatenate documents with EOS separators between them.",
        "Causal language models train on next-token prediction with teacher forcing.",
        "Forge-LLM is a from-scratch decoder trained on a free Kaggle T4 GPU.",
        "Resume safety means the loss curve after restart matches uninterrupted.",
        "Tokens flow from the streamer through the BPE into fixed-length chunks.",
    ] * 50
    tokenizer = BPETokenizer.train(" ".join(docs), vocab_size=model_cfg.vocab_size)
    trainer = Trainer(
        model_config=model_cfg,
        train_config=train_cfg,
        data_factory=lambda: iter(docs),
        tokenizer=tokenizer,
        device="cpu",
    )
    # Brief overfit so the sampler biases toward the filled vocab slots and
    # produces non-empty decoded text rather than padded-byte placeholders.
    trainer.train_steps(20)
    save_pretrained(trainer, tmp_path)

    rc = cli_main(
        [
            "generate",
            "Hello",
            "--checkpoint",
            str(tmp_path),
            "--max-new",
            "5",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() != "", "forge-llm generate produced no output on stdout"


def test_cli_bench_cache_writes_table(tmp_path: Path) -> None:
    """`forge-llm bench-cache --small ...` writes a Markdown table to --out."""
    out_path = tmp_path / "bench.md"
    rc = cli_main(
        [
            "bench-cache",
            "--small",
            "--ctx-lengths",
            "16",
            "--max-seq",
            "32",
            "--warmup",
            "0",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists() and out_path.stat().st_size > 0
    body = out_path.read_text()
    assert "| ctx | no-cache tok/s |" in body, (
        f"bench-cache output missing the speedup table header: {body!r}"
    )
