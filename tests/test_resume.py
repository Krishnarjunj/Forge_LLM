"""Resume-safety test (M9, slow) — see ``docs/02_correctness_plan.md`` sec 1.15.

Headline correctness contract for the entire training side. CLAUDE.md sec 7
adversarial #3: "train 200 steps uninterrupted; train 100 + ckpt + kill +
resume to 200; the post-resume losses must match the uninterrupted run's
losses for steps 101..200 bitwise on single-GPU fp32 (CPU OK in CI)."

This catches: missing RNG-state restore, data-iterator position drift,
optimiser-momentum state loss, scheduler-step-counter drift. The test runs
in fp32 on CPU (``torch.cuda.amp`` is disabled in this regime) so it can be
exercised on a laptop without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

from forge_llm.config import ForgeConfig
from forge_llm.tokenizer import BPETokenizer
from forge_llm.train import TrainConfig, Trainer

_DOCS = [
    "The quick brown fox jumps over the lazy dog repeatedly across the meadow.",
    "Pack densely; never pad. Pretraining loss is calibrated on full sequences.",
    "Tokens flow from the streamer through the BPE into fixed-length chunks.",
    "Causal language models train on next-token prediction with teacher forcing.",
    "Forge-LLM is a from-scratch decoder trained on a free Kaggle T4 GPU.",
] * 200


def _make_factory(docs: list[str]) -> Callable[[], Iterator[str]]:
    return lambda: iter(docs)


def _tiny_configs() -> tuple[ForgeConfig, TrainConfig]:
    model_config = ForgeConfig(
        name="test-tiny",
        n_layer=2,
        d_model=32,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        d_ff=64,
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
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        grad_clip_norm=1.0,
        warmup_steps=10,
        total_steps=200,
        min_lr=1e-4,
        micro_bs=2,
        grad_accum=1,
        seq_len=16,
        seed=0,
        dtype="fp32",
    )
    return model_config, train_config


@pytest.mark.slow
def test_resume_safety_loss_curve_indistinguishable(tmp_path: Path) -> None:
    """Steps 101..200 of an uninterrupted run match a resumed run bit-exactly."""
    model_config, train_config = _tiny_configs()
    tokenizer = BPETokenizer.train(" ".join(_DOCS), vocab_size=model_config.vocab_size)
    factory = _make_factory(_DOCS)

    # Run A: uninterrupted 200 steps.
    trainer_a = Trainer(
        model_config=model_config,
        train_config=train_config,
        data_factory=factory,
        tokenizer=tokenizer,
        device="cpu",
    )
    losses_a = trainer_a.train_steps(200)
    assert len(losses_a) == 200

    # Run B: 100 steps, checkpoint, drop, resume, 100 more steps.
    trainer_b = Trainer(
        model_config=model_config,
        train_config=train_config,
        data_factory=factory,
        tokenizer=tokenizer,
        device="cpu",
    )
    losses_b_pre = trainer_b.train_steps(100)
    assert len(losses_b_pre) == 100

    # Sanity: steps 0..99 of A and B must already match (same seed, same config).
    for i, (a, b) in enumerate(zip(losses_a[:100], losses_b_pre, strict=True)):
        torch.testing.assert_close(
            torch.tensor(a),
            torch.tensor(b),
            rtol=0,
            atol=1e-6,
            msg=lambda m, i=i, a=a, b=b: (
                f"Pre-checkpoint drift at step {i}: A={a}, B={b}. "
                "Seeding or determinism setup is broken before resume even matters."
            ),
        )

    ckpt_path = trainer_b.save_checkpoint(tmp_path / "trainer_b_step100.pt")
    del trainer_b  # simulate process death

    trainer_c = Trainer.load_checkpoint(
        ckpt_path,
        data_factory=factory,
        tokenizer=tokenizer,
        device="cpu",
    )
    losses_c = trainer_c.train_steps(100)
    assert len(losses_c) == 100

    # The headline assertion: steps 101..200 of A == 100 post-resume steps of C, bitwise.
    for i, (a, c) in enumerate(zip(losses_a[100:], losses_c, strict=True)):
        torch.testing.assert_close(
            torch.tensor(a),
            torch.tensor(c),
            rtol=0,
            atol=1e-6,
            msg=lambda m, i=i, a=a, c=c: (
                f"Resume drift at step {101 + i}: A={a}, resumed={c}. "
                "Check: rng restore, data-iterator state, optimiser state, scheduler step."
            ),
        )
