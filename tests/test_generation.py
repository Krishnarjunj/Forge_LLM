"""Tests for the generation iterator (M11).

Spec: ``docs/02_correctness_plan.md`` sec 1.13. Four cases:

* ``test_generate_deterministic_with_seed`` -- two ``generate(..., seed=0)``
  calls produce byte-identical token sequences. Catches hidden RNG.
* ``test_generate_uses_cache_by_default`` -- ``use_cache=True`` is the
  default, and the attention layer receives a non-None cache argument when
  it's set. Mock-based.
* ``test_generate_stops_on_eos`` -- force-feed an EOS token in the sampling
  stream; the iterator halts. Catches an EOS that is silently ignored.
* ``test_generate_load_checkpoint_and_emit_100_tokens`` (slow) -- save a tiny
  trainer, reload via from_pretrained, ``generate`` 100 tokens twice with
  the same seed; byte-identical sequences. End-to-end regression.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from forge_llm.config import ForgeConfig
from forge_llm.generation import generate
from forge_llm.hub import from_pretrained, save_pretrained
from forge_llm.model import ForgeForCausalLM
from forge_llm.tokenizer import BPETokenizer
from forge_llm.train import TrainConfig, Trainer


def _tokenizer() -> BPETokenizer:
    corpus = "Forge generates tokens one at a time, with a KV cache for speed. " * 40
    return BPETokenizer.train(corpus, vocab_size=512)


def _tiny_model(tokenizer: BPETokenizer) -> ForgeForCausalLM:
    cfg = ForgeConfig(
        name="test-tiny",
        n_layer=2,
        d_model=32,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        d_ff=64,
        vocab_size=tokenizer.vocab_size,
        max_seq=64,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_embeddings=True,
        init_std=0.02,
        dropout=0.0,
    )
    return ForgeForCausalLM(cfg).eval()


def test_generate_deterministic_with_seed() -> None:
    tokenizer = _tokenizer()
    model = _tiny_model(tokenizer)
    prompt = "Forge generates"

    out_a = list(generate(model, tokenizer, prompt, max_new=20, seed=0))
    out_b = list(generate(model, tokenizer, prompt, max_new=20, seed=0))
    assert out_a == out_b, "same seed must produce byte-identical token stream"
    assert len(out_a) > 0


def test_generate_uses_cache_by_default() -> None:
    """Default ``use_cache=True`` -> attention.forward receives a non-None cache."""
    tokenizer = _tokenizer()
    model = _tiny_model(tokenizer)

    saw_cache: list[bool] = []
    real_forward = type(model.model.layers[0].attn).forward

    def wrapped(self, x, freqs_cis=None, cache=None, input_pos=None):  # type: ignore[no-untyped-def]
        saw_cache.append(cache is not None)
        return real_forward(
            self, x, freqs_cis=freqs_cis, cache=cache, input_pos=input_pos
        )

    with patch.object(
        type(model.model.layers[0].attn), "forward", wrapped
    ):
        list(generate(model, tokenizer, "Forge", max_new=3, seed=0))

    assert saw_cache, "attention was never called"
    assert all(saw_cache), (
        "attention.forward saw cache=None somewhere; cache silently disabled"
    )


def test_generate_stops_on_eos() -> None:
    """Force an EOS in the sampling stream; the iterator must stop early."""
    tokenizer = _tokenizer()
    model = _tiny_model(tokenizer)

    # Patch sampling: always pick the EOS token id, so the very first
    # generated token should halt the iterator.
    eos_id = tokenizer.eos_id

    def _fixed_sampler(logits, generator=None):  # type: ignore[no-untyped-def]
        return torch.tensor([eos_id], dtype=torch.long)

    with patch("forge_llm.generation._sample_token", _fixed_sampler):
        out = list(generate(model, tokenizer, "Forge", max_new=20, seed=0))
    # Iterator halted before max_new=20.
    assert len(out) == 0, (
        f"expected 0 emitted tokens (EOS on first sample), got {len(out)}: {out!r}"
    )


@pytest.mark.slow
def test_generate_load_checkpoint_and_emit_100_tokens(tmp_path: Path) -> None:
    """Save a trainer, reload, generate 100 tokens twice with seed=0; byte-identical."""
    tokenizer = _tokenizer()
    cfg = ForgeConfig(
        name="test-tiny",
        n_layer=2,
        d_model=32,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        d_ff=64,
        vocab_size=tokenizer.vocab_size,
        max_seq=64,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_embeddings=True,
        init_std=0.02,
        dropout=0.0,
    )
    train_cfg = TrainConfig(
        lr=1e-3,
        seq_len=32,
        micro_bs=2,
        grad_accum=1,
        warmup_steps=0,
        total_steps=10,
        seed=0,
        dtype="fp32",
    )
    docs = ["Forge tokens flow steadily through the cache layer."] * 60
    trainer = Trainer(
        model_config=cfg,
        train_config=train_cfg,
        data_factory=lambda: iter(docs),
        tokenizer=tokenizer,
        device="cpu",
    )
    trainer.train_steps(5)
    save_pretrained(trainer, tmp_path)

    reloaded = from_pretrained(tmp_path, data_factory=lambda: iter(docs))
    out_a = list(generate(reloaded.model, tokenizer, "Forge", max_new=100, seed=0))
    out_b = list(generate(reloaded.model, tokenizer, "Forge", max_new=100, seed=0))
    assert out_a == out_b
    assert len(out_a) > 0
