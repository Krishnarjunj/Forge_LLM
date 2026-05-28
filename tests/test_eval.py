"""Tests for byte-level perplexity (M10).

Spec: ``docs/04_roadmap.md`` M10 -- "tests for byte-PPL math on a known small
case." Byte-PPL normalises by raw UTF-8 byte count rather than token count so
that different tokenizers on the same text yield comparable numbers
(tokenizer-agnostic, which is the point of byte-level evaluation).

* ``test_byte_ppl_uniform_logits_matches_closed_form`` -- a model that
  outputs zero logits gives a uniform distribution over the vocabulary, so
  per-token NLL is exactly ``log(V)``. The byte-PPL for a text of ``T``
  tokens scoring ``T-1`` predicted positions over ``B`` bytes must equal
  ``exp((T-1) * log(V) / B)``.
* ``test_byte_ppl_untrained_model_is_finite_and_positive`` -- the eval path
  runs end-to-end on an untrained ``ForgeForCausalLM`` and produces a
  finite, >1 PPL.
* ``test_byte_ppl_returns_inf_on_empty_input`` -- no bytes -> ``inf``;
  divides by zero must not crash.
* ``test_byte_ppl_lower_after_overfit`` -- after a handful of training steps
  on a tiny corpus, byte-PPL on the same corpus drops compared to the
  untrained baseline. Loose sanity ("strictly less") not a tight bound.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from forge_llm.config import ForgeConfig
from forge_llm.eval import eval_byte_perplexity
from forge_llm.model import ForgeForCausalLM
from forge_llm.tokenizer import BPETokenizer
from forge_llm.train import TrainConfig, Trainer


class _UniformLogitsModel(nn.Module):
    """Stub model that returns zero logits -> uniform distribution over vocab."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, x: Tensor) -> Tensor:
        batch, seq = x.shape
        return torch.zeros(batch, seq, self.vocab_size)


def _tokenizer() -> BPETokenizer:
    corpus = "Byte-level perplexity is tokenizer-agnostic by design. " * 40
    return BPETokenizer.train(corpus, vocab_size=512)


def test_byte_ppl_uniform_logits_matches_closed_form() -> None:
    """Uniform-logits model: byte-PPL == exp((T - 1) * log(V) / B) exactly."""
    tokenizer = _tokenizer()
    vocab_size = tokenizer.vocab_size
    model = _UniformLogitsModel(vocab_size)

    text = "Byte-level perplexity is tokenizer-agnostic by design."
    tokens = tokenizer.encode(text)
    assert len(tokens) >= 2, "test text must tokenize to >= 2 tokens"

    ppl = eval_byte_perplexity(
        model,
        tokenizer,
        [text],
        seq_len=len(tokens) + 4,  # fits in one chunk
    )

    bytes_count = len(text.encode("utf-8"))
    expected = math.exp((len(tokens) - 1) * math.log(vocab_size) / bytes_count)
    assert math.isfinite(ppl)
    torch.testing.assert_close(torch.tensor(ppl), torch.tensor(expected), rtol=1e-5, atol=1e-5)


def test_byte_ppl_untrained_model_is_finite_and_positive() -> None:
    """An untrained Forge model produces a finite, >1 PPL on a small text."""
    tokenizer = _tokenizer()
    model_config = ForgeConfig(
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
    model = ForgeForCausalLM(model_config)

    text = "Byte-level perplexity is tokenizer-agnostic by design."
    ppl = eval_byte_perplexity(model, tokenizer, [text], seq_len=64)
    assert math.isfinite(ppl)
    assert ppl > 1.0, f"untrained PPL must exceed 1.0, got {ppl}"


def test_byte_ppl_returns_inf_on_empty_input() -> None:
    """No bytes -> ``inf`` rather than a divide-by-zero crash."""
    tokenizer = _tokenizer()
    model = _UniformLogitsModel(tokenizer.vocab_size)
    ppl = eval_byte_perplexity(model, tokenizer, [], seq_len=32)
    assert ppl == float("inf")


def test_byte_ppl_lower_after_overfit() -> None:
    """After a few overfit steps on a single corpus, PPL on that corpus drops."""
    tokenizer = _tokenizer()
    model_config = ForgeConfig(
        name="test-tiny",
        n_layer=2,
        d_model=32,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        d_ff=64,
        vocab_size=tokenizer.vocab_size,
        max_seq=32,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_embeddings=True,
        init_std=0.02,
        dropout=0.0,
    )
    train_config = TrainConfig(
        lr=1e-2,
        seq_len=32,
        micro_bs=2,
        grad_accum=1,
        warmup_steps=0,
        total_steps=50,
        min_lr=1e-3,
        seed=0,
        dtype="fp32",
    )
    docs = ["The cat sat on the mat. The mat was warm and cosy."] * 400
    text = "The cat sat on the mat."

    untrained = ForgeForCausalLM(model_config)
    ppl_before = eval_byte_perplexity(untrained, tokenizer, [text], seq_len=32)

    trainer = Trainer(
        model_config=model_config,
        train_config=train_config,
        data_factory=lambda: iter(docs),
        tokenizer=tokenizer,
        device="cpu",
    )
    trainer.train_steps(50)
    ppl_after = eval_byte_perplexity(trainer.model, tokenizer, [text], seq_len=32)

    assert math.isfinite(ppl_after)
    assert ppl_after < ppl_before, (
        f"PPL did not drop after overfitting: before={ppl_before:.2f}, "
        f"after={ppl_after:.2f}. Either training is broken or eval is broken."
    )
