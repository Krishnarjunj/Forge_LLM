"""Tests for the statically-allocated KV cache (M11).

Spec: ``docs/02_correctness_plan.md`` sec 1.6 (three unit cases) plus the
headline equivalence test in sec 1.8 (cached token-by-token forward must
match the full-sequence forward at fp32 tolerance).

* ``test_kvcache_shape`` -- buffer shape is ``(max_batch, n_kv_head, max_seq,
  head_dim)``; catches the wrong allocation shape.
* ``test_kvcache_update_correctly_overwrites_positions`` -- ``update`` writes
  at the integer positions in ``input_pos`` exactly; an off-by-one or
  ``index_copy_`` arg swap fails.
* ``test_kvcache_does_not_grow_unboundedly`` -- 16 calls to ``update`` on a
  ``max_seq=16`` cache leave the buffer shape unchanged; catches an
  accidental ``cat`` instead of indexed write (which would silent-OOM on
  long generation).
* ``test_kvcache_full_vs_token_by_token_equivalence`` (the headline) --
  feed an entire sequence one token at a time through the model with a
  KVCache; the resulting logits must match the no-cache full-sequence
  forward at ``rtol=atol=1e-5``.
"""

from __future__ import annotations

import torch

from forge_llm.cache import KVCache
from forge_llm.config import ForgeConfig
from forge_llm.model import ForgeForCausalLM

_MAX_BATCH = 2
_MAX_SEQ = 16
_N_KV_HEAD = 2
_HEAD_DIM = 16


def test_kvcache_shape() -> None:
    cache = KVCache(
        max_batch=_MAX_BATCH,
        max_seq=_MAX_SEQ,
        n_kv_head=_N_KV_HEAD,
        head_dim=_HEAD_DIM,
    )
    assert cache.k_cache.shape == (_MAX_BATCH, _N_KV_HEAD, _MAX_SEQ, _HEAD_DIM)
    assert cache.v_cache.shape == (_MAX_BATCH, _N_KV_HEAD, _MAX_SEQ, _HEAD_DIM)


def test_kvcache_update_correctly_overwrites_positions() -> None:
    """Write at [0,1,2], then [3,4]; verify each slice exactly equals what was written."""
    cache = KVCache(
        max_batch=_MAX_BATCH,
        max_seq=_MAX_SEQ,
        n_kv_head=_N_KV_HEAD,
        head_dim=_HEAD_DIM,
    )

    k1 = torch.randn(_MAX_BATCH, _N_KV_HEAD, 3, _HEAD_DIM)
    v1 = torch.randn(_MAX_BATCH, _N_KV_HEAD, 3, _HEAD_DIM)
    cache.update(torch.tensor([0, 1, 2]), k1, v1)
    assert torch.equal(cache.k_cache[:, :, 0:3, :], k1)
    assert torch.equal(cache.v_cache[:, :, 0:3, :], v1)

    k2 = torch.randn(_MAX_BATCH, _N_KV_HEAD, 2, _HEAD_DIM)
    v2 = torch.randn(_MAX_BATCH, _N_KV_HEAD, 2, _HEAD_DIM)
    cache.update(torch.tensor([3, 4]), k2, v2)
    assert torch.equal(cache.k_cache[:, :, 3:5, :], k2)
    assert torch.equal(cache.v_cache[:, :, 3:5, :], v2)

    # The earlier slice must remain untouched.
    assert torch.equal(cache.k_cache[:, :, 0:3, :], k1)


def test_kvcache_does_not_grow_unboundedly() -> None:
    """Sixteen update() calls leave the buffer shape unchanged at max_seq=16."""
    cache = KVCache(
        max_batch=_MAX_BATCH,
        max_seq=_MAX_SEQ,
        n_kv_head=_N_KV_HEAD,
        head_dim=_HEAD_DIM,
    )
    for t in range(_MAX_SEQ):
        k = torch.randn(_MAX_BATCH, _N_KV_HEAD, 1, _HEAD_DIM)
        v = torch.randn(_MAX_BATCH, _N_KV_HEAD, 1, _HEAD_DIM)
        cache.update(torch.tensor([t]), k, v)
    assert cache.k_cache.shape == (_MAX_BATCH, _N_KV_HEAD, _MAX_SEQ, _HEAD_DIM)
    assert cache.v_cache.shape == (_MAX_BATCH, _N_KV_HEAD, _MAX_SEQ, _HEAD_DIM)


def test_kvcache_full_vs_token_by_token_equivalence() -> None:
    """The second headline correctness test (CLAUDE.md sec 7 adversarial #2).

    A full-sequence forward (no cache) and a token-by-token forward (with
    cache) must produce the same logits within fp32 tolerance. Catches
    every variant of "cached path silently diverged" -- wrong cache update,
    missing rotary at cached positions, wrong attention slice, etc.
    """
    cfg = ForgeConfig(
        name="test-tiny",
        n_layer=2,
        d_model=32,
        n_head=4,
        n_kv_head=2,
        head_dim=8,
        d_ff=64,
        vocab_size=256,
        max_seq=16,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_embeddings=True,
        init_std=0.02,
        dropout=0.0,
    )
    model = ForgeForCausalLM(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq))

    with torch.no_grad():
        logits_full = model(tokens)

        cache = KVCache.allocate(cfg, max_batch=1)
        per_token = []
        for t in range(cfg.max_seq):
            out = model(
                tokens[:, t : t + 1],
                cache=cache,
                input_pos=torch.tensor([t]),
            )
            per_token.append(out)
        logits_cached = torch.cat(per_token, dim=1)

    torch.testing.assert_close(logits_full, logits_cached, rtol=1e-5, atol=1e-5)
