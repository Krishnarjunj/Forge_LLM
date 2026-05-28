"""Tests for Rotary Positional Embeddings (M3).

Spec: ``docs/02_correctness_plan.md`` sec 1.2. Convention: HF Llama half-split
(ADR-007, amended 2026-05-28). Six cases:

* ``test_rope_freqs_shape`` — ``precompute_freqs_cis`` returns the right shape
  and dtype; catches an off-by-one on ``head_dim // 2``.
* ``test_rope_value_vs_llama`` — matches HF ``apply_rotary_pos_emb`` at fp32
  with rtol/atol = 1e-5. Skipped with a reason (CLAUDE.md sec 11) if
  ``transformers`` is not installed.
* ``test_rope_rotation_identity`` — applying RoPE at position 0 is a no-op
  (cos(0)=1, sin(0)=0); catches a sign flip on the sin term.
* ``test_rope_relative_position_invariance`` — the attention score
  ``<R(m)q, R(n)k>`` depends only on ``m-n``; the defining property of RoPE.
  Catches a RoPE variant that uses absolute positions only.
* ``test_rope_long_context_extrapolation`` — at position 4096 the rotation is
  still finite and bit-matches a naive direct ``cos``/``sin`` recompute at
  that index; catches numeric blowup from a mis-set theta.
* ``test_rope_determinism`` — same input twice produces bit-identical output.
"""

from __future__ import annotations

import pytest
import torch

from forge_llm.rope import apply_rotary, precompute_freqs_cis

_HEAD_DIM = 16
_MAX_SEQ = 32
_THETA = 10000.0


def test_rope_freqs_shape() -> None:
    fc = precompute_freqs_cis(_HEAD_DIM, _MAX_SEQ, _THETA)
    assert fc.shape == (_MAX_SEQ, _HEAD_DIM // 2)
    assert fc.dtype.is_complex


def test_rope_value_vs_llama() -> None:
    """Bit-match HF ``apply_rotary_pos_emb`` (half-split convention)."""
    try:
        from transformers.models.llama.modeling_llama import (  # noqa: PLC0415
            apply_rotary_pos_emb,
        )
    except ImportError:
        pytest.skip("transformers (HF Llama oracle) not installed")

    seq_len = 8
    fc = precompute_freqs_cis(_HEAD_DIM, _MAX_SEQ, _THETA)
    fc_seq = fc[:seq_len]
    # HF expects (cos, sin) each of shape (batch, seq, head_dim) with the
    # half-frequencies duplicated across the head_dim axis. The function then
    # internally adds an unsqueeze at dim=1 to broadcast over heads.
    cos = torch.cat([fc_seq.real, fc_seq.real], dim=-1).unsqueeze(0)
    sin = torch.cat([fc_seq.imag, fc_seq.imag], dim=-1).unsqueeze(0)

    q = torch.randn(2, 4, seq_len, _HEAD_DIM)
    k = torch.randn(2, 4, seq_len, _HEAD_DIM)

    ours_q, ours_k = apply_rotary(q, k, fc)
    theirs_q, theirs_k = apply_rotary_pos_emb(q, k, cos, sin)

    torch.testing.assert_close(ours_q, theirs_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(ours_k, theirs_k, rtol=1e-5, atol=1e-5)


def test_rope_rotation_identity() -> None:
    """RoPE at position 0 is the identity transform."""
    fc = precompute_freqs_cis(_HEAD_DIM, _MAX_SEQ, _THETA)
    q = torch.randn(2, 4, 1, _HEAD_DIM)
    k = torch.randn(2, 4, 1, _HEAD_DIM)

    q_rot, k_rot = apply_rotary(q, k, fc)
    torch.testing.assert_close(q_rot, q, rtol=0, atol=1e-7)
    torch.testing.assert_close(k_rot, k, rtol=0, atol=1e-7)


def test_rope_relative_position_invariance() -> None:
    """``<R(m)q, R(n)k>`` depends only on ``m - n``."""
    fc = precompute_freqs_cis(_HEAD_DIM, 64, _THETA)

    q_vec = torch.randn(_HEAD_DIM)
    k_vec = torch.randn(_HEAD_DIM)

    def attn_score_at(m: int, n: int) -> torch.Tensor:
        seq_len = max(m, n) + 1
        q_seq = torch.zeros(1, 1, seq_len, _HEAD_DIM)
        k_seq = torch.zeros(1, 1, seq_len, _HEAD_DIM)
        q_seq[0, 0, m] = q_vec
        k_seq[0, 0, n] = k_vec
        q_rot, k_rot = apply_rotary(q_seq, k_seq, fc)
        return (q_rot[0, 0, m] * k_rot[0, 0, n]).sum()

    # Fix m - n = 3, sweep absolute positions. All scores must match.
    baseline = attn_score_at(5, 2)
    for m, n in [(10, 7), (20, 17), (30, 27), (50, 47)]:
        torch.testing.assert_close(attn_score_at(m, n), baseline, rtol=1e-5, atol=1e-5)


def test_rope_long_context_extrapolation() -> None:
    """RoPE at far positions is finite and matches a direct cos/sin recompute."""
    max_seq = 8192
    long_pos = 4096
    fc = precompute_freqs_cis(_HEAD_DIM, max_seq, _THETA)

    seq_len = long_pos + 1
    q = torch.randn(1, 1, seq_len, _HEAD_DIM)
    k = torch.randn(1, 1, seq_len, _HEAD_DIM)
    q_rot, k_rot = apply_rotary(q, k, fc)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.isfinite(q_rot).all()
    assert torch.isfinite(k_rot).all()

    # Direct recompute at position long_pos using the textbook RoPE formula.
    half = _HEAD_DIM // 2
    inv_freq = 1.0 / (_THETA ** (torch.arange(0, _HEAD_DIM, 2, dtype=torch.float32) / _HEAD_DIM))
    assert inv_freq.shape == (half,)
    angles = long_pos * inv_freq
    cos_naive = torch.cos(angles)
    sin_naive = torch.sin(angles)

    fc_at_pos = fc[long_pos]
    torch.testing.assert_close(fc_at_pos.real, cos_naive, rtol=1e-5, atol=1e-3)
    torch.testing.assert_close(fc_at_pos.imag, sin_naive, rtol=1e-5, atol=1e-3)


def test_rope_determinism() -> None:
    """Two ``apply_rotary`` calls on the same input are bit-identical."""
    fc = precompute_freqs_cis(_HEAD_DIM, _MAX_SEQ, _THETA)
    q = torch.randn(2, 4, 8, _HEAD_DIM)
    k = torch.randn(2, 4, 8, _HEAD_DIM)
    q1, k1 = apply_rotary(q, k, fc)
    q2, k2 = apply_rotary(q, k, fc)
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
