"""Tests for attention (M4 MHA path, M5 GQA path).

Spec: ``docs/02_correctness_plan.md`` sec 1.3 (MHA) and sec 1.4 (GQA). This
file lands the five MHA cases in M4; the GQA cases follow in M5 and extend
this file.

* ``test_mha_shape`` — output is ``(B, T, d_model)`` on a standard config.
* ``test_mha_value_vs_torch_reference`` — bit-matches
  ``torch.nn.MultiheadAttention`` (used as the M4 oracle per CLAUDE.md sec 2)
  with weights copied across, at rtol/atol = 1e-5. The torch reference is
  forbidden in ``src/`` but explicitly allowed in ``tests/``.
* ``test_mha_causal_no_leak`` — mutating ``x[:, T-1, :]`` does not change
  ``y[:, :T-1, :]`` in any bit. Catches future-token leakage through the
  causal mask. (CLAUDE.md sec 7 adversarial #1.)
* ``test_mha_backward`` — ``torch.autograd.gradcheck`` on a tiny fp64 forward
  catches a broken gradient route through the softmax / matmul / projection.
* ``test_mha_determinism`` — two forwards on the same input must be bit-equal.
"""

from __future__ import annotations

import torch

from forge_llm.attention import GroupedQueryAttention

_D_MODEL = 64
_N_HEAD = 4
_HEAD_DIM = 16
_MAX_SEQ = 128


def _make_mha() -> GroupedQueryAttention:
    return GroupedQueryAttention(
        d_model=_D_MODEL,
        n_head=_N_HEAD,
        n_kv_head=_N_HEAD,
        head_dim=_HEAD_DIM,
        max_seq=_MAX_SEQ,
    )


def test_mha_shape() -> None:
    mha = _make_mha()
    x = torch.randn(2, 8, _D_MODEL)
    y = mha(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_mha_value_vs_torch_reference() -> None:
    """Match ``torch.nn.MultiheadAttention`` with identical weights and a causal mask."""
    mha = _make_mha()

    oracle = torch.nn.MultiheadAttention(
        embed_dim=_D_MODEL,
        num_heads=_N_HEAD,
        batch_first=True,
        bias=False,
        dropout=0.0,
    )
    with torch.no_grad():
        oracle.in_proj_weight.copy_(
            torch.cat([mha.wq.weight, mha.wk.weight, mha.wv.weight], dim=0)
        )
        oracle.out_proj.weight.copy_(mha.wo.weight)

    seq_len = 8
    x = torch.randn(2, seq_len, _D_MODEL)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf")), diagonal=1
    )

    ours = mha(x)
    theirs, _ = oracle(x, x, x, attn_mask=causal_mask, need_weights=False)

    torch.testing.assert_close(ours, theirs, rtol=1e-5, atol=1e-5)


def test_mha_causal_no_leak() -> None:
    """Mutating token T-1 must not change outputs at positions t < T-1."""
    mha = _make_mha()
    batch, seq_len = 2, 8
    x = torch.randn(batch, seq_len, _D_MODEL)
    y_orig = mha(x)

    x_mut = x.clone()
    x_mut[:, seq_len - 1, :] = torch.randn(batch, _D_MODEL)
    y_mut = mha(x_mut)

    assert torch.equal(
        y_orig[:, : seq_len - 1, :], y_mut[:, : seq_len - 1, :]
    ), "Causal mask leak: output at t < T-1 changed after mutating x[T-1]"


def test_mha_backward() -> None:
    """``gradcheck`` on a tiny fp64 forward catches bad gradient routing."""
    mha = GroupedQueryAttention(
        d_model=8, n_head=2, n_kv_head=2, head_dim=4, max_seq=16
    ).double()
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mha, (x,))


def test_mha_determinism() -> None:
    """Two forwards through the same module on the same input are bit-equal."""
    mha = _make_mha()
    x = torch.randn(2, 8, _D_MODEL)
    y1 = mha(x)
    y2 = mha(x)
    assert torch.equal(y1, y2)
