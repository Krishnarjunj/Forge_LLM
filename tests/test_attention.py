"""Tests for attention (M4 MHA path, M5 GQA path).

Spec: ``docs/02_correctness_plan.md`` sec 1.3 (MHA) and sec 1.4 (GQA). M4
lands the five MHA cases; M5 adds seven GQA cases below.

MHA cases (M4):

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

GQA cases (M5):

* ``test_gqa_shape`` — output shape preserved when ``n_kv_head < n_head``.
* ``test_gqa_reduces_to_mha_when_kv_eq_q`` — a GQA module with ``n_kv_head=2``
  whose KV weights are manually expanded into the equivalent ``n_kv_head=4``
  module produces bit-equivalent outputs, validating the KV repeat scheme.
* ``test_gqa_value_vs_llama_attention`` — bit-matches HF ``LlamaAttention``
  with identity RoPE; isolates the KV repeat from RoPE wiring. Skipped if
  ``transformers`` is not installed.
* ``test_gqa_kv_head_grouping_count`` — ``n_rep`` is computed correctly and
  the repeat produces ``(B, n_head, T, head_dim)`` from ``(B, n_kv_head, T, head_dim)``.
* ``test_gqa_causal_no_leak`` — same adversarial as MHA, on a GQA config.
* ``test_gqa_with_rope_value_vs_llama`` — full attention + RoPE matches HF
  ``LlamaAttention``. Catches RoPE applied at the wrong stage.
  Skipped if ``transformers`` is not installed.
* ``test_gqa_backward`` — ``gradcheck`` through the ``n_rep > 1`` path.
"""

from __future__ import annotations

import pytest
import torch

from forge_llm.attention import GroupedQueryAttention
from forge_llm.rope import precompute_freqs_cis

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


# ---------------------------------------------------------------------------
# GQA cases (M5)
# ---------------------------------------------------------------------------

_GQA_D_MODEL = 64
_GQA_N_HEAD = 4
_GQA_N_KV_HEAD = 2
_GQA_HEAD_DIM = 16
_GQA_MAX_SEQ = 128


def _make_gqa() -> GroupedQueryAttention:
    return GroupedQueryAttention(
        d_model=_GQA_D_MODEL,
        n_head=_GQA_N_HEAD,
        n_kv_head=_GQA_N_KV_HEAD,
        head_dim=_GQA_HEAD_DIM,
        max_seq=_GQA_MAX_SEQ,
    )


def test_gqa_shape() -> None:
    gqa = _make_gqa()
    x = torch.randn(2, 8, _GQA_D_MODEL)
    y = gqa(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_gqa_reduces_to_mha_when_kv_eq_q() -> None:
    """GQA with manually-expanded KV weights matches the equivalent MHA module.

    Build a ``n_kv_head=2`` GQA, then build an ``n_kv_head=n_head=4`` "MHA"
    module whose ``wk``/``wv`` are obtained by ``repeat_interleave``-ing the
    GQA's wk/wv along the head axis. If the GQA forward correctly repeats KV
    by ``n_rep=2``, the two modules must produce bit-equal outputs.

    This is a stronger statement than just "forward runs": it pins the
    semantic of the repeat (each KV head broadcasts across ``n_rep`` query
    heads, not, e.g., is averaged).
    """
    head_dim, max_seq = 16, 32
    n_head, n_kv_head_small = 4, 2
    d_model = n_head * head_dim
    n_rep = n_head // n_kv_head_small

    gqa = GroupedQueryAttention(
        d_model=d_model,
        n_head=n_head,
        n_kv_head=n_kv_head_small,
        head_dim=head_dim,
        max_seq=max_seq,
    )
    mha = GroupedQueryAttention(
        d_model=d_model,
        n_head=n_head,
        n_kv_head=n_head,
        head_dim=head_dim,
        max_seq=max_seq,
    )
    with torch.no_grad():
        mha.wq.weight.copy_(gqa.wq.weight)
        mha.wo.weight.copy_(gqa.wo.weight)
        gqa_wk = gqa.wk.weight.view(n_kv_head_small, head_dim, d_model)
        gqa_wv = gqa.wv.weight.view(n_kv_head_small, head_dim, d_model)
        mha.wk.weight.copy_(
            gqa_wk.repeat_interleave(n_rep, dim=0).view(n_head * head_dim, d_model)
        )
        mha.wv.weight.copy_(
            gqa_wv.repeat_interleave(n_rep, dim=0).view(n_head * head_dim, d_model)
        )

    x = torch.randn(2, 8, d_model)
    torch.testing.assert_close(gqa(x), mha(x), rtol=1e-6, atol=1e-6)


def test_gqa_value_vs_llama_attention() -> None:
    """Match HF ``LlamaAttention`` with identity RoPE (isolates the KV repeat)."""
    try:
        from transformers import LlamaConfig  # noqa: PLC0415
        from transformers.models.llama.modeling_llama import (  # noqa: PLC0415
            LlamaAttention,
        )
    except ImportError:
        pytest.skip("transformers (HF Llama oracle) not installed")

    d_model, n_head, n_kv_head, head_dim, max_seq = 32, 8, 2, 4, 32
    seq_len = 8

    config = LlamaConfig(
        hidden_size=d_model,
        num_attention_heads=n_head,
        num_key_value_heads=n_kv_head,
        head_dim=head_dim,
        max_position_embeddings=max_seq,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
    )
    ours = GroupedQueryAttention(
        d_model=d_model,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        max_seq=max_seq,
    )
    theirs = LlamaAttention(config, layer_idx=0)
    with torch.no_grad():
        theirs.q_proj.weight.copy_(ours.wq.weight)
        theirs.k_proj.weight.copy_(ours.wk.weight)
        theirs.v_proj.weight.copy_(ours.wv.weight)
        theirs.o_proj.weight.copy_(ours.wo.weight)

    # Identity RoPE: cos=ones, sin=zeros makes the rotation a no-op so this
    # test isolates the KV repeat from the RoPE wiring.
    cos = torch.ones(2, seq_len, head_dim)
    sin = torch.zeros(2, seq_len, head_dim)
    x = torch.randn(2, seq_len, d_model)

    ours_out = ours(x, freqs_cis=None)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    theirs_out, _ = theirs(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=causal.view(1, 1, seq_len, seq_len),
    )
    torch.testing.assert_close(ours_out, theirs_out, rtol=1e-5, atol=1e-5)


def test_gqa_kv_head_grouping_count() -> None:
    """``n_rep`` and the KV head repeat produce the right shape."""
    n_head, n_kv_head, head_dim = 8, 2, 8
    d_model = n_head * head_dim
    gqa = GroupedQueryAttention(
        d_model=d_model,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        max_seq=32,
    )

    assert gqa.n_rep == n_head // n_kv_head == 4

    batch, seq_len = 2, 8
    x = torch.randn(batch, seq_len, d_model)
    k_proj = gqa.wk(x).view(batch, seq_len, n_kv_head, head_dim).transpose(1, 2)
    assert k_proj.shape == (batch, n_kv_head, seq_len, head_dim)
    k_repeated = k_proj.repeat_interleave(gqa.n_rep, dim=1)
    assert k_repeated.shape == (batch, n_head, seq_len, head_dim)

    # And the forward must run end-to-end, exercising the same repeat internally.
    y = gqa(x)
    assert y.shape == (batch, seq_len, d_model)


def test_gqa_causal_no_leak() -> None:
    """Mutating x[T-1] does not change the GQA output prefix."""
    gqa = _make_gqa()
    batch, seq_len = 2, 8
    x = torch.randn(batch, seq_len, _GQA_D_MODEL)
    y_orig = gqa(x)

    x_mut = x.clone()
    x_mut[:, seq_len - 1, :] = torch.randn(batch, _GQA_D_MODEL)
    y_mut = gqa(x_mut)
    assert torch.equal(
        y_orig[:, : seq_len - 1, :], y_mut[:, : seq_len - 1, :]
    ), "Causal mask leak in GQA: output at t < T-1 changed after mutating x[T-1]"


def test_gqa_with_rope_value_vs_llama() -> None:
    """Full GQA + RoPE matches HF ``LlamaAttention``."""
    try:
        from transformers import LlamaConfig  # noqa: PLC0415
        from transformers.models.llama.modeling_llama import (  # noqa: PLC0415
            LlamaAttention,
        )
    except ImportError:
        pytest.skip("transformers (HF Llama oracle) not installed")

    d_model, n_head, n_kv_head, head_dim, max_seq = 32, 8, 2, 4, 32
    seq_len = 8

    config = LlamaConfig(
        hidden_size=d_model,
        num_attention_heads=n_head,
        num_key_value_heads=n_kv_head,
        head_dim=head_dim,
        max_position_embeddings=max_seq,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
    )
    ours = GroupedQueryAttention(
        d_model=d_model,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
        max_seq=max_seq,
    )
    theirs = LlamaAttention(config, layer_idx=0)
    with torch.no_grad():
        theirs.q_proj.weight.copy_(ours.wq.weight)
        theirs.k_proj.weight.copy_(ours.wk.weight)
        theirs.v_proj.weight.copy_(ours.wv.weight)
        theirs.o_proj.weight.copy_(ours.wo.weight)

    fc = precompute_freqs_cis(head_dim, max_seq, theta=10000.0)
    fc_seq = fc[:seq_len]
    cos = torch.cat([fc_seq.real, fc_seq.real], dim=-1).unsqueeze(0).expand(2, -1, -1)
    sin = torch.cat([fc_seq.imag, fc_seq.imag], dim=-1).unsqueeze(0).expand(2, -1, -1)

    x = torch.randn(2, seq_len, d_model)
    ours_out = ours(x, freqs_cis=fc)

    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    theirs_out, _ = theirs(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=causal.view(1, 1, seq_len, seq_len),
    )
    torch.testing.assert_close(ours_out, theirs_out, rtol=1e-5, atol=1e-5)


def test_gqa_backward() -> None:
    """``gradcheck`` through the ``n_rep > 1`` path on a tiny fp64 GQA."""
    gqa = GroupedQueryAttention(
        d_model=8, n_head=4, n_kv_head=2, head_dim=2, max_seq=16
    ).double()
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(gqa, (x,))
