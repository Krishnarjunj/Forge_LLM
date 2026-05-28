"""Tests for the pre-norm transformer block (M7).

Spec: ``docs/02_correctness_plan.md`` sec 1.9. Four cases:

* ``test_block_shape`` — output shape matches input on ``(2, 8, 64)``.
* ``test_block_value_vs_llama_decoder_layer`` — bit-matches HF
  ``LlamaDecoderLayer`` at rtol/atol = 1e-5 with weights copied across.
  Catches a wrong pre/post-norm ordering or a misrouted residual. Skipped
  (CLAUDE.md sec 11) when ``transformers`` is not installed.
* ``test_block_backward`` — ``torch.autograd.gradcheck`` on a tiny fp64
  forward exercises the residual + RoPE path.
* ``test_block_residual_routing`` — zero out the attention's output
  projection so ``attn(...) = 0``, then verify ``block(x) == x + mlp(norm2(x))``.
  Catches a residual connected to the wrong tensor (e.g., skipping the post-
  attention residual or wiring it to ``norm1(x)`` instead of ``x``).
"""

from __future__ import annotations

import pytest
import torch

from forge_llm.block import TransformerBlock
from forge_llm.rope import precompute_freqs_cis

_D_MODEL = 64
_N_HEAD = 4
_N_KV_HEAD = 2
_HEAD_DIM = 16
_D_FF = 176
_MAX_SEQ = 128
_EPS = 1e-5
_THETA = 10000.0


def _make_block() -> TransformerBlock:
    return TransformerBlock(
        d_model=_D_MODEL,
        n_head=_N_HEAD,
        n_kv_head=_N_KV_HEAD,
        head_dim=_HEAD_DIM,
        d_ff=_D_FF,
        max_seq=_MAX_SEQ,
        eps=_EPS,
    )


def test_block_shape() -> None:
    block = _make_block()
    fc = precompute_freqs_cis(_HEAD_DIM, _MAX_SEQ, _THETA)
    x = torch.randn(2, 8, _D_MODEL)
    y = block(x, fc)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_block_value_vs_llama_decoder_layer() -> None:
    """Bit-match HF ``LlamaDecoderLayer`` with copied weights."""
    try:
        from transformers import LlamaConfig  # noqa: PLC0415
        from transformers.models.llama.modeling_llama import (  # noqa: PLC0415
            LlamaDecoderLayer,
        )
    except ImportError:
        pytest.skip("transformers (HF Llama oracle) not installed")

    seq_len = 8
    config = LlamaConfig(
        hidden_size=_D_MODEL,
        intermediate_size=_D_FF,
        num_attention_heads=_N_HEAD,
        num_key_value_heads=_N_KV_HEAD,
        head_dim=_HEAD_DIM,
        max_position_embeddings=_MAX_SEQ,
        rms_norm_eps=_EPS,
        rope_theta=_THETA,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        hidden_act="silu",
        attn_implementation="eager",
    )
    ours = _make_block()
    theirs = LlamaDecoderLayer(config, layer_idx=0)
    with torch.no_grad():
        theirs.input_layernorm.weight.copy_(ours.norm1.weight)
        theirs.self_attn.q_proj.weight.copy_(ours.attn.wq.weight)
        theirs.self_attn.k_proj.weight.copy_(ours.attn.wk.weight)
        theirs.self_attn.v_proj.weight.copy_(ours.attn.wv.weight)
        theirs.self_attn.o_proj.weight.copy_(ours.attn.wo.weight)
        theirs.post_attention_layernorm.weight.copy_(ours.norm2.weight)
        theirs.mlp.gate_proj.weight.copy_(ours.mlp.w_gate.weight)
        theirs.mlp.up_proj.weight.copy_(ours.mlp.w_up.weight)
        theirs.mlp.down_proj.weight.copy_(ours.mlp.w_down.weight)

    fc = precompute_freqs_cis(_HEAD_DIM, _MAX_SEQ, _THETA)
    fc_seq = fc[:seq_len]
    cos = torch.cat([fc_seq.real, fc_seq.real], dim=-1).unsqueeze(0).expand(2, -1, -1)
    sin = torch.cat([fc_seq.imag, fc_seq.imag], dim=-1).unsqueeze(0).expand(2, -1, -1)
    x = torch.randn(2, seq_len, _D_MODEL)

    ours_out = ours(x, fc)
    causal = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    layer_out = theirs(
        hidden_states=x,
        position_embeddings=(cos, sin),
        attention_mask=causal.view(1, 1, seq_len, seq_len),
    )
    # LlamaDecoderLayer returns a tuple; first element is the hidden state.
    theirs_out = layer_out[0] if isinstance(layer_out, tuple) else layer_out

    torch.testing.assert_close(ours_out, theirs_out, rtol=1e-5, atol=1e-5)


def test_block_backward() -> None:
    """``gradcheck`` on a tiny fp64 block forward (residuals + RoPE)."""
    head_dim, max_seq = 4, 16
    block = TransformerBlock(
        d_model=8,
        n_head=2,
        n_kv_head=1,
        head_dim=head_dim,
        d_ff=16,
        max_seq=max_seq,
        eps=_EPS,
    ).double()
    fc = precompute_freqs_cis(head_dim, max_seq, _THETA).to(torch.complex128)
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x_: block(x_, fc), (x,))


def test_block_residual_routing() -> None:
    """With ``attn.wo`` zeroed, ``block(x) == x + mlp(norm2(x))``."""
    block = _make_block()
    with torch.no_grad():
        # Zero out the attention output projection so attn(norm1(x)) == 0,
        # which collapses the first residual to just ``x``.
        block.attn.wo.weight.zero_()

    fc = precompute_freqs_cis(_HEAD_DIM, _MAX_SEQ, _THETA)
    x = torch.randn(2, 8, _D_MODEL)

    actual = block(x, fc)
    expected = x + block.mlp(block.norm2(x))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
