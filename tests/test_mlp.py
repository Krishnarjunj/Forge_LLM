"""Tests for SwiGLU feedforward (M6).

Spec: ``docs/02_correctness_plan.md`` sec 1.5. Five cases:

* ``test_swiglu_shape`` — output shape equals input shape on ``(2, 8, 64)``.
* ``test_swiglu_value_vs_llama_mlp`` — bit-matches HF ``LlamaMLP`` at fp32
  with rtol/atol = 1e-5 when ``w_gate``, ``w_up``, ``w_down`` weights are
  copied across. Skipped with a reason (CLAUDE.md sec 11) if ``transformers``
  is not installed.
* ``test_swiglu_uses_silu_not_gelu`` — single-element manual check: builds a
  rank-one input and identity-style projections, computes
  ``SiLU(2) * 3 = 2 * sigmoid(2) * 3 ≈ 5.2848`` by hand, and verifies our
  output matches. A silent SiLU -> GELU swap would yield ``GELU(2) * 3 ≈
  5.864``, which fails this assertion.
* ``test_swiglu_backward`` — ``torch.autograd.gradcheck`` on a tiny fp64
  forward.
* ``test_swiglu_determinism`` — two forwards on the same input must be bit-equal.
"""

from __future__ import annotations

import pytest
import torch

from forge_llm.mlp import SwiGLU

_D_MODEL = 64
_D_FF = 176


def test_swiglu_shape() -> None:
    mlp = SwiGLU(d_model=_D_MODEL, d_ff=_D_FF)
    x = torch.randn(2, 8, _D_MODEL)
    y = mlp(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_swiglu_value_vs_llama_mlp() -> None:
    """Bit-match HF ``LlamaMLP`` with copied weights."""
    try:
        from transformers import LlamaConfig  # noqa: PLC0415
        from transformers.models.llama.modeling_llama import LlamaMLP  # noqa: PLC0415
    except ImportError:
        pytest.skip("transformers (HF Llama oracle) not installed")

    config = LlamaConfig(
        hidden_size=_D_MODEL,
        intermediate_size=_D_FF,
        hidden_act="silu",
        mlp_bias=False,
    )
    ours = SwiGLU(d_model=_D_MODEL, d_ff=_D_FF)
    theirs = LlamaMLP(config)
    with torch.no_grad():
        theirs.gate_proj.weight.copy_(ours.w_gate.weight)
        theirs.up_proj.weight.copy_(ours.w_up.weight)
        theirs.down_proj.weight.copy_(ours.w_down.weight)

    x = torch.randn(2, 8, _D_MODEL)
    torch.testing.assert_close(ours(x), theirs(x), rtol=1e-5, atol=1e-5)


def test_swiglu_uses_silu_not_gelu() -> None:
    """Numerical check that the activation is SiLU (x*sigmoid(x)), not GELU.

    Builds rank-one projections so that ``w_gate(x) = [2, 0]`` and
    ``w_up(x) = [3, 0]`` for a chosen input ``[2, 3]``. Then the inner
    activation is ``SiLU(2) * 3 + SiLU(0) * 0 = 2*sigmoid(2)*3``, which the
    down projection extracts at output index 0. GELU at x=2 gives a different
    numerical value (~1.954 vs SiLU's ~1.762), so this assertion fails for
    any GELU substitution.
    """
    # fp64 throughout so the spec's atol=1e-7 is achievable -- 5.28 in fp32
    # has ~6e-7 of ULP slack, which would shadow real bugs.
    d_model, d_ff = 2, 2
    mlp = SwiGLU(d_model=d_model, d_ff=d_ff).double()
    with torch.no_grad():
        mlp.w_gate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64))
        mlp.w_up.weight.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64))
        mlp.w_down.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64))

    x = torch.tensor([[[2.0, 3.0]]], dtype=torch.float64)
    expected_index_0 = 2.0 * torch.sigmoid(torch.tensor(2.0, dtype=torch.float64)).item() * 3.0
    out = mlp(x).flatten()
    assert abs(out[0].item() - expected_index_0) < 1e-7, (
        f"SwiGLU output index 0 = {out[0].item()}, expected SiLU value "
        f"{expected_index_0}; activation may be wrong (GELU instead of SiLU?)"
    )
    assert abs(out[1].item() - 0.0) < 1e-7


def test_swiglu_backward() -> None:
    """``gradcheck`` on a tiny fp64 forward."""
    mlp = SwiGLU(d_model=8, d_ff=16).double()
    x = torch.randn(1, 4, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mlp, (x,))


def test_swiglu_determinism() -> None:
    """Two forwards through the same module on the same input are bit-equal."""
    mlp = SwiGLU(d_model=_D_MODEL, d_ff=_D_FF)
    x = torch.randn(2, 8, _D_MODEL)
    y1 = mlp(x)
    y2 = mlp(x)
    assert torch.equal(y1, y2)
