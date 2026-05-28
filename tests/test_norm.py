"""Tests for RMSNorm (M2).

Spec: ``docs/02_correctness_plan.md`` sec 1.1. Five cases:

* ``test_rmsnorm_shape`` — output shape equals input shape on ``(2, 8, 64)``.
* ``test_rmsnorm_value_vs_llama`` — matches HF ``LlamaRMSNorm`` at fp32 with
  rtol/atol = 1e-6 when weights are copied across. Skipped with a reason
  string (CLAUDE.md sec 11) if ``transformers`` isn't installed in the env.
* ``test_rmsnorm_backward`` — ``torch.autograd.gradcheck`` on a tiny fp64
  forward catches a broken gradient through ``rsqrt`` / the variance reduction.
* ``test_rmsnorm_determinism`` — two forward passes with the same input and
  the same layer must be bit-identical (``torch.equal``).
* ``test_rmsnorm_dtype_promotion`` — fp16 in -> fp16 out, but the fp16 output
  cast back to fp32 must agree with the fp32-path output at atol=1e-3. If
  internal compute were fp16 the variance step would lose precision badly
  enough that this bound fails.
"""

from __future__ import annotations

import pytest
import torch

from forge_llm.norm import RMSNorm

_HIDDEN = 64
_EPS = 1e-5


def test_rmsnorm_shape() -> None:
    layer = RMSNorm(hidden_size=_HIDDEN, eps=_EPS)
    x = torch.randn(2, 8, _HIDDEN)
    y = layer(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_rmsnorm_value_vs_llama() -> None:
    """Match the HF LlamaRMSNorm oracle exactly at fp32 (CLAUDE.md sec 4)."""
    try:
        from transformers.models.llama.modeling_llama import (  # noqa: PLC0415
            LlamaRMSNorm,
        )
    except ImportError:
        pytest.skip("transformers (HF Llama oracle) not installed")

    ours = RMSNorm(hidden_size=_HIDDEN, eps=_EPS)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(_HIDDEN))

    oracle = LlamaRMSNorm(hidden_size=_HIDDEN, eps=_EPS)
    with torch.no_grad():
        oracle.weight.copy_(ours.weight)

    x = torch.randn(2, 8, _HIDDEN)
    torch.testing.assert_close(ours(x), oracle(x), rtol=1e-6, atol=1e-6)


def test_rmsnorm_backward() -> None:
    """``gradcheck`` on a tiny fp64 forward catches bad rsqrt gradient routing."""
    layer = RMSNorm(hidden_size=8, eps=_EPS).double()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(8, dtype=torch.float64))
    x = torch.randn(2, 8, 8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))


def test_rmsnorm_determinism() -> None:
    """Two forwards through the same layer on the same input must be bit-equal."""
    layer = RMSNorm(hidden_size=_HIDDEN, eps=_EPS)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(_HIDDEN))
    x = torch.randn(2, 8, _HIDDEN)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.equal(y1, y2)


def test_rmsnorm_dtype_promotion() -> None:
    """fp16 in -> fp16 out, but internal compute is fp32 (no precision loss)."""
    layer_fp32 = RMSNorm(hidden_size=_HIDDEN, eps=_EPS)
    with torch.no_grad():
        layer_fp32.weight.copy_(torch.randn(_HIDDEN))
    x_fp32 = torch.randn(2, 8, _HIDDEN)
    y_fp32 = layer_fp32(x_fp32)

    layer_fp16 = RMSNorm(hidden_size=_HIDDEN, eps=_EPS)
    with torch.no_grad():
        layer_fp16.weight.copy_(layer_fp32.weight)
    layer_fp16 = layer_fp16.half()
    x_fp16 = x_fp32.half()
    y_fp16 = layer_fp16(x_fp16)

    assert y_fp16.dtype == torch.float16, "fp16 input must produce fp16 output"
    # Per CLAUDE.md sec 4: fp16 tolerance is atol=rtol=1e-3.
    torch.testing.assert_close(y_fp16.float(), y_fp32, rtol=1e-3, atol=1e-3)
