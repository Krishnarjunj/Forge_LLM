"""RMSNorm with fp32 compute and dtype-preserving I/O (M2).

Spec: ``docs/02_correctness_plan.md`` sec 1.1. The reduction is over the last
dimension; epsilon is added inside the ``rsqrt`` (matching the HF Llama
reference). The variance step runs in fp32 even when the input is fp16, which
is required for numerical stability under T4 mixed-precision training
(CLAUDE.md sec 6).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps: float = eps
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise ``x`` to unit RMS along its last axis, then scale by ``weight``.

        Shape: ``x`` is ``(..., hidden_size)``; the return has the same shape
        and the same dtype as ``x``.
        """
        input_dtype = x.dtype
        # Promote to at least fp32 for the variance/rsqrt step: fp16/bf16 lose
        # too much precision in the squared sum, and fp64 (gradcheck) must not
        # be silently downcast to fp32.
        compute_dtype = torch.promote_types(input_dtype, torch.float32)
        x_compute = x.to(compute_dtype)
        variance = x_compute.pow(2).mean(-1, keepdim=True)
        x_normed = x_compute * torch.rsqrt(variance + self.eps)
        return self.weight * x_normed.to(input_dtype)
