"""SwiGLU feedforward (M6).

Three bias-free linears: ``w_gate`` and ``w_up`` (both ``d_model -> d_ff``)
and ``w_down`` (``d_ff -> d_model``). Forward computes
``w_down(SiLU(w_gate(x)) * w_up(x))`` -- the Llama MLP recipe.
"""

from __future__ import annotations

from typing import cast

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """SwiGLU feedforward block (Llama MLP)."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply SwiGLU.

        Shape: ``x`` is ``(..., d_model)``; the return has the same shape and
        the same dtype as ``x``.
        """
        return cast(Tensor, self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
