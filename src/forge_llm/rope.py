"""Rotary Positional Embeddings (M3, HF Llama half-split convention).

See ADR-007 in ``docs/DECISIONS.md`` (amended 2026-05-28) and the spec at
``docs/02_correctness_plan.md`` sec 1.2.

The rotation pairs are halves: ``x[..., :d/2]`` rotates with ``x[..., d/2:]``,
implemented as ``y = x*cos + rotate_half(x)*sin`` with
``rotate_half([a, b]) = [-b, a]``. ``cos``/``sin`` tables are derived on the
fly from the complex ``freqs_cis`` cache, which is the more compact form
(half the memory of separate cos/sin tables).
"""

from __future__ import annotations

import torch
from torch import Tensor


def precompute_freqs_cis(
    head_dim: int,
    max_seq: int,
    theta: float = 10000.0,
) -> Tensor:
    """Precompute ``e^{i * m * theta_i}`` for every position/pair.

    Shape: ``(max_seq, head_dim // 2)``, complex64.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_seq, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


def _rotate_half(x: Tensor) -> Tensor:
    """Split the last dim in halves, negate the second half, swap.

    HF Llama convention: ``rotate_half([a, b]) = [-b, a]``.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(
    q: Tensor,
    k: Tensor,
    freqs_cis: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key tensors.

    Shape:
      ``q``, ``k``:    ``(B, H, T, head_dim)`` with ``head_dim`` even.
      ``freqs_cis``:   ``(max_seq, head_dim // 2)`` complex; ``T <= max_seq``.

    Returns rotated ``(q, k)`` with the same shape and dtype as the inputs.
    """
    seq_len = q.shape[-2]
    fc = freqs_cis[:seq_len]
    cos = torch.cat([fc.real, fc.real], dim=-1)
    sin = torch.cat([fc.imag, fc.imag], dim=-1)
    # Add (B, H) broadcast dims, then cast to q's dtype so an fp16/bf16 path
    # stays in low precision (matching HF Llama).
    cos = cos.unsqueeze(0).unsqueeze(0).to(q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(q.dtype)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
