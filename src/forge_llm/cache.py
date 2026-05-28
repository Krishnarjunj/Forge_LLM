"""Statically-allocated KV cache (M11; gpt-fast pattern).

One ``KVCache`` instance is a *single layer's* cache: paired ``k_cache`` and
``v_cache`` buffers of shape ``(max_batch, n_kv_head, max_seq, head_dim)``,
allocated once and indexed (not concatenated) on every step. ``.update`` does
the per-step write and returns the full buffers so the caller can slice them
to the valid prefix for the attention matmul.

``KVCache.allocate(config, max_batch)`` returns an ``nn.ModuleList`` of one
``KVCache`` per attention layer -- the value the model accepts as its
``cache`` argument.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from forge_llm.config import ForgeConfig


class KVCache(nn.Module):
    """Single-layer KV cache (statically-allocated buffers)."""

    k_cache: Tensor
    v_cache: Tensor

    def __init__(
        self,
        max_batch: int,
        max_seq: int,
        n_kv_head: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        shape = (max_batch, n_kv_head, max_seq, head_dim)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=dtype))

    def update(
        self,
        input_pos: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Write ``k``/``v`` at the seq positions in ``input_pos``.

        Shape:
          ``input_pos``: ``(T_new,)`` int64 -- target indices into the seq dim.
          ``k``, ``v``:  ``(B, n_kv_head, T_new, head_dim)``.

        Returns the full ``(k_cache, v_cache)`` buffers; the caller slices to
        the valid prefix.
        """
        self.k_cache[:, :, input_pos] = k
        self.v_cache[:, :, input_pos] = v
        return self.k_cache, self.v_cache

    @classmethod
    def allocate(
        cls,
        config: ForgeConfig,
        max_batch: int,
        dtype: torch.dtype = torch.float32,
    ) -> nn.ModuleList:
        """Build one cache per attention layer for the given config."""
        return nn.ModuleList(
            [
                cls(
                    max_batch=max_batch,
                    max_seq=config.max_seq,
                    n_kv_head=config.n_kv_head,
                    head_dim=config.head_dim,
                    dtype=dtype,
                )
                for _ in range(config.n_layer)
            ]
        )
