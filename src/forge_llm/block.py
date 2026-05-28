"""Pre-norm transformer decoder block (M7).

Wires together ``RMSNorm`` + ``GroupedQueryAttention`` + ``RMSNorm`` +
``SwiGLU`` in the pre-norm Llama style:

* ``h = x + attn(norm1(x), freqs_cis)``
* ``out = h + mlp(norm2(h))``

Constructor takes the same scalar fields ``ForgeConfig`` will surface in M8;
M8 just unpacks the dataclass into this signature.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from forge_llm.attention import GroupedQueryAttention
from forge_llm.mlp import SwiGLU
from forge_llm.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-norm decoder block: norm -> attn -> +residual -> norm -> mlp -> +residual."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        d_ff: int,
        max_seq: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=eps)
        self.attn = GroupedQueryAttention(
            d_model=d_model,
            n_head=n_head,
            n_kv_head=n_kv_head,
            head_dim=head_dim,
            max_seq=max_seq,
        )
        self.norm2 = RMSNorm(d_model, eps=eps)
        self.mlp = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor,
        cache: Any = None,
        input_pos: Tensor | None = None,
    ) -> Tensor:
        """One pre-norm decoder block.

        Shape: ``x`` is ``(B, T, d_model)``; ``freqs_cis`` is
        ``(max_seq, head_dim // 2)`` complex. ``cache`` and ``input_pos`` are
        threaded through to the attention layer for the M11 generation path.
        Returns ``(B, T, d_model)`` with the same dtype as ``x``.
        """
        h = x + self.attn(
            self.norm1(x), freqs_cis=freqs_cis, cache=cache, input_pos=input_pos
        )
        return h + self.mlp(self.norm2(h))
