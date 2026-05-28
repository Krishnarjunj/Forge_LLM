"""Grouped-Query Attention (M4: MHA path; M5: GQA grouping + RoPE wiring).

A single module that implements both MHA (``n_kv_head == n_head``) and GQA
(``n_kv_head < n_head``). M4 covers the MHA path; the GQA-specific path
(KV head repeat, RoPE-on-q-and-K-before-repeat) is implemented here too so
M5 is a small test-only extension, but RoPE is applied only when ``freqs_cis``
is supplied.

Causal masking is unconditional -- Forge-LLM is decoder-only
(``forge_llm.md`` and CLAUDE.md sec 2). Softmax runs in fp32 (or higher) for
mixed-precision stability per CLAUDE.md sec 6.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from forge_llm.rope import apply_rotary


class GroupedQueryAttention(nn.Module):
    """Multi-/Grouped-Query attention with causal self-attention."""

    _causal_mask: Tensor  # registered buffer; declared so mypy strict can see it.

    def __init__(
        self,
        d_model: int,
        n_head: int,
        n_kv_head: int,
        head_dim: int,
        max_seq: int,
        attn_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model != n_head * head_dim:
            raise ValueError(
                f"d_model ({d_model}) must equal n_head * head_dim "
                f"({n_head} * {head_dim} = {n_head * head_dim})"
            )
        if n_head % n_kv_head != 0:
            raise ValueError(
                f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
            )
        self.d_model: int = d_model
        self.n_head: int = n_head
        self.n_kv_head: int = n_kv_head
        self.head_dim: int = head_dim
        self.n_rep: int = n_head // n_kv_head
        self.scale: float = 1.0 / math.sqrt(head_dim)

        self.wq = nn.Linear(d_model, n_head * head_dim, bias=attn_bias)
        self.wk = nn.Linear(d_model, n_kv_head * head_dim, bias=attn_bias)
        self.wv = nn.Linear(d_model, n_kv_head * head_dim, bias=attn_bias)
        self.wo = nn.Linear(n_head * head_dim, d_model, bias=attn_bias)

        # Causal mask is built once and sliced per-forward; -inf above the
        # diagonal so softmax pushes those entries to exactly zero regardless
        # of the (finite) score value -- this is what makes the adversarial
        # leak test pass byte-identically rather than approximately.
        mask = torch.triu(
            torch.full((max_seq, max_seq), float("-inf")), diagonal=1
        )
        self.register_buffer("_causal_mask", mask, persistent=False)

    def forward(
        self,
        x: Tensor,
        freqs_cis: Tensor | None = None,
    ) -> Tensor:
        """Causal self-attention.

        Shape:
          ``x``:         ``(B, T, d_model)``
          ``freqs_cis``: ``None`` (M4 MHA without RoPE) or
                         ``(max_seq, head_dim // 2)`` complex (applied to Q, K
                         before the KV repeat and the matmul).
        Returns ``(B, T, d_model)`` with the same dtype as ``x``.
        """
        bsz, seq_len, _ = x.shape

        q = (
            self.wq(x)
            .view(bsz, seq_len, self.n_head, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.wk(x)
            .view(bsz, seq_len, self.n_kv_head, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.wv(x)
            .view(bsz, seq_len, self.n_kv_head, self.head_dim)
            .transpose(1, 2)
        )

        if freqs_cis is not None:
            q, k = apply_rotary(q, k, freqs_cis)

        # MHA: n_rep == 1 -> no-op. GQA: each KV head is repeated n_rep times
        # so the matmul against Q lines up.
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + self._causal_mask[:seq_len, :seq_len]

        # Promote to at least fp32 for softmax stability; on fp32/fp64 inputs
        # this is a no-op so the value-vs-oracle test still hits bit-equality.
        softmax_dtype = torch.promote_types(scores.dtype, torch.float32)
        attn = F.softmax(scores.to(softmax_dtype), dim=-1).to(scores.dtype)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        return self.wo(out)
