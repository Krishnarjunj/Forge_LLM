"""Adversarial causal-mask-leak test (CLAUDE.md sec 7 adversarial #1).

Mutate the value at token ``T-1`` repeatedly; assert that the output at all
positions ``t < T-1`` is **byte-identical** to the unmutated forward. A single
differing bit means future information leaked through the causal mask.

This test is the binding decoder-only-causality invariant for the whole
project. It runs on an MHA-only mini-model in M4; once GQA exists (M5) the
analogous ``test_gqa_causal_no_leak`` covers the GQA path in
``tests/test_attention.py``.
"""

from __future__ import annotations

import torch

from forge_llm.attention import GroupedQueryAttention


def test_causal_mask_adversarial_no_future_leak() -> None:
    """Repeatedly mutating x[:, T-1, :] never changes y[:, :T-1, :] in any bit."""
    mha = GroupedQueryAttention(d_model=32, n_head=4, n_kv_head=4, head_dim=8, max_seq=64)

    batch, seq_len, d_model = 4, 16, 32
    x = torch.randn(batch, seq_len, d_model)
    y_orig = mha(x)
    y_prefix = y_orig[:, : seq_len - 1, :]

    for trial in range(5):
        x_mut = x.clone()
        x_mut[:, seq_len - 1, :] = torch.randn(batch, d_model)
        y_mut = mha(x_mut)
        assert torch.equal(y_prefix, y_mut[:, : seq_len - 1, :]), (
            f"Causal-mask leak detected on trial {trial}: "
            f"mutating x[T-1] changed the output prefix."
        )
