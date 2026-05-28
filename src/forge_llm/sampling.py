"""Pure sampling utilities (M11).

Four logit transforms used by ``forge_llm.generation.generate``: ``top_k``,
``top_p`` (nucleus), ``apply_temperature`` (with safe T=0 handling), and
``repetition_penalty`` (CTRL paper). Pure tensor-in / tensor-out so they can
be composed and unit-tested without instantiating a model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def top_k(logits: Tensor, k: int) -> Tensor:
    """Keep the top-``k`` logits along the last dim; set the rest to ``-inf``."""
    if k <= 0:
        return logits
    if k >= logits.shape[-1]:
        return logits
    vals, _ = torch.topk(logits, k=k, dim=-1)
    threshold = vals[..., -1:].expand_as(logits)
    return torch.where(logits >= threshold, logits, torch.full_like(logits, float("-inf")))


def top_p(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering: keep the smallest set of logits whose cumulative
    softmax probability is >= ``p``; set the rest to ``-inf``.

    The largest logit is always kept (so we never return all ``-inf``).
    """
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(descending=True, dim=-1)
    cumprobs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    # In sorted order: tokens are removed once cumulative prob has passed p,
    # but the first token that crosses p is kept (shift the mask right).
    remove_sorted = cumprobs > p
    remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
    remove_sorted[..., 0] = False
    # Scatter the boolean back to original positions.
    remove = torch.zeros_like(remove_sorted)
    remove.scatter_(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Scale logits by ``1 / temperature``.

    ``temperature == 0`` is a sentinel for greedy decoding: returns a logit
    vector whose softmax is a one-hot at the argmax (no divide-by-zero crash).
    """
    if temperature == 0.0:
        max_idx = logits.argmax(dim=-1, keepdim=True)
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(-1, max_idx, 0.0)
        return out
    return logits / temperature


def repetition_penalty(
    logits: Tensor, history: Tensor, penalty: float
) -> Tensor:
    """Apply the CTRL-paper repetition penalty in place over ``history`` token ids.

    For each token id ``i`` in ``history``: if ``logits[..., i] > 0`` divide
    by ``penalty``, otherwise multiply by ``penalty`` (so positive logits are
    pushed down and negative logits pushed more negative).
    """
    if penalty == 1.0:
        return logits
    out = logits.clone()
    selected = out.index_select(-1, history)
    scaled = torch.where(selected > 0, selected / penalty, selected * penalty)
    out.index_copy_(-1, history, scaled)
    return out
