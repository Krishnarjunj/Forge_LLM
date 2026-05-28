"""Byte-level perplexity evaluation (M10).

Byte-PPL normalises the total negative log-likelihood (in nats) by the raw
UTF-8 byte count of the scored text, not by the token count. The number is
tokenizer-agnostic: any two models trained with different BPEs can be
compared on the same text by their byte-PPL alone.

Formula: ``byte_ppl = exp(sum_token_nll_nats / total_bytes)`` where
``sum_token_nll_nats`` is the total cross-entropy over all predicted
positions and ``total_bytes`` is the cumulative UTF-8 byte length of the
texts that were scored.

The first token of each text-block contributes 0 to the NLL sum (no context
to predict it from) but its bytes still count -- standard convention,
matching nanoGPT and HuggingFace's eval scripts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from forge_llm.tokenizer import BPETokenizer


def eval_byte_perplexity(
    model: nn.Module,
    tokenizer: BPETokenizer,
    texts: Iterable[str],
    seq_len: int,
    max_bytes: int | None = None,
    device: str | torch.device = "cpu",
) -> float:
    """Compute byte-level perplexity over an iterable of texts.

    Texts are tokenised, packed into non-overlapping ``seq_len`` chunks, and
    each chunk is scored end-to-end (``T-1`` predicted positions per chunk).
    Any residual partial chunk at the end is scored once. ``max_bytes`` caps
    the byte budget so a long stream can be eval'd in bounded time.
    """
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2 for next-token loss, got {seq_len}")

    device_t = torch.device(device)
    model.eval()
    total_nll_nats = 0.0
    total_bytes = 0
    token_buffer: list[int] = []

    def _score(tokens: list[int]) -> float:
        if len(tokens) < 2:
            return 0.0
        inputs = torch.tensor(tokens[:-1], device=device_t, dtype=torch.long).unsqueeze(0)
        targets = torch.tensor(tokens[1:], device=device_t, dtype=torch.long).unsqueeze(0)
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="sum",
        )
        return float(loss.item())

    with torch.no_grad():
        for text in texts:
            total_bytes += len(text.encode("utf-8"))
            token_buffer.extend(tokenizer.encode(text))

            while len(token_buffer) >= seq_len:
                chunk = token_buffer[:seq_len]
                token_buffer = token_buffer[seq_len:]
                total_nll_nats += _score(chunk)

            if max_bytes is not None and total_bytes >= max_bytes:
                break

        # Residual partial chunk (so short texts and trailing tokens are not lost).
        total_nll_nats += _score(token_buffer)

    if total_bytes == 0:
        return float("inf")
    return math.exp(total_nll_nats / total_bytes)
