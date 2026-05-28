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

import argparse
import logging
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from forge_llm.tokenizer import BPETokenizer

logger = logging.getLogger("forge_llm.eval")


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


# ---------------------------------------------------------------------------
# CLI: wired by both scripts/eval_perplexity.py and forge_llm.cli (M12).
# ---------------------------------------------------------------------------


def _stream_wikitext(split: str = "validation", max_texts: int | None = None) -> Iterator[str]:
    """Stream WikiText-103 valid split via HF datasets; hard-fail if offline."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "eval --dataset wikitext-103 requires the `datasets` package. "
            "Install with: pip install 'datasets>=2.18'"
        ) from exc
    try:
        ds = load_dataset(
            "Salesforce/wikitext",
            name="wikitext-103-raw-v1",
            split=split,
            streaming=True,
        )
    except (ConnectionError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to open Salesforce/wikitext (split={split}): {exc}. Are you online?"
        ) from exc
    for i, row in enumerate(ds):
        if max_texts is not None and i >= max_texts:
            break
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def _read_local_corpus(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped:
                yield stripped


def main(argv: list[str] | None = None) -> int:
    """``forge-llm eval`` and ``python scripts/eval_perplexity.py`` entrypoint."""
    parser = argparse.ArgumentParser(
        description="Byte-level perplexity evaluation on a Forge checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Directory written by forge_llm.hub.save_pretrained.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset",
        choices=["wikitext-103"],
        help="HF dataset to stream (validation split).",
    )
    source.add_argument(
        "--local-corpus",
        type=Path,
        help="UTF-8 text file, one document per line.",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=None)
    parser.add_argument("--max-texts", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Lazy imports avoid circular hub <-> train <-> eval through the CLI.
    from forge_llm.hub import from_pretrained  # noqa: PLC0415

    tokenizer = BPETokenizer.load(args.checkpoint / "tokenizer.json")
    trainer = from_pretrained(args.checkpoint, data_factory=lambda: iter([]), device=args.device)

    if args.dataset == "wikitext-103":
        texts: Iterator[str] = _stream_wikitext(max_texts=args.max_texts)
    else:
        texts = _read_local_corpus(args.local_corpus)

    ppl = eval_byte_perplexity(
        trainer.model,
        tokenizer,
        texts,
        seq_len=args.seq_len,
        max_bytes=args.max_bytes,
        device=args.device,
    )
    logger.info("byte-PPL = %.4f", ppl)
    print(f"{ppl:.4f}")
    return 0
