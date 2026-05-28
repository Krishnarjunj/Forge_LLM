"""WikiText-103 (or local-corpus) perplexity evaluation entrypoint (M10).

Loads a Forge checkpoint via ``forge_llm.hub.from_pretrained`` and computes
byte-level perplexity over a text source. Two sources supported:

* ``--dataset wikitext-103`` -- streams ``Salesforce/wikitext`` valid split
  via the ``datasets`` library. Hard-fails (CLAUDE.md sec 11) if ``datasets``
  or the network is unavailable.
* ``--local-corpus <path>`` -- reads one text per line from a UTF-8 file.
  Useful for offline smoke runs.

Usage:
    python scripts/eval_perplexity.py --checkpoint ckpt_dir --dataset wikitext-103
    python scripts/eval_perplexity.py --checkpoint ckpt_dir --local-corpus corpus.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

from forge_llm.eval import eval_byte_perplexity
from forge_llm.hub import from_pretrained
from forge_llm.tokenizer import BPETokenizer

logger = logging.getLogger("forge_llm.eval_perplexity")


def _stream_wikitext(split: str = "validation", max_texts: int | None = None) -> Iterator[str]:
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "scripts/eval_perplexity.py --dataset wikitext-103 requires the "
            "`datasets` package. Install with: pip install 'datasets>=2.18'"
        ) from exc

    try:
        ds = load_dataset(
            "Salesforce/wikitext", name="wikitext-103-raw-v1", split=split, streaming=True
        )
    except (ConnectionError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to open Salesforce/wikitext (split={split}): {exc}. "
            "Are you online?"
        ) from exc

    for i, row in enumerate(ds):
        if max_texts is not None and i >= max_texts:
            break
        text = row.get("text")
        if isinstance(text, str) and text.strip():
            yield text


def _read_local(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped:
                yield stripped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Window size for chunked scoring (default: 1024).",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Cap total bytes scored (default: no cap).",
    )
    parser.add_argument(
        "--max-texts",
        type=int,
        default=None,
        help="For --dataset wikitext-103: cap text count streamed.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device (default: cpu).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Load tokenizer from the checkpoint directory; reuse for eval.
    tokenizer = BPETokenizer.load(args.checkpoint / "tokenizer.json")

    # from_pretrained needs a data_factory for the trainer's stateful dataset,
    # but eval only reads model.* -- pass an empty stub.
    trainer = from_pretrained(
        args.checkpoint, data_factory=lambda: iter([]), device=args.device
    )

    if args.dataset == "wikitext-103":
        texts: Iterator[str] = _stream_wikitext(max_texts=args.max_texts)
    else:
        texts = _read_local(args.local_corpus)

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


if __name__ == "__main__":
    sys.exit(main())
