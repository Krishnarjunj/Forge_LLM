"""Train the byte-level BPE tokenizer on a FineWeb-Edu subset (M1).

Streams a small slice of FineWeb-Edu from HuggingFace, trains the from-scratch
:class:`forge_llm.tokenizer.BPETokenizer`, and writes the result to
``configs/tokenizer.json``. Run once, commit the JSON, and downstream training
re-uses it without re-streaming on every fresh Kaggle session.

Usage:
    python scripts/train_bpe.py
    python scripts/train_bpe.py --vocab-size 32000 --num-docs 100000
    python scripts/train_bpe.py --local-corpus path/to/corpus.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

from forge_llm.tokenizer import BPETokenizer

logger = logging.getLogger("forge_llm.train_bpe")

_DEFAULT_VOCAB_SIZE = 32000
_DEFAULT_NUM_DOCS = 100_000
_DEFAULT_OUT = Path("configs/tokenizer.json")
_DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
_DEFAULT_DATASET_NAME = "sample-10BT"


def _stream_fineweb(num_docs: int, dataset: str, name: str) -> Iterator[str]:
    """Yield up to ``num_docs`` document texts from FineWeb-Edu (streaming).

    Hard-fails with a clear message if the ``datasets`` package or the network
    is unavailable (CLAUDE.md sec 11: no silent fallbacks).
    """
    try:
        # Lazy import: `--help` and `--local-corpus` paths must work even if
        # the optional `datasets` extra is not installed.
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "scripts/train_bpe.py requires the `datasets` package. "
            "Install with: pip install 'datasets>=2.18'"
        ) from exc

    logger.info("opening %s (config=%s) in streaming mode", dataset, name)
    try:
        ds = load_dataset(dataset, name=name, split="train", streaming=True)
    except (ConnectionError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to open {dataset} ({name}): {exc}. "
            "Are you online? Is HF_TOKEN set for gated repos?"
        ) from exc

    for i, row in enumerate(ds):
        if i >= num_docs:
            break
        text = row.get("text")
        if isinstance(text, str) and text:
            yield text


def _read_local_corpus(path: Path) -> Iterator[str]:
    logger.info("reading local corpus from %s", path)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if stripped:
                yield stripped


def _train(corpus: Iterable[str], vocab_size: int, out: Path) -> None:
    docs = list(corpus)
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    logger.info("collected %d documents (~%.1f MB UTF-8)", len(docs), total_bytes / 1_000_000)
    if not docs:
        raise RuntimeError("corpus is empty; cannot train BPE")

    logger.info("training BPE (vocab_size=%d)", vocab_size)
    tokenizer = BPETokenizer.train(docs, vocab_size=vocab_size)

    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out)
    logger.info("wrote tokenizer to %s (vocab_size=%d)", out, tokenizer.vocab_size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=_DEFAULT_VOCAB_SIZE,
        help=f"BPE vocabulary size (default: {_DEFAULT_VOCAB_SIZE}).",
    )
    parser.add_argument(
        "--num-docs",
        type=int,
        default=_DEFAULT_NUM_DOCS,
        help=f"Number of FineWeb-Edu docs to stream (default: {_DEFAULT_NUM_DOCS}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output tokenizer JSON path (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=_DEFAULT_DATASET,
        help=f"HF dataset id (default: {_DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=_DEFAULT_DATASET_NAME,
        help=f"HF dataset config name (default: {_DEFAULT_DATASET_NAME}).",
    )
    parser.add_argument(
        "--local-corpus",
        type=Path,
        default=None,
        help="Path to a UTF-8 text file (one doc per line). Skips streaming.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.local_corpus is not None:
        corpus: Iterable[str] = _read_local_corpus(args.local_corpus)
    else:
        corpus = _stream_fineweb(args.num_docs, args.dataset, args.dataset_name)

    _train(corpus, vocab_size=args.vocab_size, out=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
