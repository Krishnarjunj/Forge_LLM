"""FineWeb-Edu streaming dataset with packing and iterator-state save/load (M9).

``PackedDataset`` is the boundary between the upstream text stream and the
trainer's tensor input: it concatenates tokenised documents (one ``eos_id``
between docs, no padding) and emits fixed-length ``seq_len`` chunks. State
save/load is intentionally minimal -- just the count of sequences emitted --
because HF's streaming datasets do not support random access, so resume must
re-create the stream from scratch and fast-forward. That makes resume O(N)
in the number of steps already taken, which is acceptable for the few
thousand steps we expect on free Kaggle T4.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import IterableDataset

from forge_llm.tokenizer import BPETokenizer

logger = logging.getLogger("forge_llm.data")


class PackedDataset(IterableDataset[Tensor]):
    """Pack a streaming corpus into fixed ``seq_len`` chunks separated by EOS."""

    def __init__(
        self,
        doc_iter_factory: Callable[[], Iterator[str]],
        tokenizer: BPETokenizer,
        seq_len: int,
        eos_id: int,
    ) -> None:
        super().__init__()
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        self._factory = doc_iter_factory
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.eos_id = eos_id
        self._global_step: int = 0

    def __iter__(self) -> Iterator[Tensor]:
        docs = self._factory()
        buffer: list[int] = []
        emitted = 0

        while True:
            # Fill the buffer to at least seq_len tokens.
            while len(buffer) < self.seq_len:
                try:
                    doc = next(docs)
                except StopIteration:
                    return
                buffer.extend(self._tokenizer.encode(doc))
                buffer.append(self.eos_id)

            seq = buffer[: self.seq_len]
            buffer = buffer[self.seq_len :]
            emitted += 1

            # Fast-forward path on resume: skip until we've passed the
            # previously-recorded position.
            if emitted <= self._global_step:
                continue

            # Update state BEFORE yield so a consumer that snapshots
            # state_dict between two next() calls sees the count of
            # sequences actually delivered so far (not N-1).
            self._global_step = emitted
            yield torch.tensor(seq, dtype=torch.long)

    def state_dict(self) -> dict[str, Any]:
        return {"global_step": self._global_step}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._global_step = int(state["global_step"])


def fineweb_doc_iterator(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    split: str = "train",
) -> Iterator[str]:
    """Stream document texts from FineWeb-Edu via ``datasets``.

    Hard-fails (CLAUDE.md sec 11) when ``datasets`` is missing or the network
    is unavailable; the trainer never silently falls back to a stub stream.
    """
    try:
        # Lazy import so unit tests don't pay the import cost.
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "forge_llm.data.fineweb_doc_iterator requires `datasets`. "
            "Install with: pip install 'datasets>=2.18'"
        ) from exc

    logger.info("opening %s (config=%s) in streaming mode", dataset_name, dataset_config)
    try:
        ds = load_dataset(
            dataset_name, name=dataset_config, split=split, streaming=True
        )
    except (ConnectionError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"failed to open {dataset_name} ({dataset_config}): {exc}. "
            "Are you online? Is HF_TOKEN set for gated repos?"
        ) from exc

    for row in ds:
        text = row.get("text")
        if isinstance(text, str) and text:
            yield text
