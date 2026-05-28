"""Tests for the data pipeline (M9).

Spec: ``docs/02_correctness_plan.md`` sec 1.14. Three cases, all offline (no
HuggingFace stream): the test mocks the document iterator by passing a
factory that yields strings from an inline list.

* ``test_packed_dataset_yields_correct_seq_len`` — every yielded tensor is
  exactly ``seq_len`` tokens. Catches a packing bug that leaks shorter
  sequences at the start or end of a doc boundary.
* ``test_packed_dataset_iterator_step_preserves_position`` — save state after
  yielding N sequences; build a fresh dataset, load_state_dict, and verify
  the next N sequences exactly match the original's sequences N+1..2N. This
  is the resume-correctness invariant for the data side: any drift here
  produces a loss-curve discontinuity at resume time.
* ``test_packed_dataset_no_eos_at_arbitrary_position_leak`` — EOS appears
  only at the token-stream positions that correspond to doc boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
import torch

from forge_llm.data import PackedDataset
from forge_llm.tokenizer import BPETokenizer

_DOCS = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack densely; never pad. Pretraining loss is calibrated on full sequences.",
    "Tokens flow from the streamer through the BPE into fixed-length chunks.",
    "Causal language models train on next-token prediction with teacher forcing.",
    "Resume safety means the loss curve after restart is bit-identical to the uninterrupted run.",
] * 20

_SEQ_LEN = 32


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    """BPE trained on the mock corpus; vocab_size=512 is the M1 standard."""
    return BPETokenizer.train(" ".join(_DOCS), vocab_size=512)


def _make_factory(docs: list[str]) -> Callable[[], Iterator[str]]:
    """Return a callable that produces a fresh iterator over ``docs`` each call."""
    return lambda: iter(docs)


def test_packed_dataset_yields_correct_seq_len(tokenizer: BPETokenizer) -> None:
    """Every yielded chunk is exactly seq_len tokens long."""
    dataset = PackedDataset(
        doc_iter_factory=_make_factory(_DOCS),
        tokenizer=tokenizer,
        seq_len=_SEQ_LEN,
        eos_id=tokenizer.eos_id,
    )
    seen = 0
    for seq in dataset:
        assert seq.shape == (_SEQ_LEN,), f"expected ({_SEQ_LEN},), got {seq.shape}"
        assert seq.dtype == torch.long
        seen += 1
        if seen >= 8:
            break
    assert seen == 8


def test_packed_dataset_iterator_step_preserves_position(
    tokenizer: BPETokenizer,
) -> None:
    """save_state -> rebuild -> load_state yields the exact tail of the original."""
    n_save = 10
    n_tail = 10

    # Original dataset: yield n_save + n_tail sequences, capture state at step n_save.
    original = PackedDataset(
        doc_iter_factory=_make_factory(_DOCS),
        tokenizer=tokenizer,
        seq_len=_SEQ_LEN,
        eos_id=tokenizer.eos_id,
    )
    it = iter(original)
    head = [next(it) for _ in range(n_save)]
    state_at_n_save = dict(original.state_dict())  # snapshot before tail
    tail_reference = [next(it) for _ in range(n_tail)]

    # Resumed dataset: build fresh, load the snapshotted state, iterate n_tail steps.
    resumed = PackedDataset(
        doc_iter_factory=_make_factory(_DOCS),
        tokenizer=tokenizer,
        seq_len=_SEQ_LEN,
        eos_id=tokenizer.eos_id,
    )
    resumed.load_state_dict(state_at_n_save)
    resumed_iter = iter(resumed)
    resumed_tail = [next(resumed_iter) for _ in range(n_tail)]

    # Resumed tail must match the original's tail exactly.
    assert len(head) == n_save
    for i, (a, b) in enumerate(zip(tail_reference, resumed_tail, strict=True)):
        assert torch.equal(a, b), (
            f"resumed sequence {i} (offset {n_save + i}) does not match original; "
            "the data iterator's resume offset is wrong."
        )


def test_packed_dataset_no_eos_at_arbitrary_position_leak(
    tokenizer: BPETokenizer,
) -> None:
    """EOS appears only at positions that correspond to doc boundaries.

    Build the expected EOS-position set from the docs, flatten the dataset's
    output, and assert every EOS in the flattened stream sits at a boundary.
    A bug that, e.g., inserts EOS at every ``seq_len`` boundary would fail
    this check.
    """
    eos = tokenizer.eos_id

    # Expected token stream: tokenize each doc, separate with one EOS.
    expected_stream: list[int] = []
    expected_eos_positions: set[int] = set()
    for doc in _DOCS:
        expected_stream.extend(tokenizer.encode(doc))
        expected_eos_positions.add(len(expected_stream))  # next slot is the EOS
        expected_stream.append(eos)

    dataset = PackedDataset(
        doc_iter_factory=_make_factory(_DOCS),
        tokenizer=tokenizer,
        seq_len=_SEQ_LEN,
        eos_id=eos,
    )
    flat: list[int] = []
    for seq in dataset:
        flat.extend(seq.tolist())

    assert flat, "PackedDataset emitted no chunks on a corpus that should fit several"

    # 1. Every EOS the dataset emits sits at an expected doc boundary.
    actual_eos_positions = {i for i, tok in enumerate(flat) if tok == eos}
    leaked = actual_eos_positions - expected_eos_positions
    assert not leaked, (
        f"EOS leaked at non-boundary positions {sorted(leaked)[:10]}...; "
        "the packer is emitting EOS at chunk boundaries or some other artefact."
    )

    # 2. The dataset's output is a prefix of the expected concat-with-EOS stream
    #    (the dataset stops when the doc stream is exhausted with < seq_len in
    #    the buffer, so it can be shorter than the expected stream).
    assert flat == expected_stream[: len(flat)], (
        "packed stream diverges from the expected concat-with-EOS sequence"
    )
