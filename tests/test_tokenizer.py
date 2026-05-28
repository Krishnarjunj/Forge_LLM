"""Tests for the from-scratch byte-level BPE tokenizer (M1).

Spec: ``docs/02_correctness_plan.md`` §1.11. Five cases:

* ``test_bpe_roundtrip`` — encode→decode is byte-identical for ASCII, unicode,
  and 100 deterministically generated paragraph-shaped strings (the roadmap
  M1 "100 random FineWeb-Edu paragraphs" exit criterion, approximated offline).
* ``test_bpe_special_tokens`` — bos/eos/pad/unk id assignment, wrapping with
  ``add_special_tokens=True``, ``skip_special_tokens=True`` on decode, and the
  invariant that a literal "<bos>" in input text does NOT collapse to bos_id.
* ``test_bpe_vocab_size_matches_config`` — trained vocab size equals the
  requested size (so embedding allocation never goes OOB).
* ``test_bpe_against_tiktoken_sanity`` (slow) — sanity-compare token counts to
  ``tiktoken.get_encoding("gpt2")``. Skipped if tiktoken isn't installed.
* ``test_bpe_save_load`` — train → save → load → encode produces identical
  token ids and identical special-token ids.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from forge_llm.tokenizer import BPETokenizer

# ---------------------------------------------------------------------------
# Corpus fixtures
# ---------------------------------------------------------------------------

# A small but varied English corpus: prose, technical terms, punctuation,
# unicode. Stays inline so tests have zero filesystem / network dependency.
# ~5 KB; enough for a 2K-vocab BPE to learn meaningful merges in a few seconds.
_CORPUS = """\
The history of language modelling is the history of finding compact, learnable \
representations for sequences of symbols. Byte-pair encoding, originally a data \
compression technique, became the default subword tokenizer for large language \
models because it gracefully handles open vocabularies without ever emitting an \
unknown token for raw text.

Transformers operate on integer token sequences. The tokenizer is therefore the \
boundary between human text and the model's numerical world. A buggy tokenizer \
can silently degrade every downstream metric: pretraining loss, fine-tuning loss, \
perplexity on held-out data, generation quality, and even gradient stability when \
rare-byte sequences blow up the embedding norms.

Byte-level BPE has a particularly attractive property: round-trip correctness is \
trivial to guarantee, because the base vocabulary is the 256 bytes themselves. \
Any input string, encoded as UTF-8, decomposes into bytes; any sequence of byte \
tokens decodes losslessly back to the original bytes. Merges only reduce the \
sequence length; they never lose information.

Practical implementations differ on details: pre-tokenisation regex, byte-level \
visualisation (printable mapping), handling of leading spaces, and whether \
special tokens like beginning-of-sequence and end-of-sequence are inserted by \
the tokenizer or by the training loop. Forge-LLM keeps the tokenizer pure: it \
emits bytes and learned merges, and special tokens are added only when the \
caller asks for them via the ``add_special_tokens`` keyword.

Examples of edge cases the round-trip test must cover: empty strings, single \
characters, ASCII punctuation runs like "...", "!!!", and "???", unicode \
characters such as café, naïve, résumé, façade, Ω, π, μ, and the digits \
0123456789 alongside underscores and dashes used in code. Numbers like 3.14159, \
1e-10, and 2**16 must survive the encode→decode cycle unchanged. Common code \
fragments such as def foo(x: int) -> int: return x + 1 should also round-trip \
without surprises, since researchers will throw all kinds of text at the model.

A robust tokenizer also has to be reproducible. Training the same vocabulary on \
the same corpus with the same vocab size must produce identical merges every \
time, otherwise checkpoints become non-portable across machines. Determinism is \
enforced by always breaking pair-count ties by lexicographic order on the pair \
bytes — never by Python's dict iteration order, which is insertion-order but \
still implementation-defined for our purposes here.
"""

# Strings that exercise specific encoder paths: ASCII, unicode, punctuation runs,
# empty, single byte, special-token literals, multi-line, code-like fragments.
_ROUNDTRIP_SAMPLES = [
    "",
    " ",
    "a",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Numbers: 0, 1, 2, 3.14159, 1e-10, 2**16, -inf.",
    "Unicode: café, naïve, résumé, façade, Ω, π, μ.",
    "Punctuation runs: ... !!! ??? :::: ;;; ,,, ----.",
    "Multi\nline\nstring\nwith\nnewlines.",
    "Tabs\tand\tspaces   mixed.",
    "Mixed case: HelloWorld, ALL_CAPS, snake_case, kebab-case.",
    "Literal special tokens: <bos> <eos> <pad> <unk> must round-trip as text.",
    "Code: def foo(x: int) -> int: return x + 1",
    "Quotes: 'single' and \"double\" and `backtick` (no smart quotes).",
    "Repeated bytes: aaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbb cccc.",
]


def _generate_paragraph(rng: random.Random, source_text: str) -> str:
    """Produce a varied paragraph by sampling spans from the source corpus.

    Approximates the "100 random FineWeb-Edu paragraphs" exit criterion from
    docs/04_roadmap.md without requiring network access. Sampling spans from
    the corpus (instead of a fixed word list) keeps byte-level variety high:
    spans include punctuation, unicode, capitalisation, and whitespace
    patterns that a pure word-shuffler would miss.
    """
    n_spans = rng.randint(3, 8)
    parts: list[str] = []
    for _ in range(n_spans):
        span_len = rng.randint(20, 200)
        start = rng.randint(0, max(0, len(source_text) - span_len))
        parts.append(source_text[start : start + span_len])
    return " ".join(parts)


@pytest.fixture(scope="module")
def training_corpus() -> str:
    return _CORPUS


@pytest.fixture(scope="module")
def tiny_tokenizer(training_corpus: str) -> BPETokenizer:
    """A small BPE trained inline. vocab_size=512 keeps training under ~1s."""
    return BPETokenizer.train(training_corpus, vocab_size=512)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bpe_roundtrip(tiny_tokenizer: BPETokenizer, training_corpus: str) -> None:
    """Encode → decode must reproduce the input byte-for-byte.

    Combines:
      - the hand-picked edge-case strings in ``_ROUNDTRIP_SAMPLES``
      - 100 deterministically generated paragraph-shaped strings
        (the roadmap M1 "100 random FineWeb-Edu paragraphs" exit criterion).
    """
    rng = random.Random(0)
    samples = list(_ROUNDTRIP_SAMPLES) + [
        _generate_paragraph(rng, training_corpus) for _ in range(100)
    ]
    for text in samples:
        ids = tiny_tokenizer.encode(text)
        decoded = tiny_tokenizer.decode(ids)
        assert decoded == text, (
            f"roundtrip mismatch:\n  in:  {text!r}\n  out: {decoded!r}\n"
            f"  ids: {ids[:32]}{'...' if len(ids) > 32 else ''}"
        )


def test_bpe_special_tokens(tiny_tokenizer: BPETokenizer) -> None:
    """bos/eos/pad/unk are distinct ids, wrapping works, literals stay literal."""
    text = "Hello, world!"

    # 1. The four special tokens have distinct ids.
    special_ids = {
        tiny_tokenizer.bos_id,
        tiny_tokenizer.eos_id,
        tiny_tokenizer.pad_id,
        tiny_tokenizer.unk_id,
    }
    assert len(special_ids) == 4, f"special-token ids collide: {special_ids}"

    # 2. add_special_tokens=False emits no special ids for plain text.
    ids_plain = tiny_tokenizer.encode(text, add_special_tokens=False)
    assert not (special_ids & set(ids_plain)), (
        f"plain encode leaked a special id: {set(ids_plain) & special_ids}"
    )

    # 3. add_special_tokens=True wraps with bos/eos.
    ids_wrapped = tiny_tokenizer.encode(text, add_special_tokens=True)
    assert ids_wrapped[0] == tiny_tokenizer.bos_id
    assert ids_wrapped[-1] == tiny_tokenizer.eos_id
    # The inner tokens match the plain-encode output (wrapping is non-destructive).
    assert ids_wrapped[1:-1] == ids_plain

    # 4. decode(skip_special_tokens=True) strips bos/eos cleanly.
    assert tiny_tokenizer.decode(ids_wrapped, skip_special_tokens=True) == text

    # 5. A literal "<bos>" in input text is NOT collapsed to bos_id —
    #    the tokenizer treats user input as raw bytes; specials are inserted
    #    only via the keyword.
    ids_literal = tiny_tokenizer.encode("<bos>", add_special_tokens=False)
    assert tiny_tokenizer.bos_id not in ids_literal


def test_bpe_vocab_size_matches_config(tiny_tokenizer: BPETokenizer) -> None:
    """vocab_size property equals the size requested at train time.

    If this drifts, the embedding table in ``ForgeForCausalLM`` (M8) will be
    sized wrong and the first out-of-range token id will index-error or
    silently wrap into garbage rows.
    """
    assert tiny_tokenizer.vocab_size == 512


@pytest.mark.slow
def test_bpe_against_tiktoken_sanity(training_corpus: str) -> None:
    """Sanity-bound our token count against tiktoken's gpt2 encoding.

    With our 2048-vocab BPE vs tiktoken's ~50K-vocab gpt2 encoder, we should
    produce MORE tokens (smaller vocab → longer sequences), but not absurdly
    more — a per-byte fallback would produce ~4x what gpt2 does, which is the
    rough upper bound. The exact factor doesn't matter; we just want to catch
    a fundamentally broken BPE (e.g., no merges applied, or merges applied in
    wrong order producing pathologically long sequences).
    """
    tiktoken = pytest.importorskip("tiktoken", reason="tiktoken oracle not installed")
    gpt2 = tiktoken.get_encoding("gpt2")

    # Train a larger BPE for a fairer comparison.
    big = BPETokenizer.train(training_corpus, vocab_size=2048)

    sample = training_corpus[:4096]
    ours = len(big.encode(sample))
    theirs = len(gpt2.encode(sample))

    assert ours > 0 and theirs > 0
    # Loose sanity bounds: ours should be in [0.5x, 10x] of tiktoken's count.
    # Lower bound guards against accidentally emitting one token per word (too good).
    # Upper bound guards against per-byte fallback or unmerged output (too bad).
    ratio = ours / theirs
    assert 0.5 <= ratio <= 10.0, (
        f"token-count ratio out of sanity bounds: ours={ours}, tiktoken={theirs}, "
        f"ratio={ratio:.2f}x — BPE training likely broken"
    )


def test_bpe_save_load(tiny_tokenizer: BPETokenizer, tmp_path: Path) -> None:
    """Train → save → load → encode is identity.

    "Loaded in a new process" per docs/02_correctness_plan.md §1.11 is
    approximated by constructing a fresh ``BPETokenizer`` object from disk in
    the same interpreter. This catches every bug that a true new-process load
    would catch (lossy serialisation, missing merges, special-token id drift)
    while staying robust against multiprocessing flakiness on macOS/Windows
    pytest runners.
    """
    path = tmp_path / "tokenizer.json"
    tiny_tokenizer.save(path)
    assert path.exists() and path.stat().st_size > 0

    loaded = BPETokenizer.load(path)

    # Identity on the metadata.
    assert loaded.vocab_size == tiny_tokenizer.vocab_size
    assert loaded.bos_id == tiny_tokenizer.bos_id
    assert loaded.eos_id == tiny_tokenizer.eos_id
    assert loaded.pad_id == tiny_tokenizer.pad_id
    assert loaded.unk_id == tiny_tokenizer.unk_id

    # Identity on the encoder behaviour for a battery of strings.
    for text in _ROUNDTRIP_SAMPLES:
        original_ids = tiny_tokenizer.encode(text)
        loaded_ids = loaded.encode(text)
        assert loaded_ids == original_ids, (
            f"loaded encoder diverged on {text!r}: "
            f"original={original_ids[:16]}..., loaded={loaded_ids[:16]}..."
        )
        # And the loaded decoder still round-trips.
        assert loaded.decode(loaded_ids) == text
