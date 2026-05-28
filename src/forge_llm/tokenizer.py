"""From-scratch byte-level BPE tokenizer (M1).

The base vocabulary is the 256 raw bytes; on top of that we place four special
tokens (bos, eos, pad, unk) at fixed ids 256..259, and then learned merges fill
ids 260..vocab_size-1. The arrangement makes encode -> decode byte-identical on
any UTF-8 input: every byte already has its own id, so merges only ever
compress, never lose information.

The implementation does not import ``transformers``, ``tiktoken`` or any other
tokenizer library -- those are reserved for tests as oracles (CLAUDE.md sec 2).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Final

_BASE_VOCAB: Final[int] = 256
_BOS_ID: Final[int] = 256
_EOS_ID: Final[int] = 257
_PAD_ID: Final[int] = 258
_UNK_ID: Final[int] = 259
_NUM_SPECIAL: Final[int] = 4
_MIN_VOCAB: Final[int] = _BASE_VOCAB + _NUM_SPECIAL

_BOS_TEXT: Final[str] = "<bos>"
_EOS_TEXT: Final[str] = "<eos>"
_PAD_TEXT: Final[str] = "<pad>"
_UNK_TEXT: Final[str] = "<unk>"

# Splits text into ``non-whitespace + trailing whitespace`` chunks (and bare
# whitespace runs). Merges only happen within a chunk so the BPE never learns
# cross-word merges -- the same trick GPT-2 uses. The alternation covers every
# character in any input, so concatenating the chunks reproduces the input.
_PRETOKEN_RE: Final[re.Pattern[str]] = re.compile(r"\S+\s*|\s+")

_FORMAT_VERSION: Final[int] = 1
_SPECIAL_IDS: Final[frozenset[int]] = frozenset({_BOS_ID, _EOS_ID, _PAD_ID, _UNK_ID})


class BPETokenizer:
    """Byte-level BPE tokenizer (train / encode / decode / save / load)."""

    def __init__(
        self,
        vocab_size: int,
        merges: Sequence[tuple[int, int]],
    ) -> None:
        if vocab_size < _MIN_VOCAB:
            raise ValueError(
                f"vocab_size must be >= {_MIN_VOCAB} "
                f"(256 base bytes + 4 special tokens), got {vocab_size}"
            )
        self._vocab_size: int = vocab_size
        self._merges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in merges]
        self._merge_rank: dict[tuple[int, int], int] = {
            pair: i for i, pair in enumerate(self._merges)
        }
        self._id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(_BASE_VOCAB)}
        self._id_to_bytes[_BOS_ID] = _BOS_TEXT.encode("utf-8")
        self._id_to_bytes[_EOS_ID] = _EOS_TEXT.encode("utf-8")
        self._id_to_bytes[_PAD_ID] = _PAD_TEXT.encode("utf-8")
        self._id_to_bytes[_UNK_ID] = _UNK_TEXT.encode("utf-8")
        for i, (a, b) in enumerate(self._merges):
            new_id = _MIN_VOCAB + i
            self._id_to_bytes[new_id] = self._id_to_bytes[a] + self._id_to_bytes[b]

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def bos_id(self) -> int:
        return _BOS_ID

    @property
    def eos_id(self) -> int:
        return _EOS_ID

    @property
    def pad_id(self) -> int:
        return _PAD_ID

    @property
    def unk_id(self) -> int:
        return _UNK_ID

    @classmethod
    def train(
        cls,
        corpus: str | Iterable[str],
        vocab_size: int,
    ) -> BPETokenizer:
        """Learn BPE merges from ``corpus`` and return a tokenizer.

        Tie-breaking on pair frequency is by lexicographic order on the
        ``(a, b)`` id tuple, never by ``dict`` iteration order (CLAUDE.md sec
        11). With identical inputs this produces identical merges on every run
        and every machine.
        """
        if vocab_size < _MIN_VOCAB:
            raise ValueError(
                f"vocab_size must be >= {_MIN_VOCAB} "
                f"(256 base bytes + 4 special tokens), got {vocab_size}"
            )

        chunks: Iterable[str] = [corpus] if isinstance(corpus, str) else corpus

        word_counts: Counter[tuple[int, ...]] = Counter()
        for chunk in chunks:
            for word in _PRETOKEN_RE.findall(chunk):
                word_counts[tuple(word.encode("utf-8"))] += 1

        unique_words: list[tuple[int, ...]] = list(word_counts.keys())
        words: list[list[int]] = [list(w) for w in unique_words]
        counts: list[int] = [word_counts[w] for w in unique_words]

        merges: list[tuple[int, int]] = []
        n_merges_target = vocab_size - _MIN_VOCAB

        for _ in range(n_merges_target):
            pair_counts: Counter[tuple[int, int]] = Counter()
            for word, c in zip(words, counts, strict=True):
                for j in range(len(word) - 1):
                    pair_counts[(word[j], word[j + 1])] += c
            if not pair_counts:
                break

            best_pair: tuple[int, int] = max(
                pair_counts.items(),
                key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]),
            )[0]
            new_id = _MIN_VOCAB + len(merges)
            merges.append(best_pair)
            best_a, best_b = best_pair

            for w_idx, word in enumerate(words):
                n = len(word)
                if n < 2:
                    continue
                merged: list[int] = []
                i = 0
                while i < n:
                    if (
                        i + 1 < n
                        and word[i] == best_a
                        and word[i + 1] == best_b
                    ):
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[w_idx] = merged

        return cls(vocab_size=vocab_size, merges=merges)

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Encode ``text`` to a list of token ids.

        ``add_special_tokens=True`` wraps the output with ``[bos_id, ..., eos_id]``.
        A literal ``"<bos>"`` substring in ``text`` is *not* collapsed to
        ``bos_id``; special tokens are emitted only via this keyword.
        """
        ids: list[int] = []
        if add_special_tokens:
            ids.append(_BOS_ID)
        for word in _PRETOKEN_RE.findall(text):
            ids.extend(self._encode_word(word.encode("utf-8")))
        if add_special_tokens:
            ids.append(_EOS_ID)
        return ids

    def _encode_word(self, raw: bytes) -> list[int]:
        if not raw:
            return []
        tokens: list[int] = list(raw)
        while len(tokens) >= 2:
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                rank = self._merge_rank.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            new_id = _MIN_VOCAB + best_rank
            tokens = [*tokens[:best_idx], new_id, *tokens[best_idx + 2 :]]
        return tokens

    def decode(
        self,
        ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode a sequence of token ids back to text."""
        parts: list[bytes] = []
        for tid in ids:
            if skip_special_tokens and tid in _SPECIAL_IDS:
                continue
            parts.append(self._id_to_bytes[tid])
        return b"".join(parts).decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Write the tokenizer to ``path`` as JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "vocab_size": self._vocab_size,
            "merges": [[a, b] for (a, b) in self._merges],
        }
        out.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer previously written by :meth:`save`."""
        src = Path(path)
        payload = json.loads(src.read_text())
        version = payload.get("version")
        if version != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported tokenizer file version: {version} "
                f"(expected {_FORMAT_VERSION})"
            )
        vocab_size = int(payload["vocab_size"])
        merges = [(int(a), int(b)) for a, b in payload["merges"]]
        return cls(vocab_size=vocab_size, merges=merges)
