"""A byte-level byte-pair-encoding tokenizer.

Training starts from the 256 byte values and repeatedly merges the most frequent
adjacent pair into a new token, recording the merges in order. Encoding replays those
merges in the same order. Because the base vocabulary is every possible byte, any text
encodes without an unknown token and decodes back to exactly the bytes that went in.

Text is first split into chunks (runs of letters, runs of digits, single other
characters, with a leading space kept on the word it precedes) and merges never cross a
chunk boundary. That keeps tokens from straddling words and makes training count each
distinct chunk once instead of scanning the whole corpus on every merge.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path

Pair = tuple[int, int]

BASE_VOCAB = 256
FORMAT_VERSION = 1
# A word with its leading space, a run of digits, a run of whitespace, or one other char.
_CHUNK = re.compile(r" ?[^\W\d_]+| ?\d+|\s+|.", re.DOTALL)


class TokenizerError(ValueError):
    pass


def _chunks(text: str) -> list[bytes]:
    return [m.group().encode("utf-8") for m in _CHUNK.finditer(text)]


def _pair_counts(words: dict[tuple[int, ...], int]) -> Counter[Pair]:
    counts: Counter[Pair] = Counter()
    for word, freq in words.items():
        for pair in pairwise(word):
            counts[pair] += freq
    return counts


def _merge(word: tuple[int, ...], pair: Pair, new_id: int) -> tuple[int, ...]:
    out = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class Tokenizer:
    """Encode text to token ids and back. Build one with :meth:`train` or :meth:`load`."""

    def __init__(self, merges: Iterable[Pair]) -> None:
        self.merges: list[Pair] = [(int(a), int(b)) for a, b in merges]
        self._rank: dict[Pair, int] = {}
        self._bytes: list[bytes] = [bytes([i]) for i in range(BASE_VOCAB)]
        for pair in self.merges:
            new_id = len(self._bytes)
            if pair in self._rank or max(pair) >= new_id or min(pair) < 0:
                raise TokenizerError(f"invalid merge {pair} at position {len(self._rank)}")
            self._rank[pair] = new_id
            self._bytes.append(self._bytes[pair[0]] + self._bytes[pair[1]])
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._bytes)

    @classmethod
    def train(cls, text: str, vocab_size: int) -> Tokenizer:
        """Learn ``vocab_size - 256`` merges from ``text``.

        Ties between equally frequent pairs go to the smaller pair, so the same text
        always gives the same tokenizer. Training stops early when no pair occurs twice.
        """
        if vocab_size < BASE_VOCAB:
            raise TokenizerError(f"vocab_size must be at least {BASE_VOCAB}")
        words: dict[tuple[int, ...], int] = {
            tuple(chunk): freq for chunk, freq in Counter(_chunks(text)).items()
        }
        merges: list[Pair] = []
        for new_id in range(BASE_VOCAB, vocab_size):
            counts = _pair_counts(words)
            if not counts:
                break
            pair, freq = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if freq < 2:
                break
            merges.append(pair)
            merged: dict[tuple[int, ...], int] = {}
            for word, count in words.items():
                new_word = _merge(word, pair, new_id)
                merged[new_word] = merged.get(new_word, 0) + count
            words = merged
        return cls(merges)

    def _encode_chunk(self, chunk: bytes) -> tuple[int, ...]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        word = tuple(chunk)
        while len(word) > 1:
            # Apply the earliest-learned merge present; that is the order training used.
            ranked = [
                (self._rank[pair], pair) for pair in set(pairwise(word)) if pair in self._rank
            ]
            if not ranked:
                break
            new_id, pair = min(ranked)
            word = _merge(word, pair, new_id)
        self._cache[chunk] = word
        return word

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in _chunks(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        try:
            return b"".join(self._bytes[i] for i in ids)
        except IndexError as err:
            raise TokenizerError("token id outside the vocabulary") from err

    def decode(self, ids: Iterable[int]) -> str:
        """Text for ``ids``. A sequence cut in the middle of a multi-byte character
        (as sampling can do) decodes with a replacement character instead of failing."""
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    def token_bytes(self, token_id: int) -> bytes:
        return self._bytes[token_id]

    def save(self, path: Path) -> None:
        """Write the merges as JSON: readable, diffable, and safe to load."""
        payload = {"format": FORMAT_VERSION, "base_vocab": BASE_VOCAB, "merges": self.merges}
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT_VERSION or payload.get("base_vocab") != BASE_VOCAB:
            raise TokenizerError(f"{path} is not a tokenizer file this version can read")
        return cls((a, b) for a, b in payload["merges"])
