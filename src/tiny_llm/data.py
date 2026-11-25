"""Turn a text corpus into token shards a training loop can read quickly.

The held-out text is the end of the corpus, cut at a line break, so validation lines are
never neighbours of training lines in the same batch. The tokenizer is trained on the
training part only: if it saw the validation text, its merges would already fit the text
the model is scored on.

Shards are flat files of little-endian unsigned 16-bit token ids, so a vocabulary must
stay under 65,536. A small JSON file next to them records what produced them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from tiny_llm.tokenizer import Tokenizer

MAX_VOCAB = 2**16
META = "meta.json"


class DataError(ValueError):
    pass


def split_text(text: str, validation_share: float = 0.1) -> tuple[str, str]:
    """Training and validation text: the last ``validation_share`` of characters, moved
    forward to the next line break so no line is split between the two."""
    if not 0 < validation_share < 1:
        raise DataError("validation_share must be between 0 and 1")
    cut = int(len(text) * (1 - validation_share))
    newline = text.find("\n", cut)
    if newline == -1 or newline == len(text) - 1:
        raise DataError("no line break after the cut point; the text is too short to split")
    return text[: newline + 1], text[newline + 1 :]


def _to_array(ids: Sequence[int]) -> array[int]:
    tokens = array("H", ids)
    if sys.byteorder != "little":
        tokens.byteswap()
    return tokens


def write_tokens(ids: Sequence[int], path: Path) -> None:
    path.write_bytes(_to_array(ids).tobytes())


def read_tokens(path: Path) -> array[int]:
    tokens = array("H")
    tokens.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        tokens.byteswap()
    return tokens


@dataclass(frozen=True)
class Prepared:
    vocab_size: int
    train_tokens: int
    validation_tokens: int
    train_characters: int
    validation_characters: int

    @property
    def characters_per_token(self) -> float:
        return self.train_characters / self.train_tokens


def prepare(corpus: Path, out: Path, vocab_size: int, validation_share: float = 0.1) -> Prepared:
    """Split the corpus, train a tokenizer on the training part, and write the shards."""
    if vocab_size > MAX_VOCAB:
        raise DataError(f"vocab_size must be at most {MAX_VOCAB} to fit 16-bit shards")
    raw = corpus.read_bytes()
    text = raw.decode("utf-8")
    train, validation = split_text(text, validation_share)
    tokenizer = Tokenizer.train(train, vocab_size)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out / "tokenizer.json")
    train_ids = tokenizer.encode(train)
    validation_ids = tokenizer.encode(validation)
    write_tokens(train_ids, out / "train.bin")
    write_tokens(validation_ids, out / "validation.bin")
    result = Prepared(
        vocab_size=tokenizer.vocab_size,
        train_tokens=len(train_ids),
        validation_tokens=len(validation_ids),
        train_characters=len(train),
        validation_characters=len(validation),
    )
    meta = {
        "source": corpus.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "validation_share": validation_share,
        "vocab_size": result.vocab_size,
        "train_tokens": result.train_tokens,
        "validation_tokens": result.validation_tokens,
        "train_characters": result.train_characters,
        "validation_characters": result.validation_characters,
        "dtype": "uint16-le",
    }
    (out / META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return result
