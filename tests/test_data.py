import json
from pathlib import Path

import pytest

from tiny_llm.cli import main
from tiny_llm.data import DataError, prepare, read_tokens, split_text, write_tokens
from tiny_llm.tokenizer import Tokenizer

TEXT = "".join(f"line {i}: the cat sat on the mat.\n" for i in range(200))


def test_the_split_falls_on_a_line_break_at_the_end() -> None:
    train, validation = split_text(TEXT, 0.1)
    assert train + validation == TEXT
    assert train.endswith("\n") and validation.startswith("line ")
    assert 0.09 < len(validation) / len(TEXT) < 0.11


def test_a_text_that_cannot_be_split_is_refused() -> None:
    with pytest.raises(DataError, match="too short"):
        split_text("one line without a break", 0.5)
    with pytest.raises(DataError, match="between 0 and 1"):
        split_text(TEXT, 0.0)


def test_tokens_round_trip_through_a_shard(tmp_path: Path) -> None:
    ids = [0, 1, 255, 256, 65535]
    write_tokens(ids, tmp_path / "t.bin")
    assert (tmp_path / "t.bin").stat().st_size == 2 * len(ids)
    assert list(read_tokens(tmp_path / "t.bin")) == ids


def test_prepare_trains_on_training_text_only(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    # The held-out tail uses a word that never appears earlier.
    corpus.write_text(TEXT + "zebra zebra zebra zebra zebra\n" * 30, encoding="utf-8")
    result = prepare(corpus, tmp_path / "out", vocab_size=320, validation_share=0.2)
    train, _ = split_text(corpus.read_text(encoding="utf-8"), 0.2)
    assert "zebra" not in train
    tokenizer = Tokenizer.load(tmp_path / "out" / "tokenizer.json")
    learned = {tokenizer.token_bytes(i) for i in range(256, tokenizer.vocab_size)}
    assert not any(b"zebra" in piece or b"zeb" in piece for piece in learned)
    assert result.validation_tokens > 0


def test_prepare_writes_shards_that_decode_back_to_the_text(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    out = tmp_path / "out"
    result = prepare(corpus, out, vocab_size=300)
    tokenizer = Tokenizer.load(out / "tokenizer.json")
    train, validation = split_text(TEXT, 0.1)
    assert tokenizer.decode(read_tokens(out / "train.bin")) == train
    assert tokenizer.decode(read_tokens(out / "validation.bin")) == validation
    meta = json.loads((out / "meta.json").read_text())
    assert meta["train_tokens"] == result.train_tokens == len(read_tokens(out / "train.bin"))
    assert meta["dtype"] == "uint16-le" and len(meta["source_sha256"]) == 64
    assert result.characters_per_token > 1


def test_a_vocabulary_too_large_for_16_bits_is_refused(tmp_path: Path) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    with pytest.raises(DataError, match="16-bit"):
        prepare(corpus, tmp_path / "out", vocab_size=70000)


def test_the_prepare_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus = tmp_path / "c.txt"
    corpus.write_text(TEXT, encoding="utf-8")
    assert main(["prepare", str(corpus), "--out", str(tmp_path / "o"), "--vocab-size", "300"]) == 0
    assert "characters per token" in capsys.readouterr().out
    assert main(["prepare", str(tmp_path / "missing.txt")]) == 1
