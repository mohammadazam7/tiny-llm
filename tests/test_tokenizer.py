import random
from pathlib import Path

import pytest

from tiny_llm.tokenizer import BASE_VOCAB, Tokenizer, TokenizerError

CORPUS = (
    "the cat sat on the mat. the cat ate the rat. "
    "that hat is the cat's hat, and the mat is the rat's mat. "
) * 20


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer.train(CORPUS, vocab_size=300)


def test_with_no_merges_tokens_are_the_utf8_bytes() -> None:
    plain = Tokenizer([])
    assert plain.vocab_size == BASE_VOCAB
    assert plain.encode("hé") == list("hé".encode())


def test_training_learns_the_most_frequent_pair_first() -> None:
    # "ab" occurs 5 times, "bc" 3 times; the first new token must be "ab".
    learned = Tokenizer.train("ab ab ab abc abc bc", vocab_size=257)
    assert learned.merges == [(ord("a"), ord("b"))]
    assert learned.token_bytes(256) == b"ab"


def test_ties_are_broken_the_same_way_every_time() -> None:
    first = Tokenizer.train("xy xy pq pq", vocab_size=258)
    again = Tokenizer.train("xy xy pq pq", vocab_size=258)
    assert first.merges == again.merges
    # Four pairs occur twice each; the smallest pair is merged first.
    assert first.merges[0] == (ord(" "), ord("p"))


def test_training_stops_when_nothing_repeats() -> None:
    assert Tokenizer.train("abcdefg", vocab_size=400).merges == []


def test_frequent_words_become_single_tokens(tokenizer: Tokenizer) -> None:
    assert len(tokenizer.encode(" the")) == 1
    assert len(tokenizer.encode(" cat")) == 1
    assert tokenizer.vocab_size <= 300


def test_encoding_compresses_the_training_text(tokenizer: Tokenizer) -> None:
    assert len(tokenizer.encode(CORPUS)) < len(CORPUS.encode()) / 2


def test_merges_never_cross_a_word_boundary(tokenizer: Tokenizer) -> None:
    for token_id in range(BASE_VOCAB, tokenizer.vocab_size):
        piece = tokenizer.token_bytes(token_id).decode()
        assert " " not in piece[1:], piece


@pytest.mark.parametrize(
    "text",
    [
        "",
        "the cat sat",
        "Unseen WORDS, with punctuation!? And numbers: 12345.",
        "tabs\tand\nnewlines\r\n  and   runs of spaces ",
        "naïve café, 日本語, emoji 🙂, mixed: ünïcödé",
        "\x00\x01 control bytes \x7f",
    ],
)
def test_any_text_round_trips_exactly(tokenizer: Tokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_random_unicode_round_trips(tokenizer: Tokenizer) -> None:
    rng = random.Random(7)
    for _ in range(200):
        text = "".join(
            chr(rng.choice([rng.randrange(32, 127), rng.randrange(160, 0x2FFF)]))
            for _ in range(rng.randrange(0, 40))
        )
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_ids_stay_inside_the_vocabulary(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode("the cat and some unseen text 🙂")
    assert all(0 <= i < tokenizer.vocab_size for i in ids)


def test_a_sequence_cut_inside_a_character_still_decodes(tokenizer: Tokenizer) -> None:
    ids = tokenizer.encode("é")
    assert tokenizer.decode(ids[:1]) == "�"
    assert tokenizer.decode_bytes(ids) == "é".encode()


def test_an_unknown_id_is_an_error(tokenizer: Tokenizer) -> None:
    with pytest.raises(TokenizerError, match="outside the vocabulary"):
        tokenizer.decode([tokenizer.vocab_size])


def test_a_saved_tokenizer_loads_to_the_same_behaviour(
    tokenizer: Tokenizer, tmp_path: Path
) -> None:
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tokenizer.merges
    text = "the cat sat on a brand new mat"
    assert loaded.encode(text) == tokenizer.encode(text)


def test_bad_files_and_bad_merges_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"format": 99, "merges": []}', encoding="utf-8")
    with pytest.raises(TokenizerError, match="not a tokenizer file"):
        Tokenizer.load(path)
    with pytest.raises(TokenizerError, match="invalid merge"):
        Tokenizer([(300, 301)])  # refers to tokens that do not exist yet
    with pytest.raises(TokenizerError, match="invalid merge"):
        Tokenizer([(97, 98), (97, 98)])
    with pytest.raises(TokenizerError, match="at least 256"):
        Tokenizer.train("abc", vocab_size=100)
