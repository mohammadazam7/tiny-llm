"""``tiny-llm``: the command line for preparing data (and, later, training and sampling)."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tiny_llm.data import DataError, prepare


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tiny-llm")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="train a tokenizer and write token shards")
    p.add_argument("corpus", type=Path, help="a UTF-8 text file")
    p.add_argument("--out", type=Path, default=Path("data/tokens"))
    p.add_argument("--vocab-size", type=int, default=512)
    p.add_argument("--validation-share", type=float, default=0.1)
    args = parser.parse_args(argv)
    try:
        result = prepare(args.corpus, args.out, args.vocab_size, args.validation_share)
    except (DataError, OSError, UnicodeDecodeError) as err:
        print(f"tiny-llm: {err}")
        return 1
    print(
        f"vocabulary {result.vocab_size}, {result.train_tokens} training and "
        f"{result.validation_tokens} validation tokens, "
        f"{result.characters_per_token:.2f} characters per token, written to {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
