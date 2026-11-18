#!/usr/bin/env bash
# Download the training text, pinned to one commit and checked against its checksum.
#
# Tiny Shakespeare: about 1.1 MB of plays by William Shakespeare (public domain), as
# collected in Andrej Karpathy's char-rnn repository. Project Gutenberg would be the
# natural source, but it is not reachable from every build environment; this copy is.
set -euo pipefail

COMMIT="6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
SHA256="86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
URL="https://raw.githubusercontent.com/karpathy/char-rnn/${COMMIT}/data/tinyshakespeare/input.txt"
OUT="${1:-data/raw/tinyshakespeare.txt}"

mkdir -p "$(dirname "$OUT")"
curl --fail --silent --show-error --location -o "$OUT.part" "$URL"
echo "${SHA256}  $OUT.part" | sha256sum --check --quiet
mv "$OUT.part" "$OUT"
echo "wrote $OUT ($(wc -c < "$OUT") bytes)"
