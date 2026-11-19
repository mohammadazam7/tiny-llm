# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- Byte-level BPE tokenizer: train, encode, decode, save and load. Merges never
  cross a word boundary, and any text round-trips exactly through encode and decode.
- `tiny-llm prepare`: Tiny Shakespeare fetched at a pinned commit and checked against
  its SHA-256, split into training and held-out text before the tokenizer is trained,
  and written as uint16 token shards.
- Project foundation: package layout, CI, contribution and security guides, design doc
  and roadmap.
