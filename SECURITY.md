# Security policy

## Reporting a problem

Please report security issues privately through GitHub's
[security advisories](https://github.com/mohammadazam7/tiny-llm/security/advisories/new)
rather than in a public issue. Expect an acknowledgement within a few days.

## Checkpoints

PyTorch checkpoints are pickle files, and loading a pickle from an untrusted source can
run code. Only load checkpoints you produced or trust. The loader in this project uses
`weights_only=True`, and the tokenizer is stored as JSON for the same reason.
