# Design: tiny-llm

Status: draft. Author: Mohammad Azam. Last updated: Nov 16, 2026.

## Context

Transformer language models are usually met as a finished product. The pieces (a
tokenizer, attention with a causal mask, a learning rate schedule, a sampling rule) are
each simple, and each has a classic way of being subtly wrong while training still
appears to work. This project builds the pieces at a size that can be trained and tested
on a CPU, so each claim about the model is backed by a test or a measurement.

## Goals

- A byte-level BPE tokenizer that round-trips any input exactly.
- A decoder-only transformer whose causal masking is proven by a test, not assumed.
- A training loop with warmup and cosine decay, gradient accumulation, clipping and
  checkpoints that resume to the same loss.
- Scaling experiments: several model sizes on the same data and step budget, reporting
  parameters, validation loss and wall-clock time, with the hardware stated.
- Checkpoints in a documented format that llm-inference-server can load.

## Non-goals

- Competing with any released model. The models here are a few million parameters.
- Distributed or multi-GPU training.
- Instruction tuning, chat formats or safety filtering.

## Design

**Tokenizer.** Byte-level BPE: start from the 256 byte values, repeatedly merge the most
frequent adjacent pair, record the merges in order. Encoding applies merges in the order
they were learned. Starting from bytes means there is no unknown token and decoding is
exact. The vocabulary and merges are saved as JSON so another program can load them.

**Data.** Public-domain books from Project Gutenberg, with the licence header and footer
removed, are tokenised once into binary shards of unsigned 16-bit ids. The last 10
percent of each book is held out for validation, so validation text is never adjacent in
a batch to training text from the same passage.

**Model.** Token embeddings plus learned position embeddings; N pre-norm blocks of causal
multi-head self-attention and a GELU MLP; a final layer norm; output weights tied to the
token embeddings. Configuration is one dataclass (layers, heads, width, context length,
vocabulary size, dropout).

**Training.** AdamW with weight decay applied to matrices only, linear warmup then
cosine decay, gradient accumulation to reach an effective batch size a CPU cannot hold
at once, gradient clipping, and mixed precision when the device supports it. A
checkpoint stores model, optimiser, step and configuration; resuming continues the same
run.

**Sampling.** Temperature, top-k and nucleus (top-p), with a seed, so a sample in the
README can be regenerated.

## Alternatives considered

- **A character-level model.** Simpler, but it skips the tokenizer, which is where much
  of the practical behaviour of language models comes from.
- **An existing tokenizer library.** Faster, but the tokenizer is part of what this
  project sets out to show and test.
- **Rotary or relative positions.** Better at long context; learned absolute positions
  are enough at a context of a few hundred tokens and keep the first version small.
  Rotary positions are a candidate experiment for M4.

## Testing plan

- Tokenizer: round trip on arbitrary bytes (property test), merges applied in learned
  order, a saved tokenizer loads to identical behaviour.
- Model: output shape, parameter count against a hand calculation, and the causal test:
  changing token t must not change any output before t.
- Training: the schedule matches its formula at chosen steps, loss falls on a tiny
  repeated dataset (the model can overfit one batch), resume gives the same next-step
  loss as an uninterrupted run.
- Sampling: with a fixed seed the output is identical; top-k never emits a token outside
  the top k.

## Open questions

- How large a model a CPU can train usefully in the time available; the first scaling
  run decides the range of sizes.
- Whether mixed precision gives any speedup on CPU at all, or belongs behind a flag that
  is only on for GPU users.
