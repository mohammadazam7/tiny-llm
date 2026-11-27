# Roadmap

Target: v1.0 between **Dec 21 and Dec 23, 2026**. Each milestone ends with a pull
request, a tagged release and a changelog entry.

- [x] **M0 (Nov 16): foundation.** Repository, CI, design doc, roadmap.
- [x] **M1 (Nov 17 to 24): tokenizer and data.** Byte-level BPE (train, encode, decode,
  save and load), corpus preparation from public-domain text, token shards with a
  held-out validation split.
- [x] **M2 (Nov 26): the model.** Decoder-only transformer with causal
  multi-head attention, pre-norm blocks and tied embeddings; tests for shapes, parameter
  counts and the causal mask.
- [ ] **M3 (Dec 4 to 11): training and sampling.** AdamW with warmup and cosine decay,
  gradient accumulation and clipping, mixed precision where supported, resumable
  checkpoints, evaluation, and temperature, top-k and nucleus sampling.
- [ ] **M4 (Dec 12 to 18): scaling experiments.** Several model sizes on the same data
  and budget; parameters against validation loss with charts and the hardware stated;
  the Attention Theater web app for looking inside a trained model.
- [ ] **M5 (Dec 19 to 23): v1.0.** Model card, final design doc, checkpoint format
  documented for llm-inference-server, release.
