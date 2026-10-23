# src/training/

Pretraining the language model in PyTorch. **Not started.**

Planned:

- Read memory-mapped `uint16` token shards produced by [src/tokenizer/](../tokenizer/) (docs
  separated by `<|endoftext|>`), sampling random fixed-length windows per batch.
- A small GPT-style decoder (pre-norm, RoPE, tied embeddings) sized to the tokenizer's vocab.
- AdamW with warmup + cosine decay, mixed precision, gradient accumulation, checkpointing.
- Logging to Weights & Biases (`wandb` is already a dependency).

This stays in Python: the compute runs in CUDA kernels either way (see
[RUST_PYTHON.md](../../RUST_PYTHON.md)).
