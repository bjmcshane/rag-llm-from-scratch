# src/inference/

Text generation from the trained model. **Not started.**

Planned:

- Load a checkpoint from [src/training/](../training/) and the tokenizer from
  [src/tokenizer/](../tokenizer/).
- Sampling (temperature, top-k / top-p) with a KV cache.
- RAG mode: prepend passages retrieved by [src/rag/](../rag/) to the prompt.

Python/PyTorch first; a Rust server with `candle` is a stretch goal (see
[RUST_PYTHON.md](../../RUST_PYTHON.md)).
