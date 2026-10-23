# src/rag/

Retrieval for the RAG model. **Not started.**

Planned:

- Chunk cleaned documents into passages and embed them with a small encoder (trained or
  fine-tuned in Python).
- A vector index: FAISS or `usearch` from Python to start, optionally a hand-written HNSW in
  Rust later (see [RUST_PYTHON.md](../../RUST_PYTHON.md)).
- Orchestration in Python: query → retrieve → rerank → assemble the prompt for
  [src/inference/](../inference/).
