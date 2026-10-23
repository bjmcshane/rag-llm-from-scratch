# Rust vs. Python: what goes where

**Rule of thumb:** Python for anything that touches the GPU or changes often; Rust for CPU-bound passes over every byte of the corpus. Bridge the two with [PyO3](https://pyo3.rs) + [maturin](https://www.maturin.rs), so Rust modules import into Python like any other package.

## Split by component

| Component | Language | Why |
|---|---|---|
| `src/data_fetch/` | **Python** | Network and rate-limit bound, not CPU bound. Rust won't speed it up. Use `asyncio`/`httpx` if throughput matters. |
| Cleaning and dedup (new, or in `src/utils/`) | **Rust** | The first real hot path: Common Crawl HTML to text extraction, language and quality filtering, MinHash/LSH near-duplicate removal, exact-hash dedup. Runs over GBs to TBs. Rust is often 10–50× faster and parallelizes across cores without fighting the GIL. |
| `src/tokenizer/` | **Rust** core + Python bindings | BPE training and encoding are tight loops over huge text, and it's the best self-contained Rust learning project here (Hugging Face's `tokenizers` takes the same approach). Write text-to-packed-`uint16`/`uint32`-token-shards in Rust too, so training can memory-map them. |
| `src/training/` | **Python** (PyTorch) | The compute runs in CUDA kernels either way. Rust would cost autograd tooling, FSDP/DDP, `torch.compile`, mixed precision and the debugging ecosystem. `candle`/`burn` can train, but it's the hard road for a first model. |
| Embedding model (retrieval) | **Python** | Same reasons as training: a small encoder trained or fine-tuned next to the LM. |
| `src/rag/` retrieval index | **Rust**, or an existing library | Building HNSW yourself is a good Rust project (memory layout and SIMD matter). The pragmatic choice is FAISS or `usearch` from Python. Keep RAG orchestration (chunking, prompt assembly, reranking) in Python. |
| `src/inference/` | **Python** first, optionally Rust later | Get generation working in PyTorch. A Rust server with `candle` (KV cache, batching, sampling) is a good stretch goal. |
| `src/utils/` (eval, configs, plotting) | **Python** | Glue code that changes often. |

## Sequencing

1. Build everything in Python first, including a naive tokenizer and dedup, so the pipeline runs start to finish on a small corpus.
2. Profile, then port the slow parts to Rust, likely in this order:
   1. Tokenizer
   2. Dedup and filtering
   3. Shard packing
3. Test each Rust module against the Python reference it replaces, so you're never debugging two languages at once.

## "Training a RAG LLM" is three projects

1. **Pretrain the language model.**
2. **Train or fine-tune an embedding model** for retrieval.
3. **Optionally fine-tune the LM to use retrieved context**, e.g. RETRO- or RA-DIT-style training on (query, retrieved passages, answer) examples.

All three training stages belong in Python. Rust's job is feeding them clean, deduplicated, tokenized data quickly, and serving retrieval fast.
