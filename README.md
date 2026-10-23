# rag-llm-from-scratch

Train a retrieval-augmented LLM from scratch: corpus → tokenizer → pretraining → retrieval → inference.
Python for anything that touches the GPU or changes often, Rust for CPU-bound passes over the
whole corpus. [RUST_PYTHON.md](RUST_PYTHON.md) explains the split and the order things get built in.

## Layout

| Path | What's there | Status |
|---|---|---|
| [src/data_fetch/](src/data_fetch/) | Downloaders per source; Common Crawl extraction + filters | Common Crawl working, others are stubs |
| [src/tokenizer/](src/tokenizer/) | Baseline byte-level BPE (Python) + GPT-2 comparison | Baseline working |
| [src/training/](src/training/) | Pretraining loop (PyTorch) | Not started |
| [src/rag/](src/rag/) | Retrieval index + RAG orchestration | Not started |
| [src/inference/](src/inference/) | Generation / serving | Not started |
| [src/utils/](src/utils/) | Eval, configs, plotting | Not started |
| [rust/](rust/) | Rust crates exposed to Python via PyO3 + maturin | `dedup` working |
| [tests/](tests/) | pytest suite | BPE tests |
| `data/` | Downloaded and generated data (git-ignored) | Created by the scripts |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and a Rust toolchain (for `rust/dedup`).

```bash
make install
```

`make help` lists every target (`lint`, `format`, `test`, `test-rust`, ...).

## Quick start

```bash
uv run python src/tokenizer/train_bpe.py --n-docs 500 --vocab-size 4096
```

```bash
uv run python src/tokenizer/one_doc.py
```

The first collects 500 clean Common Crawl docs and trains a BPE on them; the second cleans one
fresh doc and tokenizes it with GPT-2 and with that BPE. Open work is tracked in [TODO.md](TODO.md).
