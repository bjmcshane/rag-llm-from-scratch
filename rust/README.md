# rust/

Rust crates for CPU-bound passes over the whole corpus, exposed to Python with
[PyO3](https://pyo3.rs) and built by [maturin](https://www.maturin.rs).

| Crate | Python module | What it does |
|---|---|---|
| [dedup/](dedup/) | `corpus_dedup` | Exact + MinHash-LSH near-duplicate detection |

Each crate is its own Python package, listed as a path dependency in the root `pyproject.toml`
(`[tool.uv.sources]`). `uv sync` / `make install` builds it into `.venv`, and rebuilds when its
`.rs` files change. `make test-rust` runs the Rust unit tests.

Next likely crate: a Rust port of the BPE tokenizer in [src/tokenizer/](../src/tokenizer/),
tested against the Python version.
