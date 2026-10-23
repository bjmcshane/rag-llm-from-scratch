# tests/

pytest suite (`make test`, or `uv run pytest`). Tests need no network or downloaded data.

| File | Covers |
|---|---|
| `test_bpe.py` | `src/tokenizer/bpe.py`: merge rules, vocab layout, round-trips (incl. non-ASCII and emoji), special tokens, save/load, and that incremental training picks a top-count pair at every step, as recounting from scratch would |

Rust unit tests live next to the code (`rust/dedup/src/lib.rs`) and run with `make test-rust`.
