# rust/dedup (`corpus_dedup`)

Streaming corpus deduplication: exact duplicates by normalized-text hash, near-duplicates by
MinHash + LSH. Used by [src/data_fetch/common_crawl.py](../../src/data_fetch/common_crawl.py).

```python
from corpus_dedup import Deduper

d = Deduper()                       # bands=16, rows=8, threshold=0.8, ngram=5, seed=42
d.is_duplicate(text)                # False the first time; remembers the doc
d.filter_batch([t1, t2, ...])       # list[bool]; signatures computed in parallel, GIL released
len(d), d.exact_dupes, d.near_dupes
```

Docs are checked in order, so the first copy is kept and later copies are flagged.

## How it works

1. **Normalize**: lowercase and collapse whitespace, so formatting-only changes still match.
2. **Exact**: an xxh3 hash of the normalized text, kept in a set.
3. **MinHash**: split the doc into word 5-gram shingles. For each of `bands × rows` (128)
   hash functions `(a·x + b) mod 2^61−1`, keep the minimum over all shingles. Two docs match on
   any one slot with probability equal to their Jaccard similarity.
4. **LSH**: cut the signature into 16 bands of 8 rows. Docs that share any whole band become
   candidates, and a candidate counts as a duplicate only if the fraction of equal slots is at
   least `threshold`. With the defaults, pairs with Jaccard ≥ ~0.7 are very likely to collide.

Memory is about `4 × 128` bytes of signature per unique doc, plus the band buckets.

## Build and test

```bash
make install
```

```bash
make test-rust
```

`make install` (`uv sync`) builds the extension into `.venv` with maturin. `make test-rust` runs
`cargo test`. `pyo3/extension-module` is enabled only through maturin, so `cargo test` can link
against libpython.
