# TODO

## Data: Common Crawl quality

- [ ] **Model-based spam / quality classifier.** The rule filters in `src/data_fetch/filters.py` miss
  machine-generated spam that looks like fluent English (e.g. the `actual-proof.de/.../ebook-...`
  pages that pass every heuristic).
  - [ ] Build labeled data: positives from high-quality sources (Wikipedia, arXiv, Stack Exchange,
    Gutenberg), negatives from random Common Crawl docs that pass the current filters.
    Optionally have an LLM score a few thousand CC samples for "educational value" 0–5
    (FineWeb-Edu approach) and use those as labels.
  - [ ] Train a cheap classifier: fastText supervised first (fast, CPU only), then compare against
    logistic regression on embeddings from a small encoder.
  - [ ] Add `classifier_filter(text)` to `filters.py`, run after `quality_filter` and before dedup;
    pick the score threshold by hand-reading samples around the cutoff.
  - [ ] Track kept-rate and drop reasons per WARC in the `stats` Counter.
- [ ] **URL / domain blocklist.** Drop known spam, adult and gambling domains before downloading
  the body (e.g. the UT1 blocklist used by FineWeb), plus a list of domains we find by hand.
