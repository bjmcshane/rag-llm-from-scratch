# src/data_fetch/

Get raw text from each corpus source and clean it.

| File | Source | Status |
|---|---|---|
| `common_crawl.py` | Common Crawl WARC files (web pages) | Working: extract, filter, dedup |
| `filters.py` | Language ID + quality heuristics for web text | Working |
| `wiki.py` | Simple English Wikipedia dump (wikitext → plain text) | Sketch |
| `arXiv.py` | arXiv API, LaTeX source tarballs (cs.CL) | Sketch |
| `stack_exchange.py` | Stack Exchange data dump (`Posts.xml` from a `.7z`) | Sketch |
| `proj_gutenberg.py` | Project Gutenberg plain-text books, license stripped | Sketch |
| `github.py` | GitHub search API for permissively licensed repos (needs `GITHUB_TOKEN`) | Sketch |

## Common Crawl pipeline

```
warc_paths()           list the ~90k WARC files (~1 GB each) in a crawl (CRAWL = CC-MAIN-2024-33)
  └ iter_docs()        stream one WARC: HTTP 200 text/html responses → trafilatura → (url, text)
    └ iter_clean_docs()  + filters.filter_doc() + Deduper.is_duplicate(), counting drop reasons
      └ iter_corpus()    chain many WARCs: when one runs out or fails, move on to the next
        └ first_clean_doc()  the first doc that passes everything, or LookupError
```

- **Keep scanning until something passes.** `iter_corpus` / `first_clean_doc` move record by
  record and then file by file until a doc gets through the filters. `max_scanned` caps how many
  extracted docs (kept + dropped, across all files) to look at before giving up, so an overly strict
  filter fails loudly instead of streaming the whole crawl.
- **Failures.** A page that crashes the HTML extractor is skipped. A WARC that can't be
  downloaded or is cut off mid-stream is counted as `stats["warc_error"]` and skipped.
  `data.commoncrawl.org` often answers 503 under load, so requests are retried with exponential
  backoff (2, 4, 8, ... s, up to 6 tries) before a file counts as failed.
- **Stats.** Every function takes a `Counter` and adds `kept` or the drop reason for each doc
  (`lang:zh`, `too_short`, `few_sentences`, `duplicate`, ...). `scanned(stats)` is the number of
  docs seen, not counting WARC errors.
- **Dedup across files.** Pass one `Deduper` (from [rust/dedup](../../rust/dedup/)) through the
  whole run so duplicates are caught corpus-wide, not just within one WARC.

Expect a low keep rate: the first WARC of CC-MAIN-2024-33 is mostly Chinese and Japanese pages,
and roughly 1 in 15–25 extracted docs survives the English + quality filters.

```bash
uv run python src/data_fetch/common_crawl.py
```

This prints the first 10 clean docs and a breakdown of drop reasons.

## Filters (`filters.py`)

Each filter returns `None` to keep a doc, or a short reason string to drop it.
`filter_doc(text)` runs them in order:

1. **Language** (`language_filter`): fastText lid.176 via `fast-langdetect`. Keeps English
   with score ≥ 0.65.
2. **Quality** (`quality_filter`): Gopher (Rae et al. 2021) and C4 (Raffel et al. 2020)
   heuristics: word count, mean word length, symbol ratio, alphabetic words, stop words, bullet
   and ellipsis lines, sentence-ending punctuation, `{` (code/templates), boilerplate phrases.
3. **Repetition** (`repetition_filter`): duplicate lines and over-represented top 2/3/4-grams.

Still open (see [TODO.md](../../TODO.md)): a model-based spam/quality classifier and a
URL/domain blocklist.
