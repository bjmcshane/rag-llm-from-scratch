# src/

Python source, one directory per pipeline stage. Data flows top to bottom:

1. [data_fetch/](data_fetch/): pull raw text from each source and clean it (filters + dedup).
2. [tokenizer/](tokenizer/): turn clean text into token ids.
3. [training/](training/): pretrain the LM on packed token shards.
4. [rag/](rag/): embed and index documents, retrieve passages for a query.
5. [inference/](inference/): generate from the trained model, with retrieved context.

[utils/](utils/) holds shared glue (eval, configs, plotting).

There's no package install (`[tool.uv] package = false`): scripts run as files with
`uv run python src/<dir>/<script>.py`. Scripts that import from a sibling directory add it to
`sys.path` themselves (e.g. `src/tokenizer/one_doc.py` imports `common_crawl` from `data_fetch/`).
