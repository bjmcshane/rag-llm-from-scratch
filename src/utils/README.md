# src/utils/

Shared Python helpers used across stages. **Empty so far.**

Intended for glue that changes often: evaluation (perplexity, retrieval recall), config
loading (`pydantic` models), and plotting. Anything CPU-heavy over the whole corpus belongs in
[rust/](../../rust/) instead.
