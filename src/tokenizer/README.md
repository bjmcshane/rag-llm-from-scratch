# src/tokenizer/

Baseline byte-level BPE tokenizer in plain Python, plus scripts to train it on Common Crawl and
compare it with GPT-2 (`tiktoken`). It's the readable reference implementation; per
[RUST_PYTHON.md](../../RUST_PYTHON.md) the fast version will be a Rust port tested against it.

| File | What it does |
|---|---|
| `bpe.py` | `BPETokenizer`: train, encode, decode, save/load |
| `train_bpe.py` | Collect a sample of clean CC docs, train a vocab, compare with GPT-2 on held-out docs |
| `one_doc.py` | Find one clean CC doc, tokenize it with GPT-2 and our BPE, write ids to disk |

## How the BPE works

1. **Pre-tokenize** with GPT-2's regex (`GPT2_PATTERN`): contractions, letter runs, digit runs,
   punctuation runs and whitespace, each word keeping its leading space (`" world"`). Merges
   never cross these boundaries, so you don't get tokens like `"g. The"`.
2. **Bytes as the base vocabulary.** Ids 0–255 are raw UTF-8 bytes, so any string can be encoded
   (no `<unk>`) and `decode(encode(s)) == s` exactly.
3. **Train**: count every adjacent id pair (weighted by how often each chunk occurs), merge the
   most frequent pair into a new id, and repeat until `vocab_size` is reached.
4. **Encode**: split the same way, then for each chunk repeatedly apply the earliest-learned
   merge that's present. Encoded chunks are cached, since web text repeats the same words a lot.
5. **Special tokens** (`<|endoftext|>`) get the ids after the last merge. `encode()` recognizes
   them in text; `encode_ordinary()` treats them as plain text, like tiktoken.

Ids are `256 + rank` for merges, so a 4096 vocab is 256 bytes + 3839 merges + 1 special token,
and anything up to 65,536 fits in `uint16` token files.

### Training performance

Training stores each distinct chunk once with its count, and updates pair counts incrementally:
a merge only revisits the chunks that contain the merged pair (tracked in `where`). Picking
the best pair is still a linear `max` over all pairs per merge. That's fine for a few MB and a
few thousand merges; a heap and a Rust port are the next steps when the corpus grows.

## Usage

```python
from bpe import BPETokenizer

tok = BPETokenizer.train(texts, vocab_size=4096)  # texts: any iterable of str
ids = tok.encode("Hello world<|endoftext|>")
assert tok.decode(ids) == "Hello world<|endoftext|>"
tok.save("data/tokenizer/bpe_4096.json")
tok = BPETokenizer.load("data/tokenizer/bpe_4096.json")
```

Saved format (JSON): `pattern`, `merges` as `[[a, b], ...]` in rank order, and `special_tokens`.

## Scripts

```bash
uv run python src/tokenizer/train_bpe.py --n-docs 500 --vocab-size 4096
```

This caches the sample in `data/cc_sample/docs.jsonl` (later runs with a different vocab size
reuse it), trains on the first 90% of docs, writes `data/tokenizer/bpe_<V>.json`, and prints
chars/token on the held-out 10% for our BPE and GPT-2.

```bash
uv run python src/tokenizer/one_doc.py --bpe data/tokenizer/bpe_4096.json
```

This writes `data/one_doc/doc.txt`, `tokens_gpt2.bin`, `tokens_bpe_4096.bin` (`uint16`,
ending in `<|endoftext|>`) and `meta.json`. Read the ids back with
`np.fromfile(path, dtype=np.uint16)`.

Tests: `tests/test_bpe.py` (`make test`).
