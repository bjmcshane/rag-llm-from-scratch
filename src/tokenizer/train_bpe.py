"""Train the baseline BPE on a sample of clean Common Crawl docs and compare it with GPT-2.

    uv run python src/tokenizer/train_bpe.py --n-docs 500 --vocab-size 4096

The sample is cached in data/cc_sample/docs.jsonl, so re-training with a different vocab size
doesn't re-download anything. The last 10% of docs are held out for the comparison.
"""

import argparse
import json
import sys
import time
from collections import Counter
from itertools import islice
from pathlib import Path

import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_fetch"))
from bpe import BPETokenizer  # noqa: E402
from common_crawl import iter_corpus, scanned, warc_paths  # noqa: E402
from corpus_dedup import Deduper  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
SAMPLE = DATA / "cc_sample" / "docs.jsonl"


def load_sample(n_docs, path=SAMPLE):
    """First `n_docs` clean docs from the crawl, cached as JSONL ({"url", "text"} per line)."""
    if path.exists():
        docs = [json.loads(line) for line in path.read_text().splitlines()]
        if len(docs) >= n_docs:
            return docs[:n_docs]
        print(f"cached sample has {len(docs)} docs, need {n_docs}; rebuilding")
    stats, docs = Counter(), []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for url, text in islice(iter_corpus(warc_paths(), Deduper(), stats), n_docs):
            docs.append({"url": url, "text": text})
            f.write(json.dumps(docs[-1]) + "\n")
            if len(docs) % 50 == 0:
                print(f"  {len(docs)}/{n_docs} kept, {scanned(stats)} scanned")
    print(f"sample: {len(docs)} docs kept of {scanned(stats)} scanned: {dict(stats.most_common())}")
    return docs


def chars_per_token(encode, texts):
    return sum(map(len, texts)) / sum(len(encode(t)) for t in texts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=500)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=None, help="default: data/tokenizer/bpe_<V>.json")
    args = ap.parse_args()
    out = args.out or DATA / "tokenizer" / f"bpe_{args.vocab_size}.json"

    texts = [d["text"] for d in load_sample(args.n_docs)]
    split = max(1, len(texts) // 10)
    train, held_out = texts[:-split], texts[-split:]
    print(f"training on {len(train)} docs ({sum(map(len, train)) / 1e6:.2f}M chars)")

    t0 = time.perf_counter()
    tok = BPETokenizer.train(train, args.vocab_size, verbose=True)
    print(f"trained {len(tok.merges)} merges in {time.perf_counter() - t0:.1f}s")
    tok.save(out)
    print(f"saved {out}")

    assert all(tok.decode(tok.encode(t)) == t for t in held_out), "round-trip failed"
    gpt2 = tiktoken.get_encoding("gpt2")
    print(f"\nheld-out ({len(held_out)} docs), chars/token, higher = better compression:")
    print(f"  ours  (V={tok.n_vocab:>6}) {chars_per_token(tok.encode_ordinary, held_out):.2f}")
    print(f"  gpt2  (V={gpt2.n_vocab:>6}) {chars_per_token(gpt2.encode_ordinary, held_out):.2f}")
    longest = sorted(tok.vocab.values(), key=len)[-10:]
    print("longest learned tokens:", [b.decode("utf-8", "replace") for b in longest])
