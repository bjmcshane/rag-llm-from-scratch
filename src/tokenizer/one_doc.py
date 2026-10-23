"""Clean and tokenize one Common Crawl document, end to end.

Scans WARC records (moving on to the next WARC file if needed) until a doc survives the
filters + dedup, then tokenizes it with GPT-2's BPE (tiktoken) as a reference, and with our
baseline BPE too if one has been trained (src/tokenizer/train_bpe.py). Writes the text and a
uint16 token array per tokenizer to data/one_doc/.

    uv run python src/tokenizer/one_doc.py [--bpe data/tokenizer/bpe_4096.json]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_fetch"))
from bpe import BPETokenizer  # noqa: E402
from common_crawl import first_clean_doc, scanned  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"
OUT = DATA / "one_doc"


def tokenize(name, enc, text):
    ids = enc.encode_ordinary(text) + [enc.eot_token]  # <|endoftext|> separates docs in a shard
    assert enc.decode(ids[:-1]) == text  # byte-level BPE is lossless
    assert enc.n_vocab <= 2**16  # ids fit in uint16
    np.array(ids, dtype=np.uint16).tofile(OUT / f"tokens_{name}.bin")
    print(
        f"{name:<10} {len(ids):>6} tokens ({len(text) / len(ids):.2f} chars/token), first 20:",
        [enc.decode([t]) for t in ids[:20]],
    )
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpe", type=Path, default=DATA / "tokenizer" / "bpe_4096.json")
    ap.add_argument("--max-scanned", type=int, default=100_000)
    args = ap.parse_args()

    stats = Counter()
    url, text = first_clean_doc(stats=stats, max_scanned=args.max_scanned)
    print(f"{url}\n{scanned(stats)} docs scanned: {dict(stats.most_common())}")
    print(f"{len(text)} chars, {len(text.split())} words\n")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "doc.txt").write_text(text)
    meta = {"url": url, "n_chars": len(text), "n_tokens": {}}
    meta["n_tokens"]["gpt2"] = tokenize("gpt2", tiktoken.get_encoding("gpt2"), text)
    if args.bpe.exists():
        meta["n_tokens"][args.bpe.stem] = tokenize(args.bpe.stem, BPETokenizer.load(args.bpe), text)
    else:
        print(f"(no trained BPE at {args.bpe}; run src/tokenizer/train_bpe.py)")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {OUT}/")
