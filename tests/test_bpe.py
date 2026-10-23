import sys
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest
import regex

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tokenizer"))
from bpe import EOT, GPT2_PATTERN, BPETokenizer, merge  # noqa: E402

CORPUS = [
    "the cat sat on the mat. the cat ate the rat.",
    "Hello world! Hello, hello world... 12345 and 678.",
    "naïve café — 東京 🙂 emoji and accents survive byte-level BPE",
]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=300)


def test_merge():
    assert merge([1, 2, 3, 1, 2], (1, 2), 9) == [9, 3, 9]
    assert merge([1, 1, 1], (1, 1), 9) == [9, 1]  # non-overlapping, left to right


def test_first_merge_is_most_frequent_pair():
    tok = BPETokenizer.train(["aaab aaab aaab cd"], vocab_size=257, special_tokens=())
    assert list(tok.merges) == [(ord("a"), ord("a"))]


def test_vocab_layout(tok):
    assert tok.n_vocab == 300
    assert len(tok.merges) == 300 - 256 - 1
    assert tok.special_tokens == {EOT: 299}
    assert sorted(tok.merges.values()) == list(range(256, 299))


@pytest.mark.parametrize("text", CORPUS + ["", " ", "unseen words: zebra quokka\n\n\ttabs"])
def test_round_trip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_merges_compress(tok):
    text = CORPUS[0]
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_special_tokens(tok):
    ids = tok.encode(f"one{EOT}two")
    assert ids.count(tok.eot_token) == 1
    assert tok.eot_token not in tok.encode_ordinary(f"one{EOT}two")
    assert tok.decode(ids) == f"one{EOT}two"


def test_save_load(tok, tmp_path):
    path = tmp_path / "bpe.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.special_tokens == tok.special_tokens
    for text in CORPUS:
        assert loaded.encode(text) == tok.encode(text)


def test_matches_naive_training():
    """Incremental pair counting must pick a top pair at every step, as recounting from scratch.

    Ties can break either way, so check the learned merge's count equals the maximum.
    """
    tok = BPETokenizer.train(CORPUS, vocab_size=256 + 60, special_tokens=())
    words = [list(c.encode()) for t in CORPUS for c in regex.findall(GPT2_PATTERN, t)]
    for pair, new_id in tok.merges.items():
        counts = Counter(p for w in words for p in pairwise(w))
        assert counts[pair] == max(counts.values())
        words = [merge(w, pair, new_id) for w in words]
