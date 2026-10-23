"""Baseline byte-level BPE tokenizer (GPT-2 style), in plain Python.

Same recipe as GPT-2 / tiktoken, minus the speed:
  1. Pre-tokenize: a regex splits text into chunks (words with their leading space, numbers,
     punctuation runs, whitespace), so merges never cross a word boundary.
  2. Start from the 256 byte values, so any UTF-8 input is encodable and decoding is lossless.
  3. Training repeatedly merges the most frequent adjacent pair of ids into a new id.
  4. Encoding replays those merges on each chunk, lowest-rank (earliest learned) first.

This is the Python reference the Rust tokenizer will be tested against (see RUST_PYTHON.md).
"""

import json
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import regex

# GPT-2's split pattern: contractions, letter runs, number runs, other symbols, whitespace.
# `\s+(?!\S)` keeps the last space attached to the next word (" world", not " " + "world").
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"


def merge(ids, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, merges=None, special_tokens=None, pattern=GPT2_PATTERN):
        self.pattern = pattern
        self._split = regex.compile(pattern)
        self.merges = dict(merges or {})  # (id, id) -> new id; insertion order = merge rank
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self.special_tokens = dict(special_tokens or {})  # str -> id, placed after the merges
        self._special_split = (
            regex.compile("(" + "|".join(regex.escape(s) for s in self.special_tokens) + ")")
            if self.special_tokens
            else None
        )
        self._cache = {}

    @property
    def n_vocab(self):
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_token(self):
        return self.special_tokens[EOT]

    # --- training -------------------------------------------------------------------------

    @classmethod
    def train(cls, texts, vocab_size, special_tokens=(EOT,), pattern=GPT2_PATTERN, verbose=False):
        """Learn `vocab_size - 256 - len(special_tokens)` merges from an iterable of documents.

        Each distinct chunk is stored once with its count, and pair counts are updated
        incrementally: a merge only touches the chunks that contain the merged pair.
        """
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {256 + len(special_tokens)}")
        split = regex.compile(pattern)

        chunk_counts = Counter()
        for text in texts:
            chunk_counts.update(split.findall(text))
        words = [list(chunk.encode("utf-8")) for chunk in chunk_counts]
        freqs = list(chunk_counts.values())

        pair_counts = Counter()
        where = defaultdict(set)  # pair -> indices of words containing it
        for i, (w, f) in enumerate(zip(words, freqs, strict=True)):
            for pair in pairwise(w):
                pair_counts[pair] += f
                where[pair].add(i)

        merges, token_bytes = {}, {i: bytes([i]) for i in range(256)}
        for k in range(n_merges):
            if not pair_counts:
                break  # every chunk is a single token; nothing left to merge
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + k
            merges[best] = new_id
            token_bytes[new_id] = token_bytes[best[0]] + token_bytes[best[1]]
            for i in where.pop(best):
                w, f = words[i], freqs[i]
                for pair in pairwise(w):  # retract the old word's pairs...
                    pair_counts[pair] -= f
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                w = words[i] = merge(w, best, new_id)
                for pair in pairwise(w):  # ...and count the merged word's pairs
                    pair_counts[pair] += f
                    where[pair].add(i)
            pair_counts.pop(best, None)
            if verbose and (k % 500 == 0 or k == n_merges - 1):
                print(f"merge {k + 1}/{n_merges}: {best} -> {new_id} {token_bytes[new_id]!r}")

        base = 256 + len(merges)
        specials = {s: base + i for i, s in enumerate(special_tokens)}
        return cls(merges, specials, pattern)

    # --- encoding / decoding --------------------------------------------------------------

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge present; that's the order training created them.
            pair = min(pairwise(ids), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        """Encode text, treating special-token strings as plain text."""
        return [i for chunk in self._split.findall(text) for i in self._encode_chunk(chunk)]

    def encode(self, text):
        """Encode text, mapping registered special-token strings to their ids."""
        if self._special_split is None:
            return self.encode_ordinary(text)
        ids = []
        for part in self._special_split.split(text):
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode_bytes(self, ids):
        inverse = {i: s.encode("utf-8") for s, i in self.special_tokens.items()}
        return b"".join(self.vocab[i] if i in self.vocab else inverse[i] for i in ids)

    def decode(self, ids):
        # A token can end mid-way through a multi-byte character; "replace" only matters when
        # decoding a slice, the full sequence always round-trips.
        return self.decode_bytes(ids).decode("utf-8", errors="replace")

    # --- persistence ----------------------------------------------------------------------

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        merges = [[a, b] for (a, b) in self.merges]  # rank = position, id = 256 + rank
        d = {"pattern": self.pattern, "merges": merges, "special_tokens": self.special_tokens}
        Path(path).write_text(json.dumps(d))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        merges = {(a, b): 256 + rank for rank, (a, b) in enumerate(d["merges"])}
        return cls(merges, d["special_tokens"], d["pattern"])
