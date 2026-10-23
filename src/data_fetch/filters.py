"""Document filters for web text: language ID + Gopher/C4-style quality heuristics.

Each filter returns None if the document passes, or a short reason string if it's rejected,
so you can count *why* documents get dropped. Deduplication lives in Rust (rust/dedup).
"""

import re
from collections import Counter

from fast_langdetect import detect

# --- language ---------------------------------------------------------------

LANG = "en"
MIN_LANG_SCORE = 0.65


def language_filter(text, lang=LANG, min_score=MIN_LANG_SCORE):
    """fastText lid.176 language ID. Keeps docs confidently in `lang`."""
    pred = detect(text.replace("\n", " "), k=1)[0]
    if pred["lang"] != lang:
        return f"lang:{pred['lang']}"
    if pred["score"] < min_score:
        return "lang:low_confidence"
    return None


# --- quality (thresholds from Gopher, Rae et al. 2021, and C4, Raffel et al. 2020) ---------------

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLETS = ("•", "-", "*", "·", "‣", "–")
TERMINAL_PUNCT = (".", "!", "?", '"', "'", "”", "’")
BOILERPLATE = ("lorem ipsum", "javascript", "cookie", "terms of use", "privacy policy")

WORD_RE = re.compile(r"\w+", re.UNICODE)


def quality_filter(text):
    """Cheap heuristics that catch spam, boilerplate, keyword stuffing and non-prose."""
    words = text.split()
    n = len(words)
    if n < 50:
        return "too_short"
    if n > 100_000:
        return "too_long"

    mean_len = sum(len(w) for w in words) / n
    if not 3 <= mean_len <= 10:
        return "mean_word_length"

    if (text.count("#") + text.count("...") + text.count("…")) / n > 0.1:
        return "symbol_ratio"

    if sum(any(c.isalpha() for c in w) for w in words) / n < 0.8:
        return "non_alpha_words"

    lowered = {w.lower().strip(".,!?;:\"'()") for w in words}
    if len(STOP_WORDS & lowered) < 2:
        return "no_stop_words"  # real English prose always uses these

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if sum(ln.startswith(BULLETS) for ln in lines) / len(lines) > 0.9:
        return "bullet_lines"
    if sum(ln.endswith(("...", "…")) for ln in lines) / len(lines) > 0.3:
        return "ellipsis_lines"
    if sum(ln.endswith(TERMINAL_PUNCT) for ln in lines) / len(lines) < 0.5:
        return "few_sentences"  # menus, link lists, product grids
    if "{" in text:
        return "code_or_template"  # C4 rule; keep code sources (GitHub) away from this filter
    low = text.lower()
    if sum(low.count(b) for b in BOILERPLATE) / len(lines) > 0.2:
        return "boilerplate"

    return repetition_filter(lines, words)


def repetition_filter(lines, words):
    """Gopher repetition rules: SEO pages repeat the same lines and phrases over and over."""
    if (len(lines) - len(set(lines))) / len(lines) > 0.3:
        return "duplicate_lines"

    toks = [w.lower() for w in WORD_RE.findall(" ".join(words))]
    total_chars = sum(len(t) for t in toks) or 1
    for n, max_frac in ((2, 0.20), (3, 0.18), (4, 0.16)):
        grams = Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
        if not grams:
            break
        gram, count = grams.most_common(1)[0]
        if count > 1 and count * sum(len(t) for t in gram) / total_chars > max_frac:
            return f"top_{n}gram"
    return None


def filter_doc(text):
    """Language first: the quality rules assume space-separated English words."""
    return language_filter(text) or quality_filter(text)
