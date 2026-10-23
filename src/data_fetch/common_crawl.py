import gzip, requests
from collections import Counter

import trafilatura
from corpus_dedup import Deduper  # Rust, see rust/dedup
from warcio.archiveiterator import ArchiveIterator

from filters import filter_doc

BASE = "https://data.commoncrawl.org/"
CRAWL = "CC-MAIN-2024-33"

# crawls = requests.get("https://index.commoncrawl.org/collinfo.json").json()
# print([c["id"] for c in crawls[:5]])  # most recent first


def warc_paths(crawl=CRAWL):
    return gzip.decompress(requests.get(f"{BASE}crawl-data/{crawl}/warc.paths.gz").content).decode().split()


def iter_docs(warc_path, min_html_bytes=500):
    """Yield (url, text) for each HTML page in one WARC file (~1 GB, streamed)."""
    with requests.get(BASE + warc_path, stream=True) as r:
        r.raise_for_status()
        for rec in ArchiveIterator(r.raw):
            # Each fetch is stored as request + response + metadata records; only responses hold pages.
            # A WARC also starts with one warcinfo record describing the file.
            if rec.rec_type != "response":
                continue
            h = rec.http_headers
            if h.get_statuscode() != "200" or "text/html" not in (h.get_header("Content-Type") or ""):
                continue
            html = rec.content_stream().read()
            if len(html) < min_html_bytes:  # empty / placeholder pages
                continue
            text = trafilatura.extract(html)  # handles charset detection (utf-8, GB2312, ...)
            if text:
                yield rec.rec_headers.get_header("WARC-Target-URI"), text


def iter_clean_docs(warc_path, deduper, stats):
    """iter_docs + language/quality filters + dedup. `stats` counts kept docs and each drop reason.

    Pass the same Deduper across WARC files to dedup the whole corpus, not just one file.
    """
    for url, text in iter_docs(warc_path):
        reason = filter_doc(text) or ("duplicate" if deduper.is_duplicate(text) else None)
        stats[reason or "kept"] += 1
        if reason is None:
            yield url, text


if __name__ == "__main__":
    stats, deduper = Counter(), Deduper()
    for i, (url, text) in enumerate(iter_clean_docs(warc_paths()[0], deduper, stats)):
        print(f"--- {url} ({len(text)} chars)\n{text[:200]}\n")
        if i == 9:
            break
    print(f"{sum(stats.values())} docs extracted:")
    for reason, n in stats.most_common():
        print(f"  {reason:<22} {n}")
