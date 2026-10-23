import gzip, requests
from collections import Counter
from itertools import islice

import trafilatura
from corpus_dedup import Deduper  # Rust, see rust/dedup
from requests.adapters import HTTPAdapter
from urllib3.exceptions import ProtocolError
from urllib3.util import Retry
from warcio.archiveiterator import ArchiveIterator
from warcio.exceptions import ArchiveLoadFailed

from filters import filter_doc

BASE = "https://data.commoncrawl.org/"
CRAWL = "CC-MAIN-2024-33"

# data.commoncrawl.org answers 503 when overloaded; back off (2, 4, 8, ... s) before giving up.
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=6, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504), allowed_methods=["GET"]
)))

# crawls = requests.get("https://index.commoncrawl.org/collinfo.json").json()
# print([c["id"] for c in crawls[:5]])  # most recent first


def warc_paths(crawl=CRAWL):
    return gzip.decompress(session.get(f"{BASE}crawl-data/{crawl}/warc.paths.gz").content).decode().split()


def iter_docs(warc_path, min_html_bytes=500):
    """Yield (url, text) for each HTML page in one WARC file (~1 GB, streamed)."""
    with session.get(BASE + warc_path, stream=True) as r:
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
            try:
                text = trafilatura.extract(html)  # handles charset detection (utf-8, GB2312, ...)
            except Exception:  # malformed HTML can crash lxml; skip the page, not the WARC
                continue
            if text:
                yield rec.rec_headers.get_header("WARC-Target-URI"), text


def scanned(stats):
    """Docs extracted so far, kept or dropped (WARC-level errors don't count)."""
    return stats.total() - stats["warc_error"]


def iter_clean_docs(warc_path, deduper, stats, max_scanned=None):
    """iter_docs + language/quality filters + dedup. `stats` counts kept docs and each drop reason.

    Pass the same Deduper across WARC files to dedup the whole corpus, not just one file.
    Stops early once `scanned(stats)` reaches `max_scanned`.
    """
    for url, text in iter_docs(warc_path):
        reason = filter_doc(text) or ("duplicate" if deduper.is_duplicate(text) else None)
        stats[reason or "kept"] += 1
        if reason is None:
            yield url, text
        if max_scanned and scanned(stats) >= max_scanned:
            return


def iter_corpus(warcs, deduper, stats, max_scanned=None):
    """iter_clean_docs over many WARC files: when one runs out, keep going with the next.

    A WARC that fails to download or parse is counted in `stats["warc_error"]` and skipped.
    `max_scanned` caps extracted docs (kept + dropped) across all files, so a strict filter
    can't end up scanning the whole crawl looking for a match.
    """
    for warc in warcs:
        try:
            yield from iter_clean_docs(warc, deduper, stats, max_scanned)
        # OSError covers requests' errors and bad gzip; ProtocolError is a dropped stream mid-file
        except (OSError, EOFError, ProtocolError, ArchiveLoadFailed) as e:
            stats["warc_error"] += 1
            print(f"skipping {warc}: {e}")
        if max_scanned and scanned(stats) >= max_scanned:
            return


def first_clean_doc(warcs=None, deduper=None, stats=None, max_scanned=100_000):
    """Scan WARC records (and then further WARC files) until one doc passes every filter.

    Returns (url, text). Raises LookupError if nothing passes within `max_scanned` docs.
    """
    warcs = warc_paths() if warcs is None else warcs
    stats = Counter() if stats is None else stats
    found = next(iter_corpus(warcs, deduper or Deduper(), stats, max_scanned), None)
    if found is None:
        raise LookupError(f"no doc passed the filters in {scanned(stats)} scanned: {dict(stats)}")
    return found


if __name__ == "__main__":
    stats, deduper = Counter(), Deduper()
    for url, text in islice(iter_corpus(warc_paths(), deduper, stats), 10):
        print(f"--- {url} ({len(text)} chars)\n{text[:200]}\n")
    print(f"{scanned(stats)} docs extracted:")
    for reason, n in stats.most_common():
        print(f"  {reason:<22} {n}")
