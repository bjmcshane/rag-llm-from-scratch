import gzip, requests
from warcio.archiveiterator import ArchiveIterator

BASE = "https://data.commoncrawl.org/"
paths = gzip.decompress(requests.get(BASE + "crawl-data/CC-MAIN-2024-33/warc.paths.gz").content).decode().split()


# crawls = requests.get("https://index.commoncrawl.org/collinfo.json").json()
# print([c["id"] for c in crawls[:5]])  # most recent first

with requests.get(BASE + paths[0], stream=True) as r:
    for rec in ArchiveIterator(r.raw):
        if rec.rec_type == "response":
            url = rec.rec_headers.get_header("WARC-Target-URI")
            html = rec.content_stream().read()  # raw HTML bytes → send to trafilatura