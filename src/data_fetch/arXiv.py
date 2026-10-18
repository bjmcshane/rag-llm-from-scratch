import feedparser, requests, time

feed = feedparser.parse("http://export.arxiv.org/api/query?search_query=cat:cs.CL&max_results=5")
for e in feed.entries:
    arxiv_id = e.id.split("/abs/")[-1]
    src = requests.get(f"https://arxiv.org/e-print/{arxiv_id}").content  # tar.gz of LaTeX source
    time.sleep(3)  # arXiv asks for ~1 request per 3 seconds