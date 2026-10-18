import os, requests

r = requests.get(
    "https://api.github.com/search/repositories",
    params={"q": "language:python license:mit stars:>500", "per_page": 5},
    headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"},
)
for repo in r.json()["items"]:
    zip_url = f"https://github.com/{repo['full_name']}/archive/refs/heads/{repo['default_branch']}.zip"