import bz2, requests, xml.etree.ElementTree as ET
import mwparserfromhell

url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
with requests.get(url, stream=True) as r:
    for _, el in ET.iterparse(bz2.open(r.raw)):
        if el.tag.endswith("}page"):
            title = el.findtext("{*}title")
            text = mwparserfromhell.parse(el.findtext(".//{*}text") or "").strip_code()
            el.clear()  # free memory as you go