import py7zr, xml.etree.ElementTree as ET

with py7zr.SevenZipFile("datascience.stackexchange.com.7z") as z:
    z.extract(path="se", targets=["Posts.xml"])

for _, row in ET.iterparse("se/Posts.xml"):
    if row.tag == "row":
        body_html, post_type = row.get("Body"), row.get("PostTypeId")  # 1=question, 2=answer
        row.clear()