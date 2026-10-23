import requests

text = requests.get("https://www.gutenberg.org/cache/epub/1342/pg1342.txt").text
start, end = text.find("*** START OF"), text.find("*** END OF")
body = text[text.find("\n", start) + 1 : end]  # strip license header/footer