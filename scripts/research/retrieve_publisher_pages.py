"""Attempt public HTML access to six priority papers; never bypass access controls."""
import concurrent.futures
import json
from html.parser import HTMLParser
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[2] / "docs/research/2026-10-07"
OUT = ROOT / "publisher_pages"


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def fetch(record):
    result = {"key": record["key"], "url": record["url"]}
    try:
        request = urllib.request.Request(record["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8", "replace")
            result["final_url"] = response.url
        parser = TextParser()
        parser.feed(raw)
        text = "\n".join(parser.parts)
        (OUT / (record["key"] + ".txt")).write_text("SOURCE: " + record["url"] + "\n" + text, encoding="utf-8")
        result.update(status="retrieved_requires_content_check", characters=len(text))
    except Exception as error:
        result.update(status="error", error=str(error))
    return result


def main():
    jobs = json.loads((OUT / "retrieval_log.json").read_text(encoding="utf-8"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, jobs))
    (OUT / "retrieval_retry_log.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
