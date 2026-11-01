"""Retrieve public Crossref metadata for a reproducible literature screening pass."""
import concurrent.futures
import datetime
import json
import pathlib
import re
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "research" / "2026-10-07"
QUERIES = [
    "UAV fault diagnosis domain adaptation generalization",
    "UAV fault diagnosis few shot prototype learning",
    "UAV fault detection incipient early detection",
    "UAV fault diagnosis open set unknown faults",
    "UAV fault diagnosis flight maneuver disturbance robustness",
    "UAV fault detection real time online uncertainty",
    "RflyMAD fault diagnosis",
    "无人机 故障诊断 跨域 小样本",
    "UAV fault diagnosis",
    "quadrotor fault detection",
    "multirotor fault diagnosis",
    "unmanned aerial vehicle fault diagnosis",
]


def retrieve(job):
    kind, key, url = job
    record = {"kind": kind, "key": key, "url": url}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "UAVLiteratureScreening/1.0"})
        with urllib.request.urlopen(req, timeout=35) as response:
            record["message"] = json.load(response)["message"]
    except Exception as error:
        record["error"] = str(error)
    return record


def main():
    source = (ROOT / "docs/uav_fault_diagnosis_literature_2025_2026.md").read_text(encoding="utf-8")
    dois = list(dict.fromkeys(re.findall(r"https://doi.org/([^\s]+)", source)))
    jobs = [("doi", doi, "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/")) for doi in dois]
    for query in QUERIES:
        short_query = query in QUERIES[-4:]
        params = urllib.parse.urlencode({"query.title" if short_query else "query.bibliographic": query, "rows": 80 if short_query else 25,
            "filter": "from-pub-date:2024-01-01,until-pub-date:2026-10-07", "sort": "relevance"})
        jobs.append(("search", query, "https://api.crossref.org/works?" + params))
    OUT.mkdir(parents=True, exist_ok=True)
    cached = {}
    if (OUT / "crossref_records.json").exists():
        previous = json.loads((OUT / "crossref_records.json").read_text(encoding="utf-8"))
        cached = {(r["kind"], r["key"]): r for r in previous["records"] if "message" in r}
    records = list(cached.values())
    jobs = [job for job in jobs if (job[0], job[1]) not in cached]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        for result in pool.map(retrieve, jobs):
            records.append(result)
            print(result["kind"], result["key"], "ERROR: " + result["error"] if "error" in result else "OK", flush=True)
    (OUT / "crossref_records.json").write_text(json.dumps({
        "retrieved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": "Metadata and deposited abstracts only; not full-text verification or exhaustive search.",
        "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
