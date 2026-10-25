"""Literature scout: arXiv API search -> dedupe vs SOURCE_REGISTRY -> dual-model relevance screening (abstracts only).

Every query is logged to logs/search_<task>.jsonl; candidates + screening to evidence/<task>_candidates.json.
Screening by two model families gives agreement statistics (Cohen's kappa) so screening reliability is measured,
not assumed. Abstract screening is triage only - never evidence for a claim.
Usage: python arxiv_scout.py LIT-06 queries.txt "<screening question>"
"""
import csv, json, os, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(__file__))
from llm_pool import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NS = {"a": "http://www.w3.org/2005/Atom"}


def search(q, n=25):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "start": 0, "max_results": n, "sortBy": "relevance"})
    for attempt in range(4):
        try:
            x = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-lab/0.1"}), timeout=60).read()
            break
        except Exception as e:
            time.sleep(5 * (attempt + 1)); x = None; err = str(e)
    if x is None:
        return None, err
    out = []
    for e in ET.fromstring(x).findall("a:entry", NS):
        aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.rsplit("/abs/", 1)[-1])
        out.append({"arxiv_id": aid, "title": " ".join(e.find("a:title", NS).text.split()),
                    "abstract": " ".join(e.find("a:summary", NS).text.split()),
                    "date": e.find("a:published", NS).text[:10],
                    "authors": "; ".join(a.find("a:name", NS).text for a in e.findall("a:author", NS)[:6])})
    return out, None


def kappa(a, b):
    n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n; pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else None


def main(task, qfile, question):
    known = {r["arxiv_id"] for r in csv.DictReader(open(os.path.join(ROOT, "SOURCE_REGISTRY.csv"))) if r["arxiv_id"]}
    log = open(os.path.join(ROOT, "logs", f"search_{task}.jsonl"), "a")
    cands, seen = [], set()
    for q in [l.strip() for l in open(qfile) if l.strip()]:
        res, err = search(q)
        new = [r for r in (res or []) if r["arxiv_id"] not in known and r["arxiv_id"] not in seen]
        for r in new:
            seen.add(r["arxiv_id"]); r["query"] = q; cands.append(r)
        log.write(json.dumps({"ts": time.strftime("%FT%TZ", time.gmtime()), "source": "arxiv_api", "query": q,
                              "n_results": len(res or []), "n_new": len(new), "error": err}) + "\n"); log.flush()
        print(f"{q!r}: {len(res or [])} results, {len(new)} new", flush=True)
        time.sleep(3)
    reqs = []
    for c in cands:
        for m in ("deepseek", "minimax"):
            reqs.append({"model": m, "max_tokens": 3000, "meta": {"task": task, "arxiv": c["arxiv_id"]},
                         "messages": [{"role": "user", "content":
                             f"Screening question: {question}\n\nTitle: {c['title']}\nAbstract: {c['abstract']}\n\n"
                             'Reply ONLY with JSON {"relevance": 0-5, "include": true|false, "reason": "<=25 words", '
                             '"claim_type": "method|empirical|theory|survey|critique|system"}'}]})
    outs = Pool(tag=task).map(reqs)
    for i, c in enumerate(cands):
        c["screen"] = {}
        for m, o in zip(("deepseek", "minimax"), outs[2 * i:2 * i + 2]):
            mm = re.search(r"\{.*\}", o.get("content", ""), re.S)
            try:
                c["screen"][m] = json.loads(mm.group(0)) if mm else {"error": "no json"}
            except Exception:
                c["screen"][m] = {"error": "bad json"}
    ok = [c for c in cands if all("include" in c["screen"].get(m, {}) for m in ("deepseek", "minimax"))]
    a = [bool(c["screen"]["deepseek"]["include"]) for c in ok]; b = [bool(c["screen"]["minimax"]["include"]) for c in ok]
    stats = {"n_candidates": len(cands), "n_screened_both": len(ok), "include_deepseek": sum(a), "include_minimax": sum(b),
             "include_both": sum(x and y for x, y in zip(a, b)), "kappa": kappa(a, b) if ok else None}
    json.dump({"task": task, "question": question, "stats": stats, "candidates": cands},
              open(os.path.join(ROOT, "evidence", f"{task}_candidates.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
