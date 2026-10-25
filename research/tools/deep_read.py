"""Machine deep-reading of priority sources with automatic quote verification (Evidence Extractor role).

For each source id: download the arXiv PDF, extract text (pdftotext), ask two different-family API models to extract
protocol / metrics / results / limitations as JSON where *every* item carries a verbatim quote, then verify each quote
against the source text (exact after normalisation, or >=80% of its word 5-grams present = "near").
Outputs: research/evidence/deep_read/<id>.json (+ raw text cached in research/papers/txt/, gitignored).
Usage: python deep_read.py LIT02-002 LIT01-091 ...   (or --priority for the LITERATURE_MAP reading-debt list)
"""
import csv, json, os, re, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(__file__))
from llm_pool import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TXT, OUT = os.path.join(ROOT, "papers", "txt"), os.path.join(ROOT, "evidence", "deep_read")
PRIORITY = ("LIT02-002 LIT01-091 LIT02-065 LIT02-057 LIT02-047 LIT02-088 LIT01-012 LIT02-128 LIT02-127 LIT03-098 "
            "LIT02-017 LIT02-018 LIT01-087 LIT05-013 LIT05-046 LIT04-143 LIT01-040 LIT03-089 LIT02-008 LIT01-090 "
            "LIT02-013 LIT05-080 LIT05-022 LIT05-112 LIT04-016").split()
MAXCH = 110_000
PROMPT = """You are the Evidence Extractor of a research lab. Read the paper text below (it may be truncated) and return ONLY a JSON object:
{"title": str, "research_question": str,
 "models_and_data": [{"item": str, "quote": str}],
 "protocol": [{"step": str, "quote": str}],
 "metrics": [{"name": str, "definition": str, "quote": str}],
 "key_results": [{"claim": str, "numbers": str, "quote": str}],
 "limitations_stated_by_authors": [{"text": str, "quote": str}],
 "threats_not_stated_by_authors": [str],
 "code_or_data_url": str,
 "relevance_to_lab": str}
Rules: use ONLY the provided text; every "quote" must be copied character-for-character from the text (one sentence or
fragment, max 40 words) - never paraphrase inside "quote"; put numbers exactly as reported; 3-8 items per list; if something
is absent write "not stated". "threats_not_stated_by_authors" is your own critical assessment (no quotes needed).
Lab context: we study Gemma 4 E2B internals (KV-shared layers 15-34, per-layer embeddings) with activation patching,
logit lens, induction-head and factual-recall tracing on CPU.

PAPER TEXT:
"""


def norm(s):
    s = s.replace("-\n", "").replace("\u00ad", "")
    s = re.sub(r"[\u2018\u2019]", "'", re.sub(r"[\u201c\u201d]", '"', s))
    return re.sub(r"\s+", " ", s).strip().lower()


def verify(quote, src_norm, src_grams):
    q = norm(quote)
    if len(q) < 12 or q in ("not stated", "n/a"):
        return "skip"
    if q in src_norm:
        return "exact"
    w = q.split()
    grams = [" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))]
    hit = sum(g in src_grams for g in grams) / len(grams)
    return "near" if hit >= 0.8 else "unverified"


def fetch_text(arxiv_id):
    os.makedirs(TXT, exist_ok=True)
    p = os.path.join(TXT, arxiv_id.replace("/", "_") + ".txt")
    if not os.path.exists(p):
        pdf = p[:-4] + ".pdf"
        req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "research-lab/0.1"})
        open(pdf, "wb").write(urllib.request.urlopen(req, timeout=120).read()); time.sleep(3)  # arXiv politeness
        subprocess.run(["pdftotext", pdf, p], check=True)  # reading order; -layout interleaves 2-column lines (PL-8)
        subprocess.run(["pdftotext", "-layout", pdf, p[:-4] + ".layout.txt"], check=True); os.remove(pdf)
    return open(p, errors="ignore").read()


def reverify():
    """Re-check stored quotes against reading-order + layout text (no API calls)."""
    summary = []
    for fn in sorted(os.listdir(OUT)):
        if not fn.startswith("LIT"):
            continue
        rec = json.load(open(os.path.join(OUT, fn))); base = os.path.join(TXT, rec["arxiv_id"].replace("/", "_"))
        srcs = [norm(open(base + ext, errors="ignore").read()) for ext in (".txt", ".layout.txt") if os.path.exists(base + ext)]
        grams = set()
        for s_ in srcs:
            w = s_.split(); grams |= {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}
        joined = "\n".join(srcs)
        for m, e in rec["extractions"].items():
            stats = {"exact": 0, "near": 0, "unverified": 0, "skip": 0}
            for k, v in (e["data"] or {}).items():
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict) and "quote" in it:
                            it["quote_check"] = verify(str(it["quote"]), joined, grams); stats[it["quote_check"]] += 1
            n = stats["exact"] + stats["near"] + stats["unverified"]
            e["quote_stats"], e["verified_frac"] = stats, (round((stats["exact"] + stats["near"]) / n, 3) if n else None)
            summary.append({"source_id": rec["source_id"], "model": m, "parsed": e["data"] is not None, **stats})
        json.dump(rec, open(os.path.join(OUT, fn), "w"), indent=1, ensure_ascii=False)
        print(rec["source_id"], {m: e["verified_frac"] for m, e in rec["extractions"].items()}, flush=True)
    json.dump(summary, open(os.path.join(OUT, "_summary.json"), "w"), indent=1)


def parse_json(s):
    s = s.strip()
    m = re.search(r"\{.*\}", s, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def main(ids):
    reg = {r["source_id"]: r for r in csv.DictReader(open(os.path.join(ROOT, "SOURCE_REGISTRY.csv")))}
    os.makedirs(OUT, exist_ok=True)
    jobs, texts = [], {}
    for sid in ids:
        aid = reg[sid]["arxiv_id"]
        try:
            texts[sid] = fetch_text(aid)
        except Exception as e:
            print(sid, "fetch failed", e, flush=True); continue
        for m in ("deepseek", "minimax"):
            jobs.append((sid, m, {"model": m, "max_tokens": 32000, "meta": {"task": "deep_read", "source": sid},
                                  "messages": [{"role": "user", "content": PROMPT + texts[sid][:MAXCH]}]}))
    print("jobs", len(jobs), flush=True)
    outs = Pool(tag="deep_read").map([j[2] for j in jobs])
    per = {}
    for (sid, m, _), o in zip(jobs, outs):
        per.setdefault(sid, {})[m] = o
    summary = []
    for sid, ms in per.items():
        src = norm(texts[sid]); grams = set()
        w = src.split(); grams = {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}
        rec = {"source_id": sid, "arxiv_id": reg[sid]["arxiv_id"], "title": reg[sid]["title"], "chars": len(texts[sid]),
               "truncated": len(texts[sid]) > MAXCH, "extractions": {}}
        for m, o in ms.items():
            d = parse_json(o.get("content", "")) if not o.get("error") else None
            stats = {"exact": 0, "near": 0, "unverified": 0, "skip": 0}
            if d:
                for k, v in d.items():
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict) and "quote" in it:
                                it["quote_check"] = verify(str(it["quote"]), src, grams); stats[it["quote_check"]] += 1
            n = stats["exact"] + stats["near"] + stats["unverified"]
            rec["extractions"][m] = {"model_id": o.get("model"), "parsed": d is not None, "error": o.get("error"),
                                     "quote_stats": stats, "verified_frac": round((stats["exact"] + stats["near"]) / n, 3) if n else None,
                                     "data": d, "usage": o.get("usage")}
            summary.append({"source_id": sid, "model": m, "parsed": d is not None, **stats})
        json.dump(rec, open(os.path.join(OUT, sid + ".json"), "w"), indent=1, ensure_ascii=False)
        print(sid, {m: rec["extractions"][m]["verified_frac"] for m in rec["extractions"]}, flush=True)
    json.dump(summary, open(os.path.join(OUT, "_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:] == ["--reverify"]:
        reverify()
    else:
        main(PRIORITY if sys.argv[1:] == ["--priority"] else sys.argv[1:])
