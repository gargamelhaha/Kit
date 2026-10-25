"""Novelty audit: do given full-text sources already report a claim? Two families, quotes machine-verified against the texts.
python novelty_audit.py K-0006,K-0007 LIT03-019 LIT06-005 ..."""
import csv, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__)); from llm_pool import Pool; from deep_read import norm, verify
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
claims, sids = sys.argv[1].split(","), sys.argv[2:]
reg = {r["source_id"]: r for r in csv.DictReader(open(os.path.join(R, "SOURCE_REGISTRY.csv")))}
kb = {c["id"]: c for c in (json.loads(l) for l in open(os.path.join(R, "scientific_knowledge.jsonl")) if l.strip())}
ct = "\n".join(f"{c}: {kb[c]['statement']}" for c in claims)
reqs, meta = [], []
for sid in sids:
    aid = reg[sid]["arxiv_id"]; txt = open(os.path.join(R, "papers", "txt", aid + ".txt"), errors="ignore").read()[:110000]
    for m in ("deepseek", "minimax"):
        reqs.append({"model": m, "max_tokens": 30000, "reasoning_effort": "medium", "meta": {"task": "novelty", "src": sid},
                     "messages": [{"role": "user", "content": f"""Does the paper below already report (fully or partly) any of these claims, or a direct analogue in another model?
CLAIMS:
{ct}
Return ONLY JSON: {{"per_claim": [{{"claim": "K-....", "overlap": "same|analogous|related|none", "explanation": str, "quotes": [str]}}]}}
Quotes must be copied verbatim from the paper (<=40 words each); give none if overlap is none.

PAPER ({sid}, {reg[sid]['title']}):
{txt}"""}]}); meta.append((sid, m, txt))
outs = Pool(tag="novelty").map(reqs)
rep = []
for (sid, m, txt), o in zip(meta, outs):
    mm = re.search(r"\{.*\}", o.get("content", ""), re.S)
    try: d = json.loads(mm.group(0))
    except Exception: d = {"error": o.get("error") or "parse"}
    src = norm(txt); w = src.split(); grams = {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}
    for pc in d.get("per_claim", []):
        pc["quote_checks"] = [verify(q, src, grams) for q in pc.get("quotes", [])]
    rep.append({"source": sid, "model": m, **d})
    print(sid, m, [(pc.get("claim"), pc.get("overlap"), pc.get("quote_checks")) for pc in d.get("per_claim", [])], flush=True)
json.dump(rep, open(os.path.join(R, "audits", "NOVELTY-001_raw.json"), "w"), indent=1, ensure_ascii=False)
