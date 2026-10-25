#!/usr/bin/env python3
"""Scientific-memory tool: validate registries, merge sources, build SQLite index, stats.

  kb.py validate        schema + referential-integrity checks (exit 1 on error)
  kb.py merge-sources   merge research/evidence/LIT-*_sources.csv -> SOURCE_REGISTRY.csv (dedup by arXiv id / DOI / title)
  kb.py build           rebuild research/state/knowledge.sqlite (derived index; text files are the source of truth)
  kb.py stats           counts by status / read depth
"""
import collections, csv, glob, hashlib, json, os, re, sqlite3, sys
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KJ, HJ = os.path.join(R, "scientific_knowledge.jsonl"), os.path.join(R, "hypotheses.jsonl")
SRC, DB = os.path.join(R, "SOURCE_REGISTRY.csv"), os.path.join(R, "state", "knowledge.sqlite")
STATUSES = {"CANDIDATE", "SOURCE_SUPPORTED", "EXPERIMENTALLY_SUPPORTED", "REPLICATED", "ROBUST", "REFUTED", "INCONCLUSIVE"}
CLAIM_TYPES = {"published_result", "experimental_observation", "meta_analysis", "causal_hypothesis", "architectural_fact"}
REQ = ["id", "statement", "scope", "claim_type", "status", "primary_sources", "supporting_evidence", "contradicting_evidence",
       "conditions", "verification_method", "artifacts", "audit", "uncertainty", "last_verified", "history"]
SRC_COLS = ["source_id", "title", "authors", "year", "date", "venue_or_status", "doi", "arxiv_id", "url", "domain", "subtopic",
            "found_via", "read_depth", "key_claims", "methods", "limitations", "code_url", "relevance", "is_critique_or_replication", "notes"]
SRC_RE = re.compile(r"^LIT\d\d-\d+$")


def jl(p):
    out = []
    if os.path.exists(p):
        for n, line in enumerate(open(p, encoding="utf-8"), 1):
            if line.strip():
                try: out.append(json.loads(line))
                except json.JSONDecodeError as e: sys.exit(f"{p}:{n}: invalid JSON: {e}")
    return out


def sources():
    """Registry rows keyed by id; duplicate ids from evidence/source_dedup_map.csv resolve to the kept row."""
    if not os.path.exists(SRC):
        return {}
    src = {r["source_id"]: r for r in csv.DictReader(open(SRC, encoding="utf-8"))}
    dm = os.path.join(R, "evidence", "source_dedup_map.csv")
    aliases = {}
    if os.path.exists(dm):
        aliases = {r["duplicate_id"]: src[r["kept_id"]] for r in csv.DictReader(open(dm)) if r["kept_id"] in src}
    sources.aliases = aliases
    return src


def resolve(sid, src):
    return src.get(sid) or getattr(sources, "aliases", {}).get(sid)


def validate():
    errs, warns, src, claims = [], [], sources(), jl(KJ)
    ids = collections.Counter(c.get("id") for c in claims)
    errs += [f"duplicate claim id {i}" for i, n in ids.items() if n > 1]
    for c in claims:
        cid = c.get("id", "?")
        errs += [f"{cid}: missing field {k}" for k in REQ if k not in c]
        if c.get("status") not in STATUSES: errs.append(f"{cid}: bad status {c.get('status')}")
        if c.get("status") == "CANDIDATE": errs.append(f"{cid}: CANDIDATE belongs in the hypothesis register")
        if c.get("claim_type") not in CLAIM_TYPES: errs.append(f"{cid}: bad claim_type {c.get('claim_type')}")
        refs = c.get("primary_sources", []) + [e.get("ref") for e in c.get("supporting_evidence", []) + c.get("contradicting_evidence", [])]
        errs += [f"{cid}: unknown source {s}" for s in refs if s and SRC_RE.match(s) and not resolve(s, src)]
        errs += [f"{cid}: missing artifact {a}" for a in c.get("artifacts", []) if not os.path.exists(os.path.join(R, a))]
        errs += [f"{cid}: missing audit file {a}" for a in (c.get("audit") or []) if not os.path.exists(os.path.join(R, a))]
        if c.get("status") in {"EXPERIMENTALLY_SUPPORTED", "REPLICATED", "ROBUST"} and not c.get("artifacts"):
            errs.append(f"{cid}: experimental status without artifacts")
        if c.get("status") in {"REPLICATED", "ROBUST"} and not c.get("audit"):
            errs.append(f"{cid}: {c['status']} requires an audit")
        if c.get("status") == "SOURCE_SUPPORTED" and not c.get("primary_sources"):
            errs.append(f"{cid}: SOURCE_SUPPORTED without primary_sources")
        if not c.get("history"): errs.append(f"{cid}: empty history")
    hyps = jl(HJ)
    errs += [f"duplicate hypothesis id {i}" for i, n in collections.Counter(h.get("id") for h in hyps).items() if n > 1]
    for h in hyps:
        errs += [f"{h.get('id')}: hypothesis missing {k}" for k in ("id", "statement", "prediction", "falsification", "alternatives", "status") if k not in h]
        if h.get("status") not in STATUSES: errs.append(f"{h.get('id')}: bad status")
    for sid, r in src.items():
        if not (r.get("url") or r.get("doi") or r.get("arxiv_id")): warns.append(f"{sid}: no url/doi/arxiv")
        if r.get("read_depth") not in {"full_text", "partial", "abstract_only", "metadata_only"}: warns.append(f"{sid}: bad read_depth")
    print(f"claims={len(claims)} hypotheses={len(hyps)} sources={len(src)} errors={len(errs)} warnings={len(warns)}")
    for e in errs: print("ERROR", e)
    for w in warns[:30]: print("WARN", w)
    return 1 if errs else 0


def merge_sources():
    rows, seen, dups = [], {}, []
    for f in sorted(glob.glob(os.path.join(R, "evidence", "LIT-*_sources.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            ax = re.sub(r"v\d+$", "", (r.get("arxiv_id") or "").strip())
            key = ax or (r.get("doi") or "").lower().strip() or re.sub(r"[^a-z0-9]", "", (r.get("title") or "").lower())[:120]
            if key in seen:
                seen[key]["notes"] = (seen[key].get("notes") or "") + f" | dup:{r['source_id']}"
                dups.append((r["source_id"], seen[key]["source_id"])); continue
            r = {k: r.get(k, "") for k in SRC_COLS}; seen[key] = r; rows.append(r)
    with open(SRC, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SRC_COLS); w.writeheader(); w.writerows(rows)
    with open(os.path.join(R, "evidence", "source_dedup_map.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["duplicate_id", "kept_id"]); w.writerows(dups)
    print(f"merged {len(rows)} unique sources; {len(dups)} duplicates mapped")


def build():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    if os.path.exists(DB): os.remove(DB)
    db = sqlite3.connect(DB)
    db.executescript("""
    CREATE TABLE sources(source_id TEXT PRIMARY KEY, title TEXT, year TEXT, url TEXT, doi TEXT, arxiv_id TEXT, domain TEXT,
                         read_depth TEXT, relevance TEXT, is_critique TEXT);
    CREATE TABLE claims(id TEXT PRIMARY KEY, statement TEXT, scope TEXT, claim_type TEXT, status TEXT, last_verified TEXT, json TEXT, sha256 TEXT);
    CREATE TABLE claim_links(claim_id TEXT, ref TEXT, role TEXT);
    CREATE TABLE claim_history(claim_id TEXT, date TEXT, change TEXT);
    CREATE TABLE hypotheses(id TEXT PRIMARY KEY, statement TEXT, status TEXT, json TEXT);""")
    for r in sources().values():
        db.execute("INSERT INTO sources VALUES(?,?,?,?,?,?,?,?,?,?)", tuple(r[k] for k in ("source_id", "title", "year", "url", "doi", "arxiv_id",
                   "domain", "read_depth", "relevance", "is_critique_or_replication")))
    for c in jl(KJ):
        s = json.dumps(c, sort_keys=True, ensure_ascii=False)
        db.execute("INSERT INTO claims VALUES(?,?,?,?,?,?,?,?)", (c["id"], c["statement"], c["scope"], c["claim_type"], c["status"],
                   c["last_verified"], s, hashlib.sha256(s.encode()).hexdigest()))
        links = [(x, "primary") for x in c["primary_sources"]] + [(e.get("ref"), "support") for e in c["supporting_evidence"]] + \
                [(e.get("ref"), "contradict") for e in c["contradicting_evidence"]] + [(a, "artifact") for a in c["artifacts"]] + \
                [(a, "audit") for a in (c.get("audit") or [])]
        db.executemany("INSERT INTO claim_links VALUES(?,?,?)", [(c["id"], r_, role) for r_, role in links])
        db.executemany("INSERT INTO claim_history VALUES(?,?,?)", [(c["id"], h.get("date"), h.get("change")) for h in c["history"]])
    for h in jl(HJ):
        db.execute("INSERT INTO hypotheses VALUES(?,?,?,?)", (h["id"], h["statement"], h["status"], json.dumps(h, ensure_ascii=False)))
    db.commit(); print("built", DB)


def render():
    """Regenerate SCIENTIFIC_KNOWLEDGE.md from scientific_knowledge.jsonl (the jsonl is authoritative)."""
    src = sources()
    L = ["# Scientific knowledge (verified)", "",
         "Generated by `python research/tools/kb.py render` from `scientific_knowledge.jsonl` — do not edit by hand.",
         "Only curator-reviewed claims with evidence appear here; hypotheses live in HYPOTHESIS_REGISTER.md.",
         "Status semantics: SOURCE_SUPPORTED = a cited source states it; EXPERIMENTALLY_SUPPORTED = our data support it under the",
         "stated conditions; REPLICATED = reproduced in independent runs/implementations; ROBUST = passed preregistered robustness",
         "checks; REFUTED / INCONCLUSIVE as named.", ""]
    for c in jl(KJ):
        L += [f"## {c['id']} — {c['status']} ({c['claim_type']})", "", f"**Claim.** {c['statement']}", "", f"**Scope.** {c['scope']}", "",
              f"**Conditions.** {c['conditions']}", "", f"**Verification.** {c['verification_method']}", ""]
        if c["primary_sources"]:
            L.append("**Primary sources.** " + "; ".join(
                f"{s_} ({(resolve(s_, src) or {}).get('title', '?')[:90]})" for s_ in c["primary_sources"]))
            L.append("")
        for e in c["supporting_evidence"]:
            L.append(f"- supports: `{e.get('ref')}` — {e.get('summary', '')}")
        for e in c["contradicting_evidence"]:
            L.append(f"- contradicts: `{e.get('ref')}` — {e.get('summary', '')}")
        L += ["", f"**Artifacts.** " + ", ".join(f"`{a}`" for a in c["artifacts"]), "",
              f"**Audit.** " + (", ".join(f"`{a}`" for a in c["audit"]) if c["audit"] else "none yet"), "",
              f"**Uncertainty / limits.** {c['uncertainty']}", "", f"**Last verified.** {c['last_verified']}  ",
              "**History.** " + "; ".join(f"{h['date']}: {h['change']}" for h in c["history"]), ""]
    open(os.path.join(R, "SCIENTIFIC_KNOWLEDGE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("rendered", len(jl(KJ)), "claims")


def stats():
    src = sources()
    print("sources by read_depth:", dict(collections.Counter(r["read_depth"] for r in src.values())))
    print("sources by domain:", dict(collections.Counter(r["domain"] for r in src.values())))
    print("claims by status:", dict(collections.Counter(c["status"] for c in jl(KJ))))
    print("hypotheses by status:", dict(collections.Counter(h["status"] for h in jl(HJ))))


if __name__ == "__main__":
    sys.exit({"validate": validate, "build": build, "merge-sources": merge_sources, "stats": stats, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "validate"]() or 0)
