"""Generic independent audit by two non-Anthropic families: python audit_exp.py AUDIT-ID EXP_DIR [EXP_DIR ...] [--claims K-0007,K-0008]
Gives auditors PREREG, code, analysis and raw-result excerpt plus the *current knowledge-base wording* of the claims, and asks
whether the wording is warranted. No desired conclusion is stated."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__)); from llm_pool import Pool
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
args = sys.argv[1:]; claims = []
if "--claims" in args:
    i = args.index("--claims"); claims = args[i + 1].split(","); args = args[:i] + args[i + 2:]
aid, exps = args[0], args[1:]
parts = []
for e in exps:
    d = os.path.join(R, "experiments", e)
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if fn.endswith((".md", ".py")) or fn == "analysis.json":
            parts.append(f"=== {e}/{fn} ===\n" + open(p).read()[:30000])
        elif fn.startswith("result") and fn.endswith(".json"):
            parts.append(f"=== {e}/{fn} (first 12k chars) ===\n" + open(p).read()[:12000])
parts.append("=== experiments/lib/runner.py ===\n" + open(os.path.join(R, "experiments/lib/runner.py")).read())
kb = [json.loads(l) for l in open(os.path.join(R, "scientific_knowledge.jsonl")) if l.strip()]
cl = "\n".join(json.dumps({k: c[k] for k in ("id", "statement", "scope", "uncertainty")}, ensure_ascii=False) for c in kb if c["id"] in claims)
P = f"""You are an independent auditor from a different organisation than the experiment's author. Audit the experiments below
(preregistration, code, analysis output, raw results excerpt) and the knowledge-base claims that cite them.
Architecture: Gemma 4 E2B text decoder, 35 layers; layers 15-34 have no K/V projections and read K/V stored by layer 13 (sliding window 512)
and layer 14 (full); per-layer embeddings added at each layer; post-attention RMSNorm before the residual add; logit softcap 30.
Return ONLY JSON: {{"implementation_bugs": [{{"location": str, "problem": str, "severity": "critical|major|minor", "evidence": str}}],
"prereg_deviations": [str], "statistical_issues": [str], "alternative_explanations": [{{"explanation": str, "test_to_rule_out": str}}],
"claim_wording": [{{"claim": str, "warranted": "yes|partly|no", "required_change": str}}], "recommended_followups": [str], "overall_verdict": str}}

CLAIMS UNDER AUDIT:
{cl}

{chr(10).join(parts)}"""
outs = Pool(tag=aid).map([{"model": m, "max_tokens": 60000, "reasoning_effort": "medium", "meta": {"task": aid},
                           "messages": [{"role": "user", "content": P}]} for m in ("deepseek", "minimax")])
rep = {"prompt_chars": len(P), "experiments": exps, "claims": claims}
for m, o in zip(("deepseek", "minimax"), outs):
    mm = re.search(r"\{.*\}", o.get("content", ""), re.S)
    try: rep[m] = json.loads(mm.group(0))
    except Exception: rep[m] = {"raw": o.get("content", "")[:20000], "error": o.get("error")}
    rep[m + "_usage"] = o.get("usage")
json.dump(rep, open(os.path.join(R, "audits", f"{aid}_raw.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps({m: (rep[m].get("overall_verdict") if isinstance(rep[m], dict) else None) for m in ("deepseek", "minimax")}, indent=1, ensure_ascii=False))
