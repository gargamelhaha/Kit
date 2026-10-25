"""AUDIT-002: independent audit of E006 by two non-Anthropic model families (method + code + data, no desired conclusion)."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__)); from llm_pool import Pool
E = "experiments/E006_kv_route/"
res = json.load(open(E + "result_it.json"))
rows = [{"relation": p["relation"], "split": p["split"], "pair": f'{p["clean"]} -> {p["corrupt"]}', "T": p["T"], "D": p["D"], "S": p["S"],
         "ld_clean": round(p["ld_clean"], 2), "ld_corrupt": round(p["ld_corrupt"], 2),
         **{k: round(v, 3) for k, v in p["recovery"].items()}} for p in res["pairs"]]
code = "\n\n".join(f"=== {f} ===\n" + open(f).read() for f in (E + "PREREG.md", E + "run.py", E + "analyze.py", "experiments/lib/runner.py"))
P = f"""You are an independent auditor (different organisation from the experiment's author). Audit the experiment below.
You get the preregistration, the code, the model-architecture facts, and per-pair raw results. Decide for yourself what the
data do and do not show.
Report ONLY JSON: {{"implementation_bugs": [{{"location": str, "problem": str, "severity": "critical|major|minor", "evidence": str}}],
"prereg_deviations": [str], "statistical_issues": [str], "alternative_explanations": [{{"explanation": str, "test_to_rule_out": str}}],
"warranted_conclusions": [str], "overclaims_to_avoid": [str], "recommended_followups": [str], "overall_verdict": str}}

ARCHITECTURE FACTS: Gemma 4 E2B text decoder, 35 layers, d=1536, 8 query heads/1 KV head; sliding window 512, layers 4,9,...,34 are
full attention; layers 15-34 have no K/V projections and read K/V stored by layer 13 (sliding) and layer 14 (full); every layer adds
a per-layer-embedding (PLE) term; post-attention RMSNorm is applied to the attention output before adding it to the residual;
final logit softcap 30. Runner(last_only_from=15) computes layers 15-34 only at the final position (shown exact to 3.6e-5 earlier).

{code}

RAW PER-PAIR RESULTS (recovery values; NOISE_ATTN25 is 'retained' fraction in the clean run):
{json.dumps(rows)}
"""
outs = Pool(tag="AUDIT-002").map([{"model": m, "max_tokens": 60000, "reasoning_effort": "medium", "meta": {"task": "AUDIT-002"},
                                   "messages": [{"role": "user", "content": P}]} for m in ("deepseek", "minimax")])
rep = {}
for m, o in zip(("deepseek", "minimax"), outs):
    mm = re.search(r"\{.*\}", o.get("content", ""), re.S)
    try: rep[m] = json.loads(mm.group(0))
    except Exception: rep[m] = {"raw": o.get("content", ""), "error": o.get("error")}
    rep[m + "_usage"] = o.get("usage")
json.dump(rep, open("audits/AUDIT-002_raw.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({m: (rep[m].get("overall_verdict") if isinstance(rep[m], dict) else None) for m in ("deepseek", "minimax")}, indent=1))
