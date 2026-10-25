"""Pre-run design audit of a PREREG by two non-Anthropic families (cycle-3 process experiment).
python preaudit.py AUDIT-ID experiments/E008_window_route/PREREG.md [extra context files...]"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__)); from llm_pool import Pool
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
aid, prereg, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
ctx = "\n\n".join(f"=== {p} ===\n" + open(os.path.join(R, p)).read()[:25000] for p in [prereg] + extra)
P = f"""You review an experiment design BEFORE it is run (you are from a different organisation than the designer).
Find design flaws that would make the results uninterpretable or misleading: confounds, missing controls, metric problems,
selection effects, mechanically guaranteed predictions, underpowered cells, ways the predictions could pass for the wrong reason.
Return ONLY JSON: {{"flaws": [{{"issue": str, "severity": "critical|major|minor", "consequence": str, "fix": str}}],
"missing_controls": [str], "trivially_true_predictions": [str], "suggested_additional_predictions": [str], "verdict": str}}

{ctx}"""
outs = Pool(tag=aid).map([{"model": m, "max_tokens": 40000, "reasoning_effort": "medium", "meta": {"task": aid},
                           "messages": [{"role": "user", "content": P}]} for m in ("deepseek", "minimax")])
rep = {"prereg": prereg}
for m, o in zip(("deepseek", "minimax"), outs):
    mm = re.search(r"\{.*\}", o.get("content", ""), re.S)
    try: rep[m] = json.loads(mm.group(0))
    except Exception: rep[m] = {"raw": o.get("content", "")[:20000], "error": o.get("error")}
json.dump(rep, open(os.path.join(R, "audits", f"{aid}_raw.json"), "w"), indent=1, ensure_ascii=False)
for m in ("deepseek", "minimax"):
    for f in (rep[m].get("flaws") or []): print(m, f.get("severity"), "|", f.get("issue", "")[:200], "| FIX:", f.get("fix", "")[:200])
