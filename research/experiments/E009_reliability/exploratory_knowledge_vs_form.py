"""Exploratory (post hoc; plans the H-018 retest): separate answer-form errors from knowledge errors, recompute TEST AUROCs
for 'knowledge-correct vs knowledge-error' after removing form errors."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")); from stats import auroc
D = os.path.dirname(os.path.abspath(__file__)); V = os.environ.get("VARIANT", "it")
src = open(os.path.join(D, "analyze.py")).read(); ns = {"__file__": os.path.join(D, "analyze.py")}
exec(src.split("# (b) DoLa-style contrast")[0].replace("print(", "(lambda *a, **k: None)("), ns)
FUNC = {"the", "a", "an", "located", "in", "of", "known", "called", "one", "not", "also", "that", "to", "at", "on", "by", "is", "was", "and", "from", "**", "*", "#"}
def form_error(r):
    if r["correct"]: return False
    s = r["top1_str"].strip().lower()
    return any(t == r["target"] for t in r["lens_top1"][26:35]) or (r["target_rank_final"] <= 2 and (s in FUNC or not any(c.isalnum() for c in s)))
test, Stest = ns["test"], ns["Stest"]
keep = [i for i, r in enumerate(test) if not form_error(r)]
y = np.array([test[i]["correct"] for i in keep])
out = {"n_test": len(test), "form_errors": sum(form_error(r) for r in test), "knowledge_errors": int(len(keep) - y.sum()), "n_kept": len(keep),
       "auroc_knowledge_only": {k: auroc([Stest[i][k] for i in keep], y) for k in Stest[0]}}
out["auroc_knowledge_only"]["combined_devfit"] = auroc(np.asarray(ns["comb"])[keep], y)
json.dump(out, open(os.path.join(D, f"exploratory_knowledge_vs_form_{V}.json"), "w"), indent=1); print(json.dumps(out, indent=1))
