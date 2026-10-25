"""AUDIT-004 follow-up: relation-clustered bootstrap of the TEST AUROC differences (20 relations as clusters)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")); from stats import auroc
D = os.path.dirname(os.path.abspath(__file__)); V = os.environ.get("VARIANT", "it")
src = open(os.path.join(D, "analyze.py")).read()
ns = {"__file__": os.path.join(D, "analyze.py")}; exec(src.split("# (b) DoLa-style contrast")[0], ns)  # reuse the exact preregistered pipeline
test, Stest, comb = ns["test"], ns["Stest"], ns["comb"]
y = np.array([r["correct"] for r in test]); rel = np.array([r["relation"] for r in test]); R = sorted(set(rel))
s0 = np.array([s["S0_maxprob"] for s in Stest]); s2 = np.array([s["S2_lens_agree"] for s in Stest]); cb = np.array(comb)
rng = np.random.default_rng(0); dc, d2 = [], []
for _ in range(5000):
    idx = np.concatenate([np.where(rel == r)[0] for r in rng.choice(R, len(R), replace=True)])
    a0 = auroc(s0[idx], y[idx])
    if a0 is None: continue
    dc.append(auroc(cb[idx], y[idx]) - a0); d2.append(auroc(s2[idx], y[idx]) - a0)
out = {"n_relations": len(R), "ci_combined_minus_S0": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
       "ci_S2_minus_S0": [float(np.percentile(d2, 2.5)), float(np.percentile(d2, 97.5))]}
json.dump(out, open(os.path.join(D, f"robust_relation_bootstrap_{V}.json"), "w"), indent=1); print(out)
