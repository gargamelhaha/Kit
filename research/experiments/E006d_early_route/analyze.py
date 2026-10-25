import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib")); from stats import pair_cluster_ci
D = os.path.dirname(os.path.abspath(__file__)); ps = json.load(open(os.path.join(D, "result_it.json")))["pairs"]
conds = list(ps[0]["retained"])
med = {c: round(float(np.median([p["retained"][c] for p in ps])), 4) for c in conds}
ci = {c: pair_cluster_ci(ps, lambda p, c=c: p["retained"][c], strata=lambda p: p["relation"]) for c in conds}
a = {"n": len(ps), "median": med, "ci": ci, "P1": med["ALL_ATTN_MEAN"] <= 0.15, "P2": med["EARLY_ATTN_MEAN"] >= 0.7, "P3": med["LATE_MLP_MEAN"] >= 0.5,
     "by_relation": {r: {c: round(float(np.median([p["retained"][c] for p in ps if p["relation"] == r])), 3) for c in conds} for r in ("R1", "R2")}}
json.dump(a, open(os.path.join(D, "analysis.json"), "w"), indent=1, default=lambda o: o.item()); print(json.dumps(a, indent=1, default=lambda o: o.item()))
