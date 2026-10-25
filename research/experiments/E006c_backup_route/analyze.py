"""Preregistered E006c analysis (PREREG.md)."""
import json, os
import numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from stats import pair_cluster_ci
D = os.path.dirname(os.path.abspath(__file__)); r = json.load(open(os.path.join(D, "result_it.json"))); ps = r["pairs"]
rng = np.random.default_rng(0)


def cboot(cond, n=10000):
    """AUDIT-004 fix: multiplicity-preserving subject bootstrap, stratified by relation."""
    return pair_cluster_ci(ps, lambda p: p["retained"][cond], strata=lambda p: p["relation"], n=n)


conds = list(ps[0]["retained"])
med = {c: round(float(np.median([p["retained"][c] for p in ps])), 4) for c in conds}
key = ["B25_NOISE", "B25_MEAN", "B25_MEANNM", "B25_ZERO", "H0_NOISE", "H0_MEAN", "H1to7_NOISE", "HALL_NOISE", "LATE_ALL_MEAN"]
ci = {c: cboot(c) for c in key}
joint = {c: med[c] for c in conds if c.startswith("JOINT_25_")}
jmin = min(joint.items(), key=lambda x: x[1])
ci[jmin[0]] = cboot(jmin[0])
dif = [max(abs(p["ld_eager_clean"] - p["ld_clean"]), abs(p["ld_eager_corrupt"] - p["ld_corrupt"])) for p in ps]
m0 = [p["mass_h"][0] for p in ps]; ratio = [p["mass_h"][0] / p["uniform"] for p in ps]
a = {"n_pairs": len(ps), "median": med, "cluster_ci": ci, "joint_min": jmin, "joint_all": joint,
     "eager_sdpa_max_abs_ld_diff": max(dif),
     "C1": max(dif) <= 1e-3,
     "C2": med["B25_MEAN"] >= 0.7 and med["B25_MEANNM"] >= 0.7, "C2_withdraw_nonnecessity": min(med["B25_MEAN"], med["B25_MEANNM"]) < 0.6,
     "C3": med["H1to7_NOISE"] >= 0.9 and med["H0_NOISE"] <= 0.6,
     "C4": jmin[1] <= 0.5 and med["LATE_ALL_MEAN"] <= 0.3,
     "C5": float(np.median(m0)) >= 0.5 and np.mean([x >= 3 for x in ratio]) >= 0.8,
     "head0_mass_median": float(np.median(m0)), "head_avg_mass_median": round(float(np.median([np.mean(p["mass_h"]) for p in ps])), 3), "head_mass_medians": [round(float(np.median([p["mass_h"][h] for p in ps])), 3) for h in range(8)],
     "frac_ratio_ge3": float(np.mean([x >= 3 for x in ratio])),
     "by_relation": {k: {c: round(float(np.median([p["retained"][c] for p in ps if p["relation"] == k])), 4) for c in key} for k in ("R1", "R2")}}
json.dump(a, open(os.path.join(D, "analysis.json"), "w"), indent=1, default=lambda o: o.item())
print(json.dumps({k: v for k, v in a.items() if k not in ("joint_all", "median")}, indent=1, default=lambda o: o.item()))
print("joint", joint)
