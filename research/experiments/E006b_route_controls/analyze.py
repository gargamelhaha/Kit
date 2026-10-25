"""Preregistered E006b criteria (PREREG.md)."""
import json, os
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); r = json.load(open(os.path.join(D, "result_it.json"))); ps = r["pairs"]
def summ(v):
    v = np.array(v, float); b = np.median(v[np.random.default_rng(0).integers(0, len(v), (10000, len(v)))], axis=1)
    return {"n": len(v), "median": round(float(np.median(v)), 4), "ci95": [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)],
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
L = range(15, 35)
ret = {k: summ([p["retained"][k] for p in ps]) for k in ps[0]["retained"]}
mass = {k: summ([p["mass"][k] for p in ps]) for k in ps[0]["mass"]}
uni = summ([p["uniform_mass"] for p in ps])
others = {l: ret[f"NOISE_ATTN_{l}"]["median"] for l in L if l != 25}
heads = {h: ret[f"NOISE_HEAD25_{h}"]["median"] for h in range(8)}
a = {"n_pairs": len(ps), "retained": ret, "mass": mass, "uniform_mass": uni,
     "Q1": mass["MASS25_0"]["median"] >= 0.3 and mass["MASS25_0"]["median"] >= 3 * uni["median"],
     "Q2": ret["NOISE_ATTN_25"]["median"] <= 0.5 and min(others.values()) >= 0.8,
     "Q2_min_other_layer": min(others.items(), key=lambda x: x[1]),
     "Q3": ret["MEAN_ATTN25"]["median"] <= 0.6,
     "Q4_lowest_head": min(heads.items(), key=lambda x: x[1]), "Q4": min(heads, key=heads.get) == 0,
     "underpowered": len(ps) < 12}
json.dump(a, open(os.path.join(D, "analysis.json"), "w"), indent=1, default=lambda o: o.item())
print(json.dumps({k: a[k] for k in ("n_pairs", "Q1", "Q2", "Q2_min_other_layer", "Q3", "Q4", "Q4_lowest_head")}, default=lambda o: o.item()))
print("NOISE_ATTN25", ret["NOISE_ATTN_25"], "MEAN_ATTN25", ret["MEAN_ATTN25"])
print("heads", heads); print("mass25", {h: mass[f"MASS25_{h}"]["median"] for h in range(8)}, "uniform", uni["median"])
print("mass_L", {l: mass[f"MASS_{l}"]["median"] for l in L}); print("noise_L", others)
