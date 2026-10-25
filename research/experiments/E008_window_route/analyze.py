"""Preregistered E008 analysis (PREREG.md incl. 'Changes after pre-audit')."""
import json, os
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); r = json.load(open(os.path.join(D, "result_it.json")))
rng = np.random.default_rng(0)
FULL = {19, 24, 29, 34}


def usable(c): return abs(c["ld_clean"] - c["ld_corrupt"]) >= 2


def cboot(cells, cond, key, n=10000):
    units = sorted({key(c) for c in cells}); meds = []
    for _ in range(n):
        pick = list(rng.choice(units, len(units), replace=True))
        w = [c["recovery"][cond] for u in pick for c in cells if key(c) == u]
        if w: meds.append(np.median(w))
    return [round(float(np.percentile(meds, 2.5)), 4), round(float(np.percentile(meds, 97.5)), 4)]


def classify(m13, m14, mb):
    if m13 >= 0.5 and m14 < 0.2: return "L13-only"
    if m14 >= 0.5 and m13 < 0.2: return "L14-only"
    if m13 >= 0.2 and m14 >= 0.2: return "mixed"
    if mb >= 0.5: return "redundant"
    return "neither"


def summarize(cells, key, kv_suffix="D"):
    u = [c for c in cells if usable(c)]
    out = {"n": len(cells), "n_usable": len(u), "acc_clean": float(np.mean([c["clean_correct"] for c in cells])),
           "acc_corrupt": float(np.mean([c["corrupt_correct"] for c in cells])), "denominators": [round(c["ld_clean"] - c["ld_corrupt"], 2) for c in cells]}
    if len(u) < 6:
        out["status"] = "INCONCLUSIVE (<6 usable)"; return out
    conds = list(u[0]["recovery"])
    out["median"] = {k: round(float(np.median([c["recovery"][k] for c in u])), 4) for k in conds}
    att = [f"ATTN_{L}" for L in range(15, 35)]
    am = [max(att, key=lambda k: c["recovery"][k]) for c in u]
    out["attn_argmax_counts"] = {k: am.count(k) for k in sorted(set(am))}
    out["top_attn_by_median"] = max(att, key=lambda k: out["median"][k])
    for k in (f"KV13_{kv_suffix}", f"KV14_{kv_suffix}", f"KVBOTH_{kv_suffix}", "ATTN_25", out["top_attn_by_median"]):
        out.setdefault("ci", {})[k] = cboot(u, k, key)
    out["null_max_abs"] = max(abs(c["recovery"]["NULL"]) for c in u)
    m = out["median"]; out["route"] = classify(m[f"KV13_{kv_suffix}"], m[f"KV14_{kv_suffix}"], m[f"KVBOTH_{kv_suffix}"])
    return out


A = {}
for fl in ("A", "B"):
    for d in sorted({c["d"] for c in r["A"] if c["filler"] == fl}):
        A[f"{fl}{d}"] = summarize([c for c in r["A"] if c["filler"] == fl and c["d"] == d], key=lambda c: c["clean"])
B = {f"d{d}": summarize([c for c in r["B"] if c["d"] == d], key=lambda c: c["words"][["red", "blue", "green"].index(c["queried"])])
     for d in sorted({c["d"] for c in r["B"]})}


def side(cell, k, thr, ge=True):
    if "median" not in cell: return "INCONCLUSIVE"
    m, (lo, hi) = cell["median"][k], cell["ci"].get(k, (None, None))
    ok = m >= thr if ge else m <= thr
    if not ok: return "fail"
    return "supported" if (lo >= thr if ge else hi <= thr) else "weak"


a = {"A": A, "B": B}
a["A1_mechanical"] = {k: (abs(v["median"]["KV13_D"]) <= 0.05 if "median" in v else None) for k, v in A.items() if k.endswith(("525", "700"))}
a["A2"] = {k: {"KV14_D>=0.5": side(v, "KV14_D", 0.5), "top_attn_is_full_layer": (int(v["top_attn_by_median"].split("_")[1]) in FULL) if "median" in v else None}
           for k, v in A.items() if k.endswith("700")}
a["A3"] = {k: {"KV13_D>=0.5": side(v, "KV13_D", 0.5), "ATTN_25_argmax_frac": (v["attn_argmax_counts"].get("ATTN_25", 0) / v["n_usable"]) if "median" in v else None}
           for k, v in A.items() if k.endswith(("60", "400"))}
a["A4_accuracy"] = {k: v["acc_clean"] for k, v in A.items()}
a["B_routes"] = {k: v.get("route") for k, v in B.items()}
a["B_MM_prediction_d60"] = (B.get("d60", {}).get("route") == "L13-only")
a["B_DS_prediction_d60"] = (B.get("d60", {}).get("route") == "L14-only")
a["H011_d700_routes"] = {k: v.get("route") for k, v in A.items() if k.endswith("700")}
json.dump(a, open(os.path.join(D, "analysis.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
for k, v in list(A.items()) + list(B.items()):
    print(k, v.get("n_usable"), "acc", round(v["acc_clean"], 2), v.get("route"), {x: v["median"][x] for x in ("KV13_D", "KV14_D", "KVBOTH_D", "ATTN_25")} if "median" in v else v.get("status"),
          v.get("top_attn_by_median"), v.get("ci"))
print({k: a[k] for k in ("A1_mechanical", "A2", "A3", "B_routes", "B_MM_prediction_d60", "B_DS_prediction_d60")})
