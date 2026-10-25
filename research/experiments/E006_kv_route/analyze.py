"""Preregistered analysis for E006 (criteria fixed in PREREG.md before data). Writes analysis.json."""
import json, os, statistics as st
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
res = json.load(open(os.path.join(D, "result_" + os.environ.get("VARIANT", "it") + ".json")))
pairs = res["pairs"]


def summ(vals):
    v = np.array(vals, float)
    rng = np.random.default_rng(0)
    boots = np.median(v[rng.integers(0, len(v), (10000, len(v)))], axis=1) if len(v) else []
    return {"n": len(v), "median": round(float(np.median(v)), 4) if len(v) else None,
            "mean": round(float(v.mean()), 4) if len(v) else None,
            "ci95_median": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)] if len(v) else None}


def block(ps):
    if not ps:
        return None
    conds = list(ps[0]["recovery"])
    out = {c: summ([p["recovery"][c] for p in ps]) for c in conds}
    late = [f"ATTN_{L}" for L in range(15, 35)]
    argmax = [max(late, key=lambda c: p["recovery"][c]) for p in ps]
    out["_attn_argmax_counts"] = {c: argmax.count(c) for c in sorted(set(argmax))}
    out["_frac_kv13_minus_kv14_ge_0.5"] = round(np.mean([p["recovery"]["KV13_all"] - p["recovery"]["KV14_all"] >= 0.5 for p in ps]), 3)
    out["_frac_attn25_is_max"] = round(argmax.count("ATTN_25") / len(ps), 3)
    out["_null_max_abs"] = max(abs(p["recovery"]["NULL"]) for p in ps)
    return out


ho = [p for p in pairs if p["split"] == "held_out"]
a = {"n_pairs": len(pairs), "n_held_out": len(ho), "held_out": block(ho),
     "in_sample": block([p for p in pairs if p["split"] == "in_sample"]),
     "R1_held_out": block([p for p in ho if p["relation"] == "R1"]),
     "R2_held_out": block([p for p in ho if p["relation"] == "R2"])}
h = a["held_out"]; med = lambda c: h[c]["median"]
a["predictions"] = {
    "control_null_ok": bool(h["_null_max_abs"] < 1e-4),
    "P1": med("KV13_all") >= 0.8 and med("KV14_all") <= 0.2 and h["_frac_kv13_minus_kv14_ge_0.5"] >= 0.7,
    "P2": med("KVBOTH_all") >= 0.9,
    "P3": med("KV13_D") >= 0.7 and med("KV13_T") <= 0.3,
    "P4": med("ATTN_25") >= 0.5 and h["_frac_attn25_is_max"] >= 0.5,
    "P5": med("NOISE_ATTN25") <= 0.5,
    "underpowered": len(ho) < 12}
P = a["predictions"]
a["H-008"] = "SUPPORTED" if P["P1"] else ("REFUTED" if med("KV14_all") >= med("KV13_all") else "INCONCLUSIVE")
a["H-009"] = ("SUPPORTED" if P["P4"] and P["P5"] else "PARTIAL (sufficient, not necessary)" if P["P4"] else "REFUTED")
# AUDIT-002 fixes: enforce the preregistered failure rules in code (they did not bind in this run)
if not P["control_null_ok"]:
    a["H-008"] = a["H-009"] = "INVALID (NULL control failed)"
elif P["underpowered"]:
    a["H-008"] = a["H-009"] = "INCONCLUSIVE (underpowered)"


def cluster_boot(ps, cond, n=10000):
    """AUDIT-002 robustness (exploratory): resample subjects, keep all held-out pairs whose clean AND corrupt subject were drawn."""
    rng = np.random.default_rng(0); subs = sorted({p["clean"] for p in ps} | {p["corrupt"] for p in ps}); meds = []
    for _ in range(n):
        pick = set(rng.choice(subs, len(subs), replace=True))
        w = [p["recovery"][cond] for p in ps if p["clean"] in pick and p["corrupt"] in pick]
        if w: meds.append(np.median(w))
    return [round(float(np.percentile(meds, 2.5)), 4), round(float(np.percentile(meds, 97.5)), 4)]


a["subject_cluster_bootstrap_ci95_median"] = {c: cluster_boot(ho, c) for c in
    ("KV13_all", "KV14_all", "KVBOTH_all", "KV13_D", "KV13_T", "ATTN_25", "NOISE_ATTN25", "HEAD25_0")}
ins = {frozenset((p["clean"], p["corrupt"])) for p in pairs if p["split"] == "in_sample"}
a["split_check"] = {"held_out_pairs_overlapping_in_sample_pairs": sum(frozenset((p["clean"], p["corrupt"])) in ins for p in ho),
                    "held_out_R1_pairs_sharing_a_subject_with_in_sample": sum(bool({p["clean"], p["corrupt"]} & {s for q in ins for s in q})
                                                                         for p in ho if p["relation"] == "R1")}
json.dump(a, open(os.path.join(D, "analysis_" + os.environ.get("VARIANT", "it") + ".json"), "w"), indent=1, default=lambda o: o.item())
print(json.dumps({k: a[k] for k in ("n_pairs", "n_held_out", "predictions", "H-008", "H-009")}, indent=1, default=lambda o: o.item()))
for c in ("KV13_all", "KV14_all", "KVBOTH_all", "KV13_D", "KV14_D", "KV13_S", "KV13_T", "ATTN_25", "NOISE_ATTN25", "NULL"):
    print(c, h[c])
