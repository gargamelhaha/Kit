"""Evaluate E002-E004 strictly against their PREREG thresholds. Writes <exp>/analysis.json and prints a summary.
Bootstrap CIs (10k resamples over items/pairs, seed 0) are reported for medians/means where applicable."""
import json, os, statistics as st, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); rng = np.random.default_rng(0)
def boot(vals, f=np.median, n=10000):
    v = np.asarray(vals, float)
    if len(v) == 0: return None
    bs = [f(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
def save(exp, res):
    json.dump(res, open(os.path.join(H, exp, "analysis.json"), "w"), indent=1); print(exp, json.dumps(res, indent=1)[:3000])

def e002(var):
    p = os.path.join(H, "E002_logit_lens", f"result_{var}.json")
    if not os.path.exists(p): return
    d = json.load(open(p)); items = [i for i in d["items"] if i["correct"]]
    first = []; stay = []; early = []
    for i in items:
        r = i["ranks"]  # index 0..34 = resid_pre of layer k, 35 = final residual
        f = next(k for k, x in enumerate(r) if all(y == 1 for y in r[k:]) or x == 1)
        f1 = next(k for k, x in enumerate(r) if x == 1)
        first.append(f1); stay.append(sum(1 for x in r[f1 + 1:] if x == 1) / max(1, len(r[f1 + 1:])))  # AUDIT-001 #4
        early.append(st.median(r[:6]))
    res = {"variant": var, "n_items": len(d["items"]), "n_correct": len(items),
           "first_top1_layer": {"median": st.median(first), "ci95": boot(first), "values": first},
           "P1_median_first_top1_ge_24": st.median(first) >= 24,
           "median_rank_layers0_5": st.median(early), "P2_median_rank_le5_gt_1000": st.median(early) > 1000,
           "stay_top1_fraction_median": st.median(stay), "P3_stay_ge_0.8": st.median(stay) >= 0.8,
           "by_kind": {k: st.median([f for f, i in zip(first, items) if i["kind"] == k]) for k in ("landmark", "capital")},
           "median_rank_by_layer": [st.median([i["ranks"][k] for i in items]) for k in range(36)],
           "median_prob_by_layer": [round(st.median([i["probs"][k] for i in items]), 5) for k in range(36)]}
    save("E002_logit_lens", res) if var == "it" else json.dump(res, open(os.path.join(H, "E002_logit_lens", f"analysis_{var}.json"), "w"), indent=1)

def e003(var="it"):
    p = os.path.join(H, "E003_induction", f"result_{var}.json")
    if not os.path.exists(p): return
    d = json.load(open(p)); r1, r2 = d["clean_nll"]["repeat1"], d["clean_nll"]["repeat2"]
    sc = np.array(d["scores"]); top = d["top_heads"]; rnd = [x["d_repeat2"] for x in d["random_ablation"]]
    t2, t1 = d["top_ablation"]["d_repeat2"], d["top_ablation"]["d_repeat1"]
    res = {"variant": var, "clean_nll": d["clean_nll"], "P1_ratio": r2 / r1, "P1_pass": r2 < 0.25 * r1,
           "max_induction_score": float(sc.max()), "P2_pass": bool(sc.max() > 0.3), "top_heads": top,
           "n_heads_score_gt_0.3": int((sc > 0.3).sum()), "heads_gt_0.3_by_block": {"layers_0_14": int((sc[:15] > 0.3).sum()), "layers_15_34": int((sc[15:] > 0.3).sum())},
           "top5_d_repeat2": t2, "top5_d_repeat1": t1, "n_random": len(rnd), "random_d_repeat2_median": float(np.median(rnd)) if rnd else None,
           "random_d_repeat2_max": float(max(rnd)) if rnd else None,
           "P3_pass": bool(rnd) and len(rnd) == 30 and t2 > max(rnd) and t2 >= 3 * float(np.median(rnd)),
           "P3_empirical_p": (1 + sum(x >= t2 for x in rnd)) / (1 + len(rnd)) if rnd else None,
           "P4_ratio_d1_over_d2": t1 / t2 if t2 else None, "P4_pass": (t1 / t2 < 0.25) if t2 else None,
           "per_layer_max_score": [round(float(x), 3) for x in sc.max(1)]}
    save("E003_induction", res)

def e004(var="it"):
    p = os.path.join(H, "E004_factual_patching", f"result_{var}.json")
    if not os.path.exists(p): return
    d = json.load(open(p)); P = d["pairs"]
    p1 = [max(pr["grid_early"][L][pr["last_subj_pos"]] for L in range(15)) for pr in P]
    p2 = [max(pr["final_pos_late"]) for pr in P]
    at14 = [pr["grid_early"][14][pr["last_subj_pos"]] for pr in P]
    tmpl = []
    for pr in P:
        T, s = pr["T"], pr["last_subj_pos"]
        tmpl.append([st.median([pr["grid_early"][L][q] for q in range(s + 1, T - 1)]) if T - 1 > s + 1 else 0 for L in range(15)])
    tmpl_med = [st.median([t[L] for t in tmpl]) for L in range(15)]
    ctrl = [c for pr in P if pr["control"] for c in pr["control"]["recovery"]]
    # mean recovery maps
    lsp = [[pr["grid_early"][L][pr["last_subj_pos"]] for L in range(15)] for pr in P]
    fin_early = [[pr["grid_early"][L][pr["T"] - 1] for L in range(15)] for pr in P]
    res = {"variant": var, "n_pairs": len(P), "n_items_correct": sum(i["correct"] for i in d["items"]), "n_items": len(d["items"]),
           "P1_frac_pairs_lastsubj_max_ge_0.5": sum(x >= 0.5 for x in p1) / len(P), "P1_pass": sum(x >= 0.5 for x in p1) / len(P) >= 0.7,
           "P2_frac_pairs_final_late_max_ge_0.5": sum(x >= 0.5 for x in p2) / len(P), "P2_pass": sum(x >= 0.5 for x in p2) / len(P) >= 0.7,
           "P3_median_at14": st.median(at14), "P3_median_max": st.median(p1), "P3_pass": st.median(at14) < st.median(p1),
           "P4_template_median_by_layer": [round(x, 3) for x in tmpl_med], "P4_pass": max(tmpl_med) < 0.2,
           "control_median_abs": float(np.median(np.abs(ctrl))) if ctrl else None, "control_pass": (float(np.median(np.abs(ctrl))) < 0.1) if ctrl else None,
           "mean_recovery_last_subject_by_layer": [round(float(np.mean([r[L] for r in lsp])), 3) for L in range(15)],
           "mean_recovery_final_pos_by_layer_0_14": [round(float(np.mean([r[L] for r in fin_early])), 3) for L in range(15)],
           "mean_recovery_final_pos_by_layer_15_34": [round(float(np.mean([pr["final_pos_late"][k] for pr in P])), 3) for k in range(20)],
           "pairs": [(pr["clean"], pr["corrupt"], round(pr["ld_clean"], 2), round(pr["ld_corrupt"], 2)) for pr in P]}
    save("E004_factual_patching", res)

if __name__ == "__main__":
    e002("it"); e002("base"); e003(); e004()
