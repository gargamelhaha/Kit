"""Preregistered E007 criteria (PREREG.md). T=2048 FULL arm missing (RSS guard), so predictions are evaluated on T<=1024."""
import json, os, statistics as st
D = os.path.dirname(os.path.abspath(__file__)); r = json.load(open(os.path.join(D, "result_it.json")))
rows = r["rows"]
for x in rows:  # sensitivity of the speedup to the summary statistic (timing noise on a shared host)
    x["speedup_min_over_min"] = round(min(x["full_s"]) / min(x["skip_s"]), 3)
    x["speedup_mean"] = round(st.mean(x["full_s"]) / st.mean(x["skip_s"]), 3)
sp = {x["T"]: x["speedup_median"] for x in rows}
a = {"rows": rows, "missing": "T=2048: FULL arm exceeded the 6.8 GB RSS guard (rc=3); see result_b_it.json (exploratory)",
     "P1_exact_all_T": all(x["max_abs_dlogit"] <= 1e-3 and x["top1_same"] for x in rows),
     "P1_includes_T_gt_512": any(x["T"] > 512 for x in rows),
     "P2_speedup_ge_2_for_T_ge_256": all(v >= 2.0 for t, v in sp.items() if t >= 256),
     "P3_speedup_le_3.30": all(v <= 3.30 for v in sp.values()),
     "P4_speedup_1024_ge_64": sp.get(1024, 0) >= sp[64],
     "cost_model_note": "Post-hoc: the preregistered cost model wrongly assumed attention-score FLOPs equal in both arms; SKIP also "
                        "drops the T x T attention of layers 15-34, so 3.30 is not an upper bound for large T. P3 therefore is not "
                        "a valid test of the cost model and is reported as preregistered only."}
json.dump(a, open(os.path.join(D, "analysis.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in a.items() if k != "rows"}, indent=1))
for x in rows: print(x["T"], x["speedup_median"], x["speedup_min_over_min"], x["speedup_mean"], x["max_abs_dlogit"])
