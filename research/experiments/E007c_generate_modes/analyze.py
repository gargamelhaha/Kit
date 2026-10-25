"""E007c analysis (PREREG.md + changes after pre-audit)."""
import json, os, statistics as st
D = os.path.dirname(os.path.abspath(__file__)); r = json.load(open(os.path.join(D, "result_it.json")))
rows = r["rows"]
by = {}
for x in rows:
    by.setdefault((x["cond"], x["item"]), {})[x["arm"]] = x
med = lambda v: st.median(v) if v else None
tab, P1 = [], True
for (c, i), arms in by.items():
    ref, sk, bf = arms.get("REF"), arms.get("SKIP"), arms.get("SKIP+BF16")
    if not (ref and sk): continue
    exact = all(v["identical"] for v in sk["vs_ref"]); dmax = max(v["max_dscore_before_div"] for v in sk["vs_ref"])
    P1 &= exact and dmax <= 1e-3
    row = {"cond": c, "item": i, "T": ref["T"], "batch": ref["batch"], "G": ref["G"],
           "skip_identical": exact, "skip_max_dlogit": dmax, "cache_maxdiff": sk.get("cache_maxdiff_vs_ref"),
           "prefill_speedup": round(med(ref["prefill_s"]) / med(sk["prefill_s"]), 2),
           "total_speedup_skip": round(med(ref["wall_s"]) / med(sk["wall_s"]), 2),
           "decode_s_ref": med(ref["decode_s_per_step"]), "decode_s_skip": med(sk["decode_s_per_step"])}
    if bf:
        row.update({"decode_speedup_bf16_vs_skip": round(med(sk["decode_s_per_step"]) / med(bf["decode_s_per_step"]), 2),
                    "total_speedup_bf16_vs_ref": round(med(ref["wall_s"]) / med(bf["wall_s"]), 2),
                    "bf16_frac_tokens_same_vs_skip": round(st.mean(v["frac_identical_tokens_all"] for v in bf["vs_skip"]), 3),
                    "bf16_identical_rows_vs_skip": f'{sum(v["identical"] for v in bf["vs_skip"])}/{len(bf["vs_skip"])}',
                    "bf16_first_div": [v["first_div"] for v in bf["vs_skip"]], "bf16_max_dlogit_before_div": max(v["max_dscore_before_div"] for v in bf["vs_skip"])})
    if "ref_cached_vs_full_reprefill" in ref:
        row["ref_cached_vs_full_reprefill_identical"] = all(v["identical"] for v in ref["ref_cached_vs_full_reprefill"])
    tab.append(row)
g = lambda c, k: [t[k] for t in tab if t["cond"] == c]
a = {"table": tab, "P1_exact_all": P1,
     "P2": all(v >= 2 for v in g("C2", "prefill_speedup") + g("C3", "prefill_speedup")) and all(v >= 1.5 for v in g("C1", "prefill_speedup")),
     "P3_C2_total_ge_1.3": all(v >= 1.3 for v in g("C2", "total_speedup_skip")), "P3_C1_total_descriptive": g("C1", "total_speedup_skip"),
     "P4_decode_bf16_ge_1.3": all(t.get("decode_speedup_bf16_vs_skip", 0) >= 1.3 for t in tab),
     "P4_quality_mean_frac_same": round(st.mean(t["bf16_frac_tokens_same_vs_skip"] for t in tab if "bf16_frac_tokens_same_vs_skip" in t), 3)}
json.dump(a, open(os.path.join(D, "analysis.json"), "w"), indent=1)
for t in tab: print(t)
print({k: v for k, v in a.items() if k != "table"})
