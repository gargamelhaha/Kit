"""Preregistered E009 analysis (PREREG.md): error detection AUROCs, DoLa-style contrast, early exit. DEV = even index, TEST = odd."""
import json, os, math
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); V = os.environ.get("VARIANT", "it")
recs = [json.loads(l) for l in open(os.path.join(D, f"result_{V}.jsonl"))]
order = {json.loads(l)["id"]: n for n, l in enumerate(open(os.path.join(D, "..", "..", "benchmarks", "factset_v1.jsonl")))}
for r in recs:
    r["split"] = "dev" if order[r["id"]] % 2 == 0 else "test"
dev, test = [r for r in recs if r["split"] == "dev"], [r for r in recs if r["split"] == "test"]
rng = np.random.default_rng(0)


def auroc(score, y):
    s, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg): return None
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1  # ties are rare for continuous signals
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def signals(r):
    tops, fin = r["lens_top1"], r["top1"]
    agree = np.mean([t == fin for t in tops[26:36]])
    stay = next((k for k in range(36) if all(t == fin for t in tops[k:])), 35)
    return {"S0_maxprob": r["final_maxprob"], "S1_margin": r["margin"], "S2_lens_agree": agree, "S3_neg_first_stable": -stay,
            "S4_lens28_maxprob": r["lens_maxprob"][28], "S5_neg_attn25h0_entropy": -(r["attn25h0_entropy"] or 0.0)}


def boot_diff(fa, fb, y, n=5000):
    y = np.asarray(y); fa, fb = np.asarray(fa), np.asarray(fb); out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a, b = auroc(fa[i], y[i]), auroc(fb[i], y[i])
        if a is not None and b is not None: out.append(a - b)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


Sdev = [signals(r) for r in dev]; Stest = [signals(r) for r in test]
ydev = [r["correct"] for r in dev]; ytest = [r["correct"] for r in test]
names = list(Sdev[0])
res = {"n": len(recs), "n_dev": len(dev), "n_test": len(test), "acc_dev": float(np.mean(ydev)), "acc_test": float(np.mean(ytest)),
       "test_errors": int(len(ytest) - sum(ytest)), "auroc_test": {k: auroc([s[k] for s in Stest], ytest) for k in names}}
# combined detector: logistic regression fit on DEV (standardised, plain gradient descent, L2)
X = np.array([[s[k] for k in names] for s in Sdev], float); mu, sd = X.mean(0), X.std(0) + 1e-9
Xt = (np.array([[s[k] for k in names] for s in Stest], float) - mu) / sd; X = (X - mu) / sd
w, b, y = np.zeros(len(names)), 0.0, np.asarray(ydev, float)
for _ in range(5000):
    p = 1 / (1 + np.exp(-(X @ w + b))); g = X.T @ (p - y) / len(y) + 1e-2 * w; w -= 0.5 * g; b -= 0.5 * float((p - y).mean())
comb = Xt @ w + b
res["combined_weights"] = dict(zip(names, w.round(3).tolist()))
res["auroc_test"]["combined"] = auroc(comb, ytest)
res["ci_combined_minus_S0"] = boot_diff(comb, [s["S0_maxprob"] for s in Stest], ytest)
res["ci_S2_minus_S0"] = boot_diff([s["S2_lens_agree"] for s in Stest], [s["S0_maxprob"] for s in Stest], ytest)
A0 = res["auroc_test"]["S0_maxprob"]
res["P_a1"] = res["auroc_test"]["combined"] - A0 >= 0.03 and res["ci_combined_minus_S0"][0] > 0
res["P_a2"] = res["auroc_test"]["S2_lens_agree"] >= A0 - 0.02
res["P_a3"] = A0 >= 0.70
res["underpowered_a"] = res["test_errors"] < 60


# (b) DoLa-style contrast
def contrast_pred(r, bucket):
    ks = [k for k in range(bucket[0], bucket[1] + 1)]
    k = max(ks, key=lambda k: r["jsd"][k])
    lp = np.array(r["plausible_lp"])  # [36, P] log-probs of plausible tokens
    sc = lp[35] - lp[k]
    return r["plausible"][int(sc.argmax())]


buckets = {"16-20": (16, 20), "21-25": (21, 25), "26-30": (26, 30)}
dev_acc = {n: float(np.mean([contrast_pred(r, bk) == r["target"] for r in dev])) for n, bk in buckets.items()}
best = max(dev_acc, key=dev_acc.get)
g_ok = np.array([r["correct"] for r in test]); c_ok = np.array([contrast_pred(r, buckets[best]) == r["target"] for r in test])
b01, b10 = int((g_ok & ~c_ok).sum()), int((~g_ok & c_ok).sum())
n_ = b01 + b10
p_mc = min(1.0, 2 * sum(math.comb(n_, i) for i in range(0, min(b01, b10) + 1)) / 2 ** n_) if n_ else 1.0
res["dola"] = {"dev_acc_by_bucket": dev_acc, "dev_acc_greedy": res["acc_dev"], "chosen_bucket": best,
               "test_acc_greedy": float(g_ok.mean()), "test_acc_contrast": float(c_ok.mean()), "fixed_by_contrast": b10,
               "broken_by_contrast": b01, "mcnemar_exact_p": p_mc}
res["P_b1"] = (c_ok.mean() - g_ok.mean()) >= 0.02 and p_mc < 0.05 and b10 > b01
res["P_b2"] = b01 / len(test) <= 0.03


# (c) early exit
def exit_eval(rs, tau):
    acc, skipped = [], []
    for r in rs:
        k = next((k for k in range(20, 35) if r["lens_maxprob"][k] >= tau), 35)
        acc.append(r["lens_top1"][k] == r["target"]); skipped.append((35 - k) / 35)
    return float(np.mean(acc)), float(np.mean(skipped))


taus = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
dev_curve = {t: exit_eval(dev, t) for t in taus}
ok_taus = [t for t in taus if res["acc_dev"] - dev_curve[t][0] <= 0.01]
tau = min(ok_taus) if ok_taus else None
res["early_exit"] = {"dev_curve": {str(t): v for t, v in dev_curve.items()}, "chosen_tau": tau,
                     "test": dict(zip(("acc", "frac_layers_skipped"), exit_eval(test, tau))) if tau else None,
                     "test_curve": {str(t): exit_eval(test, t) for t in taus}}
if tau:
    res["P_c1"] = res["acc_test"] - res["early_exit"]["test"]["acc"] <= 0.02 and res["early_exit"]["test"]["frac_layers_skipped"] >= 0.15
res["acc_by_difficulty_test"] = {d: float(np.mean([r["correct"] for r in test if r["difficulty"] == d])) for d in ("easy", "medium", "hard")}
json.dump(res, open(os.path.join(D, f"analysis_{V}.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(json.dumps({k: v for k, v in res.items() if k not in ("early_exit",)}, indent=1, default=str)[:3000])
print("early exit", res["early_exit"]["chosen_tau"], res["early_exit"]["test"])
