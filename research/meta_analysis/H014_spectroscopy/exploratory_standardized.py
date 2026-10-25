"""Exploratory (post hoc): same test with columns standardised (removes domination by high-variance columns)."""
import json, os
import numpy as np
E = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "E006_kv_route")
rows, cols = [], None
for v in ("it", "base"):
    for p in json.load(open(os.path.join(E, f"result_{v}.json")))["pairs"]:
        rc = {k: x for k, x in p["recovery"].items() if k != "NULL"}; cols = cols or list(rc); rows.append([rc[c] for c in cols])
M = np.array(rows); Z = (M - M.mean(0)) / (M.std(0) + 1e-12)
def f3(X): s = np.linalg.svd(X, compute_uv=False) ** 2; return float(s[:3].sum() / s.sum()), (s / s.sum())[:6].round(3).tolist()
rng = np.random.default_rng(0)
null = [f3(np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])]))[0] for _ in range(1000)]
_, _, Vt = np.linalg.svd(Z, full_matrices=False)
out = {"top3_std": f3(Z)[0], "spectrum_std": f3(Z)[1], "null_p99": float(np.percentile(null, 99)), "null_median": float(np.median(null)),
       "c1": sorted(zip(cols, Vt[0].round(2).tolist()), key=lambda x: -abs(x[1]))[:10], "c2": sorted(zip(cols, Vt[1].round(2).tolist()), key=lambda x: -abs(x[1]))[:10]}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "exploratory_standardized.json"), "w"), indent=1); print(json.dumps(out, indent=1))
