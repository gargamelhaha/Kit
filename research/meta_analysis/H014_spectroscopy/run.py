import json, os
import numpy as np
E = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "E006_kv_route")
rows, cols, src = [], None, []
for v in ("it", "base"):
    for p in json.load(open(os.path.join(E, f"result_{v}.json")))["pairs"]:
        rc = {k: x for k, x in p["recovery"].items() if k != "NULL"}
        cols = cols or list(rc); rows.append([rc[c] for c in cols]); src.append(v)
M = np.array(rows); src = np.array(src)
def ev(X, k=3):
    X = X - X.mean(0); s = np.linalg.svd(X, compute_uv=False) ** 2; return s[:k].sum() / s.sum(), s / s.sum()
f3, spec = ev(M)
rng = np.random.default_rng(0)
null = [ev(np.column_stack([rng.permutation(M[:, j]) for j in range(M.shape[1])]))[0] for _ in range(1000)]
Xc = M - M.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
g1 = [i for i, c in enumerate(cols) if c.startswith("KV13") or c in ("ATTN_25", "HEAD25_0", "KVBOTH_all")]
g2 = [i for i, c in enumerate(cols) if i not in g1]
ratio = [float(np.abs(Vt[k, g1]).mean() / np.abs(Vt[k, g2]).mean()) for k in range(2)]
# transfer: IT-fit components on base rows
A = M[src == "it"]; B = M[src == "base"]
Ac = A - A.mean(0); _, _, Va = np.linalg.svd(Ac, full_matrices=False); Bc = B - B.mean(0)
tr = float(((Bc @ Va[:3].T) ** 2).sum() / (Bc ** 2).sum())
trn = []
for _ in range(1000):
    Q, _ = np.linalg.qr(rng.normal(size=(M.shape[1], 3))); trn.append(float(((Bc @ Q) ** 2).sum() / (Bc ** 2).sum()))
res = {"n_rows": M.shape[0], "n_cols": M.shape[1], "top3_explained": float(f3), "spectrum_top6": spec[:6].round(4).tolist(),
       "null_top3_p99": float(np.percentile(null, 99)), "null_top3_median": float(np.median(null)),
       "loading_ratio_group1_vs_2": ratio, "top_loadings_c1": sorted(zip(cols, Vt[0].round(3).tolist()), key=lambda x: -abs(x[1]))[:8],
       "top_loadings_c2": sorted(zip(cols, Vt[1].round(3).tolist()), key=lambda x: -abs(x[1]))[:8],
       "transfer_it_to_base_explained": tr, "transfer_null_random3_p99": float(np.percentile(trn, 99))}
res["P_i"] = res["top3_explained"] >= 0.8; res["P_ii"] = res["top3_explained"] > res["null_top3_p99"]
res["P_iii"] = max(ratio) >= 3; res["P_iv"] = tr >= 0.7 and tr > res["transfer_null_random3_p99"]
json.dump(res, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
