"""Shared statistics (AUDIT-004 fixes): tie-correct AUROC, two-subject cluster bootstrap with multiplicities."""
import numpy as np
from scipy.stats import rankdata


def auroc(score, y):
    """Mann-Whitney AUROC with average ranks for ties."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if not n1 or not n0: return None
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wmedian(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    m = w > 0; v, w = v[m], w[m]
    if not len(v): return None
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2)])


def pair_cluster_ci(pairs, val, strata=None, n=10000, seed=0):
    """Pairs (clean, corrupt) share subjects. Resample subjects with replacement (within strata); a pair's weight is
    mult(clean) * mult(corrupt). Returns the 95% CI of the weighted median."""
    rng = np.random.default_rng(seed)
    strata = strata or (lambda p: 0)
    groups = {}
    for p in pairs: groups.setdefault(strata(p), []).append(p)
    subs = {g: sorted({p["clean"] for p in ps} | {p["corrupt"] for p in ps}) for g, ps in groups.items()}
    meds = []
    for _ in range(n):
        vs, ws = [], []
        for g, ps in groups.items():
            S = subs[g]; cnt = dict(zip(S, np.bincount(rng.integers(0, len(S), len(S)), minlength=len(S))))
            for p in ps:
                vs.append(val(p)); ws.append(cnt[p["clean"]] * cnt[p["corrupt"]])
        m = wmedian(vs, ws)
        if m is not None: meds.append(m)
    return [round(float(np.percentile(meds, 2.5)), 4), round(float(np.percentile(meds, 97.5)), 4)]
