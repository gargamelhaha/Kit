"""E008 (see PREREG.md incl. 'Changes after pre-audit')."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute
from runner import Runner
from pairs import make_items
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_it.json"); FB = 720
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
LATE = list(range(15, 35)); BOS = tok.bos_token_id
fill = json.load(open(os.path.join(D, "..", "..", "benchmarks", "fillers_v1.json")))
F = {k: tok(" " + v["text"], add_special_tokens=False).input_ids for k, v in fill.items()}
assert all(len(f) >= FB for f in F.values()), {k: len(f) for k, f in F.items()}
T_ = lambda s: tok(s, add_special_tokens=False).input_ids


def build(sent_ids, last_idx, query_ids, d, filler):
    """[BOS] + F[:p] + sentence + F[p:p+m] + query, total filler p+m = FB; d = final index - (index of sentence token last_idx)."""
    m = d - len(query_ids) - (len(sent_ids) - 1 - last_idx)
    assert 0 <= m <= FB, (d, m)
    p = FB - m; f = F[filler]
    ids = [BOS] + f[:p] + sent_ids + f[p:p + m] + query_ids
    off = 1 + p
    assert (len(ids) - 1) - (off + last_idx) == d
    return ids, off


def run_cell(ids_a, ids_b, ans_a, ans_b, groups):
    sa, sb = R.prefix(torch.tensor([ids_a])), R.prefix(torch.tensor([ids_b]))
    at = {}
    hooks = {("attn_out", L): (lambda o, i: at.__setitem__(i, o[0, -1].clone())) for L in LATE}
    la_lg = R.suffix(sa, 1, hooks=hooks)["logits"][0]; lb_lg = R.suffix(sb, 1)["logits"][0]
    ld = lambda lg: (lg[..., ans_a] - lg[..., ans_b]).float()
    la, lb = ld(la_lg).item(), ld(lb_lg).item()
    specs = {}
    for g, pos in groups.items():
        specs[f"KV13_{g}"] = {13: pos}; specs[f"KV14_{g}"] = {14: pos}
    allp = list(range(len(ids_a)))
    specs.update({"KV13_all": {13: allp}, "KV14_all": {14: allp}, "KVBOTH_all": {13: allp, 14: allp}})
    if "D" in groups: specs["KVBOTH_D"] = {13: groups["D"], 14: groups["D"]}
    specs["NULL"] = {"null": True, 13: allp, 14: allp}
    specs.update({f"ATTN_{L}": {"attn": L} for L in LATE})
    names = list(specs); rows = [specs[n] for n in names]
    def kvh(kv, i):
        k, v = kv[0].clone(), kv[1].clone(); src = sb if False else None
        lt = "sliding_attention" if i == 13 else "full_attention"
        for r, s in enumerate(rows):
            P = s.get(i)
            if P:
                S = sb if s.get("null") else sa
                k[r, :, P] = S["shared"][lt][0][0, :, P]; v[r, :, P] = S["shared"][lt][1][0, :, P]
        return k, v
    def ah(o, i):
        o = o.clone()
        for r, s in enumerate(rows):
            if s.get("attn") == i: o[r, -1] = at[i]
        return o
    hk = {("kv_store", 13): kvh, ("kv_store", 14): kvh}; hk.update({("attn_out", L): ah for L in LATE})
    out = R.suffix(sb, len(rows), hooks=hk)["logits"]
    rec = ((ld(out) - lb) / (la - lb)).tolist() if abs(la - lb) > 1e-9 else [None] * len(rows)
    return {"ld_clean": la, "ld_corrupt": lb, "clean_top1": int(la_lg.argmax()), "corrupt_top1": int(lb_lg.argmax()),
            "recovery": dict(zip(names, rec))}


t0 = time.time(); res = {"info": info, "A": [], "B": []}
# ---------- Part A ----------
items = [i for i in make_items(R, tok, "R1") if i["correct"]]
note = lambda s: T_(f" Note: the landmark in this report is the {s}.")
q_a = T_(" The landmark named in the note is located in the city of")
for it in items:
    n = note(it["subj"]); pre = T_(" Note: the landmark in this report is the"); it["note"] = n
    it["last_in_note"] = len(T_(" Note: the landmark in this report is the " + it["subj"])) - 1
    assert n[:len(pre)] == pre
cand = [(a, b) for a in items for b in items if a is not b and len(a["note"]) == len(b["note"]) and a["last_in_note"] == b["last_in_note"]
        and a["aid"] != b["aid"]]
cand = [cand[j] for j in torch.randperm(len(cand), generator=torch.Generator().manual_seed(0)).tolist()]
seen, pairsA = set(), []
for a, b in cand:
    k = frozenset((a["subj"], b["subj"]))
    if k not in seen: seen.add(k); pairsA.append((a, b))
pairsA = pairsA[:10]
print("Part A pairs", len(pairsA), [(a["subj"], b["subj"]) for a, b in pairsA], flush=True)
cells = [(fl, d) for fl, ds in (("A", (60, 400, 505, 525, 700)), ("B", (60, 700))) for d in ds]
for fl, d in cells:
    for a, b in pairsA:
        ia, off = build(a["note"], a["last_in_note"], q_a, d, fl); ib, _ = build(b["note"], b["last_in_note"], q_a, d, fl)
        first_subj = off + len(T_(" Note: the landmark in this report is the"))
        Dp = list(range(first_subj, off + a["last_in_note"] + 1))
        qstart = len(ia) - len(q_a)
        groups = {"D": Dp, "BETWEEN": list(range(off + len(a["note"]), qstart)), "TPL": list(range(qstart, len(ia)))}
        r = run_cell(ia, ib, a["aid"], b["aid"], groups)
        r.update(filler=fl, d=d, clean=a["subj"], corrupt=b["subj"], T=len(ia), D=Dp, clean_correct=r["clean_top1"] == a["aid"],
                 corrupt_correct=r["corrupt_top1"] == b["aid"])
        res["A"].append(r); json.dump(res, open(OUT, "w"), indent=1)
        rc = r["recovery"]
        print("A", fl, d, a["subj"], "->", b["subj"], "ok", r["clean_correct"], r["corrupt_correct"], "LDs", round(r["ld_clean"], 1), round(r["ld_corrupt"], 1),
              {k: (round(rc[k], 3) if rc[k] is not None else None) for k in ("KV13_D", "KV14_D", "KVBOTH_D", "ATTN_25", "NULL")}, round(time.time() - t0), "s", flush=True)
# ---------- Part B ----------
WORDS = ["apple", "river", "candle", "mirror", "garden", "pencil", "rocket", "violin", "basket", "ladder", "tiger", "anchor",
         "button", "castle", "dragon", "feather", "hammer", "island", "jacket", "kettle", "lemon", "magnet", "needle", "orange"]
WORDS = [w for w in WORDS if len(T_(" " + w)) == 1]
COL = ["red", "blue", "green"]
g = torch.Generator().manual_seed(1)
itemsB = []
for k in range(12):
    perm = torch.randperm(len(WORDS), generator=g).tolist()
    w = [WORDS[perm[0]], WORDS[perm[1]], WORDS[perm[2]]]; alt = WORDS[perm[3]]; qc = k % 3
    itemsB.append((w, alt, qc))
for d in (60, 700):
    for w, alt, qc in itemsB:
        def sent(ws):
            return T_(f" In this game, the red box holds the word {ws[0]}, the blue box holds the word {ws[1]}, and the green box holds the word {ws[2]}.")
        wb = list(w); wb[qc] = alt
        sa_, sb_ = sent(w), sent(wb)
        assert len(sa_) == len(sb_)
        diff = [i for i in range(len(sa_)) if sa_[i] != sb_[i]]; assert len(diff) == 1
        q = T_(f" The {COL[qc]} box holds the word")
        ia, off = build(sa_, diff[0], q, d, "A"); ib, _ = build(sb_, diff[0], q, d, "A")
        # queried clause: from the colour token to the word token
        col_tok = T_(f" {COL[qc]}")[0]
        cpos = max(i for i in range(diff[0]) if sa_[i] == col_tok)
        qstart = len(ia) - len(q)
        groups = {"D": [off + diff[0]], "N": [off + cpos], "CLAUSE": list(range(off + cpos, off + diff[0] + 1)),
                  "AFTER": list(range(off + len(sa_), len(ia)))}
        aa, ab = T_(" " + w[qc])[0], T_(" " + alt)[0]
        r = run_cell(ia, ib, aa, ab, groups)
        r.update(d=d, words=w, alt=alt, queried=COL[qc], T=len(ia), clean_correct=r["clean_top1"] == aa, corrupt_correct=r["corrupt_top1"] == ab)
        res["B"].append(r); json.dump(res, open(OUT, "w"), indent=1)
        rc = r["recovery"]
        print("B", d, COL[qc], w[qc], "->", alt, "ok", r["clean_correct"], r["corrupt_correct"],
              {k: (round(rc[k], 3) if rc[k] is not None else None) for k in ("KV13_D", "KV14_D", "KVBOTH_D", "KV13_AFTER", "KV14_AFTER", "NULL")}, round(time.time() - t0), "s", flush=True)
res["wall_s"] = round(time.time() - t0); json.dump(res, open(OUT, "w"), indent=1); print("done", flush=True)
