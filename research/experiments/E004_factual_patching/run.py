import itertools, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
from datasets import LANDMARKS, LANDMARK_TPL
V = os.environ.get("VARIANT", "it"); MAXPAIRS = int(os.environ.get("MAXPAIRS", 12)); BS = 24
OUT = os.path.join(os.path.dirname(__file__), f"result_{V}.json")
model, tok, info = load(V); use_fp32_compute(model); R = Runner(model)
items = []
for s, c in LANDMARKS:
    ids = encode(tok, LANDMARK_TPL.format(s))[0]
    items.append({"subj": s, "city": c, "ids": ids, "aid": tok(" " + c, add_special_tokens=False).input_ids[0],
                  "last_subj_pos": len(encode(tok, "The " + s)[0]) - 1})
for it in items:
    lg = R.forward(torch.tensor([it["ids"]]), last_only_from=15)["logits"][0]
    it["top1"] = tok.decode(lg.argmax()); it["correct"] = int(lg.argmax().item() == it["aid"])
ok = [i for i in items if i["correct"]]
pairs = [(a, b) for a, b in itertools.permutations(ok, 2) if len(a["ids"]) == len(b["ids"]) and a["aid"] != b["aid"]
         and a["last_subj_pos"] == b["last_subj_pos"]]
pairs = [pairs[j] for j in torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0)).tolist()]
seen, chosen = set(), []
for a, b in pairs:  # use each unordered pair once
    key = frozenset((a["subj"], b["subj"]))
    if key not in seen:
        seen.add(key); chosen.append((a, b))
chosen = chosen[:MAXPAIRS]
print("correct", len(ok), "/", len(items), "pairs", len(chosen), flush=True)
def run_conditions(corrupt_ids, src, conds):
    outs = []
    for k in range(0, len(conds), BS):
        chunk = conds[k:k + BS]
        def mk(L):
            def f(h, i):
                h = h.clone()
                for r, (l, p) in enumerate(chunk):
                    if l == L:
                        if h.shape[1] == 1: h[r, 0] = src[L][0, -1]
                        else: h[r, p] = src[L][0, p]
                return h
            return f
        outs.append(R.forward(torch.tensor([corrupt_ids] * len(chunk)), hooks={("resid_pre", L): mk(L) for L in {l for l, _ in chunk}},
                              last_only_from=15)["logits"])
    return torch.cat(outs)
t0 = time.time(); res = {"info": info, "items": [{k: v for k, v in i.items() if k != "ids"} for i in items], "pairs": []}
for a, b in chosen:
    T = len(a["ids"]); ca = R.forward(torch.tensor([a["ids"]]), last_only_from=15, keep_resid=True)
    cb = R.forward(torch.tensor([b["ids"]]), last_only_from=15)
    ld = lambda lg: (lg[..., a["aid"]] - lg[..., b["aid"]]).float()
    ld_c, ld_x = ld(ca["logits"][0]).item(), ld(cb["logits"][0]).item()
    conds = [(L, p) for L in range(15) for p in range(T)] + [(L, -1) for L in range(15, 35)]
    rec = ((ld(run_conditions(b["ids"], ca["resid"], conds)) - ld_x) / (ld_c - ld_x)).tolist()
    grid = [[None] * T for _ in range(15)]; late = []
    for (L, p), r in zip(conds, rec):
        if L < 15: grid[L][p] = r
        else: late.append(r)
    third = next((c for c in ok if c is not a and c is not b and len(c["ids"]) == T and c["aid"] not in (a["aid"], b["aid"])
                  and c["last_subj_pos"] == a["last_subj_pos"]), None)
    ctrl = None
    if third:
        cc = R.forward(torch.tensor([third["ids"]]), last_only_from=15, keep_resid=True)
        cconds = [(L, a["last_subj_pos"]) for L in range(15)] + [(L, -1) for L in range(15, 35)]
        ctrl = {"third": third["subj"], "recovery": ((ld(run_conditions(b["ids"], cc["resid"], cconds)) - ld_x) / (ld_c - ld_x)).tolist()}
    res["pairs"].append({"clean": a["subj"], "corrupt": b["subj"], "T": T, "last_subj_pos": a["last_subj_pos"],
                         "tokens": [tok.decode(t) for t in a["ids"]], "ld_clean": ld_c, "ld_corrupt": ld_x,
                         "grid_early": grid, "final_pos_late": late, "control": ctrl})
    json.dump(res, open(OUT, "w"), indent=1)
    print("pair", a["subj"], "->", b["subj"], round(time.time() - t0), "s", flush=True)
res["wall_s"] = round(time.time() - t0); json.dump(res, open(OUT, "w"), indent=1); print("done")
