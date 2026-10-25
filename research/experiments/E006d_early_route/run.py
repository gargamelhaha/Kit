"""E006d (see PREREG.md)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute
from runner import Runner
from pairs import e006_plan
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_it.json")
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
ALL = list(range(35)); EARLY = list(range(15)); LATE = list(range(15, 35))
plan = e006_plan(R, tok)


def capture(ids):
    at, ml = {}, {}
    hooks = {("attn_out", L): (lambda o, i: at.__setitem__(i, o[0, -1].clone())) for L in ALL}
    hooks.update({("mlp_out", L): (lambda o, i: ml.__setitem__(i, o[0, -1].clone())) for L in LATE})
    o = R.forward(torch.tensor([ids]), hooks=hooks, last_only_from=15)
    return {"logits": o["logits"][0], "attn": at, "mlp": ml}


caps = {it["subj"]: capture(it["ids"]) for rel in ("R1", "R2") for it in plan["ok"][rel]}
by = {rel: plan["ok"][rel] for rel in ("R1", "R2")}
def loo(rel, ex, kind, L): return torch.stack([caps[i["subj"]][kind][L] for i in by[rel] if i["subj"] not in ex]).mean(0)
g = torch.Generator().manual_seed(0); res = {"info": info, "pairs": []}; t0 = time.time()
for rel, (a, b) in plan["held_out"]:
    ex = {a["subj"], b["subj"]}; ca, cb = caps[a["subj"]], caps[b["subj"]]
    others = [i for i in by[rel] if i["subj"] not in ex and i["aid"] != a["aid"]]
    rs = others[int(torch.randint(len(others), (1,), generator=g))]
    specs = {"LATE_ATTN_MEAN": {"attn": {L: loo(rel, ex, "attn", L) for L in LATE}},
             "EARLY_ATTN_MEAN": {"attn": {L: loo(rel, ex, "attn", L) for L in EARLY}},
             "ALL_ATTN_MEAN": {"attn": {L: loo(rel, ex, "attn", L) for L in ALL}},
             "LATE_ATTN_RESAMPLE": {"attn": {L: caps[rs["subj"]]["attn"][L] for L in LATE}},
             "LATE_MLP_MEAN": {"mlp": {L: loo(rel, ex, "mlp", L) for L in LATE}}}
    specs["ALL_ATTN_MEAN+LATE_MLP_MEAN"] = {"attn": specs["ALL_ATTN_MEAN"]["attn"], "mlp": specs["LATE_MLP_MEAN"]["mlp"]}
    names = list(specs); rows = [specs[n] for n in names]
    def mk(kind):
        def h(o, i):
            o = o.clone()
            for r, s in enumerate(rows):
                if i in s.get(kind, {}): o[r, -1] = s[kind][i]
            return o
        return h
    hooks = {("attn_out", L): mk("attn") for L in ALL}; hooks.update({("mlp_out", L): mk("mlp") for L in LATE})
    lg = R.forward(torch.tensor([a["ids"]] * len(rows)), hooks=hooks, last_only_from=15)["logits"]
    ld = lambda x: (x[..., a["aid"]] - x[..., b["aid"]]).float()
    la, lb = ld(ca["logits"]).item(), ld(cb["logits"]).item()
    ret = ((ld(lg) - lb) / (la - lb)).tolist()
    res["pairs"].append({"relation": rel, "clean": a["subj"], "corrupt": b["subj"], "resample_from": rs["subj"], "ld_clean": la, "ld_corrupt": lb, "retained": dict(zip(names, ret))})
    json.dump(res, open(OUT, "w"), indent=1); print(rel, a["subj"], {k: round(v, 3) for k, v in zip(names, ret)}, round(time.time() - t0), flush=True)
print("done", flush=True)
