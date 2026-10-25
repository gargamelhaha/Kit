"""E006c (see PREREG.md)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute
from runner import Runner
from pairs import e006_plan
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_it.json"); BS = 24
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
LATE = list(range(15, 35)); H = model.config.num_attention_heads
plan = e006_plan(R, tok)
e006 = json.load(open(os.path.join(D, "..", "E006_kv_route", "result_it.json")))
ref = [(p["clean"], p["corrupt"]) for p in e006["pairs"] if p["split"] == "held_out"]
mine = [(a["subj"], b["subj"]) for _, (a, b) in plan["held_out"]]
assert ref == mine, ("pair selection differs from E006", ref, mine)


def capture(ids, attn="sdpa"):
    at, hd, pr = {}, {}, {}
    hooks = {("attn_out", L): (lambda o, i: at.__setitem__(i, o[0, -1].clone())) for L in LATE}
    hooks[("head_in", 25)] = lambda x, i: hd.__setitem__(i, x[0, -1].clone())
    if attn == "eager": hooks[("attn_probs", 25)] = lambda p, i: pr.__setitem__(i, p[0, :, -1].clone())
    o = R.forward(torch.tensor([ids]), hooks=hooks, last_only_from=15, attn=attn)
    return {"logits": o["logits"][0], "attn": at, "head": hd[25], "probs": pr.get(25)}


t0 = time.time()
caps = {it["subj"]: capture(it["ids"]) for rel in ("R1", "R2") for it in plan["ok"][rel]}
print("captured", len(caps), round(time.time() - t0), "s", flush=True)
byrel = {rel: [it["subj"] for it in plan["ok"][rel]] for rel in ("R1", "R2")}


def loo_mean(rel, excl, key, L=None):
    vs = [(caps[s]["attn"][L] if key == "attn" else caps[s]["head"]) for s in byrel[rel] if s not in excl]
    return torch.stack(vs).mean(0)


def run(ids, specs):
    outs = []
    for k in range(0, len(specs), BS):
        ch = specs[k:k + BS]
        def ah(o, i):
            o = o.clone()
            for r, s in enumerate(ch):
                if i in s.get("attn", {}): o[r, -1] = s["attn"][i]
            return o
        def hh(x, i):
            x = x.clone()
            for r, s in enumerate(ch):
                for h, v in s.get("heads", {}).items(): x[r, -1, h] = v
            return x
        hooks = {("attn_out", L): ah for L in LATE}; hooks[("head_in", 25)] = hh
        outs.append(R.forward(torch.tensor([ids] * len(ch)), hooks=hooks, last_only_from=15)["logits"])
    return torch.cat(outs)


res = {"info": info, "pairs": []}
for rel, (a, b) in plan["held_out"]:
    ca, cb = caps[a["subj"]], caps[b["subj"]]; ex = {a["subj"], b["subj"]}
    ld = lambda lg: (lg[..., a["aid"]] - lg[..., b["aid"]]).float()
    la, lb = ld(ca["logits"]).item(), ld(cb["logits"]).item()
    ea, eb = capture(a["ids"], "eager"), capture(b["ids"], "eager")
    m25 = loo_mean(rel, ex, "attn", 25)
    sp = {"B25_NOISE": {"attn": {25: cb["attn"][25]}}, "B25_MEAN": {"attn": {25: m25}},
          "B25_MEANNM": {"attn": {25: m25 * ca["attn"][25].norm() / m25.norm()}}, "B25_ZERO": {"attn": {25: torch.zeros_like(m25)}},
          "H0_NOISE": {"heads": {0: cb["head"][0]}}, "H0_MEAN": {"heads": {0: loo_mean(rel, ex, "head")[0]}},
          "H1to7_NOISE": {"heads": {h: cb["head"][h] for h in range(1, H)}}, "HALL_NOISE": {"heads": {h: cb["head"][h] for h in range(H)}},
          "LATE_ALL_MEAN": {"attn": {L: loo_mean(rel, ex, "attn", L) for L in LATE}}}
    for L in LATE:
        if L != 25: sp[f"JOINT_25_{L}"] = {"attn": {25: m25, L: loo_mean(rel, ex, "attn", L)}}
    names = list(sp)
    ret = ((ld(run(a["ids"], [sp[n] for n in names])) - lb) / (la - lb)).tolist()
    span = a["subj_span"]; T = len(a["ids"])
    res["pairs"].append({"relation": rel, "clean": a["subj"], "corrupt": b["subj"], "T": T, "subj_span": span,
                         "span_tokens": [tok.decode(a["ids"][p]) for p in span], "ld_clean": la, "ld_corrupt": lb,
                         "ld_eager_clean": ld(ea["logits"]).item(), "ld_eager_corrupt": ld(eb["logits"]).item(),
                         "mass_h": ea["probs"][:, span].sum(-1).tolist(), "uniform": len(span) / T, "retained": dict(zip(names, ret))})
    json.dump(res, open(OUT, "w"), indent=1)
    print(rel, a["subj"], "->", b["subj"], {k: round(v, 3) for k, v in zip(names, ret) if k in ("B25_MEAN", "B25_MEANNM", "H1to7_NOISE", "LATE_ALL_MEAN")},
          "eager-sdpa", round(abs(ld(ea["logits"]).item() - la), 6), round(time.time() - t0), "s", flush=True)
res["wall_s"] = round(time.time() - t0); json.dump(res, open(OUT, "w"), indent=1); print("done", flush=True)
