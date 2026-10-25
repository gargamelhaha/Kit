"""E006b: block-25 route specificity controls (see PREREG.md). Reuses E006's item/pair selection."""
import json, os, sys, time, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
from datasets import LANDMARKS, LANDMARK_TPL, CAPITALS, CAPITAL_TPL
D_ = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D_, "result_it.json"); BS = 24
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
LATE = list(range(15, 35)); H = model.config.num_attention_heads

# --- identical item / pair selection as E006 (copied, not imported, to avoid re-running E006) ---
import itertools
def make_items(src, tpl, prefix):
    items = []
    for s, c in src:
        ids = encode(tok, tpl.format(s))[0]
        items.append({"subj": s, "ids": ids, "aid": tok(" " + c, add_special_tokens=False).input_ids[0],
                      "last_subj_pos": len(encode(tok, prefix + s)[0]) - 1})
    for it in items:
        it["correct"] = int(R.forward(torch.tensor([it["ids"]]), last_only_from=15)["logits"][0].argmax().item() == it["aid"])
    return items
def choose_pairs(items):
    ok = [i for i in items if i["correct"]]
    pairs = [(a, b) for a, b in itertools.permutations(ok, 2) if len(a["ids"]) == len(b["ids"]) and a["aid"] != b["aid"]
             and a["last_subj_pos"] == b["last_subj_pos"]]
    pairs = [pairs[j] for j in torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0)).tolist()]
    seen, chosen = set(), []
    for a, b in pairs:
        k = frozenset((a["subj"], b["subj"]))
        if k not in seen: seen.add(k); chosen.append((a, b))
    return chosen
chL = choose_pairs(make_items(LANDMARKS, LANDMARK_TPL, "The "))
chC = choose_pairs(make_items(CAPITALS, CAPITAL_TPL, "The capital of "))
plan = [("R1", p) for p in chL[12:24]] + [("R2", p) for p in chC[:16]]


def capture(ids, probs=False):
    attn, head, mass = {}, {}, {}
    def ca(o, i): attn[i] = o[0, -1].clone()
    def ch(x, i): head[i] = x[0, -1].clone()
    def cp(p, i): mass[i] = p[0, :, -1].clone()  # [heads, keys] at the final query
    hooks = {("attn_out", L): ca for L in LATE}; hooks[("head_in", 25)] = ch
    if probs: hooks.update({("attn_probs", L): cp for L in LATE})
    o = R.forward(torch.tensor([ids]), hooks=hooks, last_only_from=15, attn="eager" if probs else "sdpa")
    return {"logits": o["logits"][0], "attn": attn, "head": head[25], "probs": mass}


def run_specs(ids, specs):
    outs = []
    for k in range(0, len(specs), BS):
        chunk = specs[k:k + BS]
        def ah(o, i):
            o = o.clone()
            for r, s in enumerate(chunk):
                if s.get("attn") == i: o[r, -1] = s["vec"]
            return o
        def hh(x, i):
            x = x.clone()
            for r, s in enumerate(chunk):
                if s.get("head") is not None: x[r, -1, s["head"]] = s["vec"]
            return x
        hooks = {("attn_out", L): ah for L in LATE}; hooks[("head_in", 25)] = hh
        outs.append(R.forward(torch.tensor([ids] * len(chunk)), hooks=hooks, last_only_from=15)["logits"])
    return torch.cat(outs)


t0 = time.time(); caps = {}
for rel, (a, b) in plan:
    for x in (a, b):
        if x["subj"] not in caps: caps[x["subj"]] = capture(x["ids"], probs=True)
means = {rel: torch.stack([caps[a["subj"]]["attn"][25] for r, (a, b) in plan if r == rel]).mean(0) for rel in ("R1", "R2")}
res = {"info": info, "pairs": []}
for rel, (a, b) in plan:
    ca, cb = caps[a["subj"]], caps[b["subj"]]; T = len(a["ids"])
    Dp = [p for p in range(T) if a["ids"][p] != b["ids"][p]]
    ld = lambda lg: (lg[..., a["aid"]] - lg[..., b["aid"]]).float()
    la, lb = ld(ca["logits"]).item(), ld(cb["logits"]).item()
    specs = {f"NOISE_ATTN_{L}": {"attn": L, "vec": cb["attn"][L]} for L in LATE}
    specs["MEAN_ATTN25"] = {"attn": 25, "vec": means[rel]}
    specs.update({f"NOISE_HEAD25_{h}": {"head": h, "vec": cb["head"][h]} for h in range(H)})
    names = list(specs)
    ret = ((ld(run_specs(a["ids"], [specs[n] for n in names])) - lb) / (la - lb)).tolist()
    mass = {f"MASS25_{h}": ca["probs"][25][h, Dp].sum().item() for h in range(H)}
    mass.update({f"MASS_{L}": ca["probs"][L][:, Dp].sum(-1).mean().item() for L in LATE})
    res["pairs"].append({"relation": rel, "clean": a["subj"], "corrupt": b["subj"], "T": T, "D": Dp, "uniform_mass": len(Dp) / T,
                         "ld_clean": la, "ld_corrupt": lb, "retained": dict(zip(names, ret)), "mass": mass})
    json.dump(res, open(OUT, "w"), indent=1)
    print(rel, a["subj"], "->", b["subj"], {k: round(v, 3) for k, v in zip(names, ret) if k in ("NOISE_ATTN_25", "MEAN_ATTN25", "NOISE_HEAD25_0")},
          "mass25_0", round(mass["MASS25_0"], 3), round(time.time() - t0), "s", flush=True)
res["wall_s"] = round(time.time() - t0); json.dump(res, open(OUT, "w"), indent=1); print("done", flush=True)
