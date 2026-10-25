"""E006: K/V-route and block-25 attention tests for factual recall (see PREREG.md)."""
import itertools, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
from datasets import LANDMARKS, LANDMARK_TPL, CAPITALS, CAPITAL_TPL
V = os.environ.get("VARIANT", "it"); BS = 24
OUT = os.path.join(os.path.dirname(__file__), f"result_{V}.json")
model, tok, info = load(V); use_fp32_compute(model); R = Runner(model)
LATE = list(range(15, 35))


def make_items(pairs_src, tpl, prefix):
    items = []
    for s, c in pairs_src:
        ids = encode(tok, tpl.format(s))[0]
        items.append({"subj": s, "city": c, "ids": ids, "aid": tok(" " + c, add_special_tokens=False).input_ids[0],
                      "last_subj_pos": len(encode(tok, prefix + s)[0]) - 1})
    for it in items:
        lg = R.forward(torch.tensor([it["ids"]]), last_only_from=15)["logits"][0]
        it["top1"] = tok.decode(lg.argmax()); it["correct"] = int(lg.argmax().item() == it["aid"])
    return items


def choose_pairs(items):  # identical to E004's selection procedure
    ok = [i for i in items if i["correct"]]
    pairs = [(a, b) for a, b in itertools.permutations(ok, 2) if len(a["ids"]) == len(b["ids"]) and a["aid"] != b["aid"]
             and a["last_subj_pos"] == b["last_subj_pos"]]
    pairs = [pairs[j] for j in torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0)).tolist()]
    seen, chosen = set(), []
    for a, b in pairs:
        key = frozenset((a["subj"], b["subj"]))
        if key not in seen:
            seen.add(key); chosen.append((a, b))
    return ok, chosen


def capture(ids):
    """Clean/corrupt reference run: logits, stored K/V, final-position attention outputs (L>=15), block-25 head inputs."""
    attn, head = {}, {}
    def cap_attn(o, i): attn[i] = o[0, -1].clone()
    def cap_head(x, i): head[i] = x[0, -1].clone()
    hooks = {("attn_out", L): cap_attn for L in LATE}; hooks[("head_in", 25)] = cap_head
    o = R.forward(torch.tensor([ids]), hooks=hooks, last_only_from=15, return_shared=True)
    return {"logits": o["logits"][0], "shared": o["shared"], "attn": attn, "head": head[25]}


def run_specs(ids, specs, srcs):
    """Batched forward of `ids`; row r is modified according to specs[r] using reference runs srcs[name]."""
    outs = []
    for k in range(0, len(specs), BS):
        chunk = specs[k:k + BS]
        def kv_hook(kv, i):
            name = "kv13" if i == 13 else "kv14"
            key, val = kv[0].clone(), kv[1].clone()
            for r, s in enumerate(chunk):
                P = s.get(name)
                if P is not None:
                    sk, sv = srcs[s["src"]]["shared"]["sliding_attention" if i == 13 else "full_attention"]
                    key[r, :, P] = sk[0, :, P]; val[r, :, P] = sv[0, :, P]
            return key, val
        def attn_hook(o, i):
            o = o.clone()
            for r, s in enumerate(chunk):
                if s.get("attn") == i:
                    o[r, -1] = srcs[s["src"]]["attn"][i]
            return o
        def head_hook(x, i):
            x = x.clone()
            for r, s in enumerate(chunk):
                if s.get("head") is not None:
                    x[r, -1, s["head"]] = srcs[s["src"]]["head"][s["head"]]
            return x
        hooks = {("kv_store", 13): kv_hook, ("kv_store", 14): kv_hook, ("head_in", 25): head_hook}
        hooks.update({("attn_out", L): attn_hook for L in LATE})
        outs.append(R.forward(torch.tensor([ids] * len(chunk)), hooks=hooks, last_only_from=15)["logits"])
    return torch.cat(outs)


def run_pair(a, b):
    T = len(a["ids"]); allp = list(range(T))
    D = [p for p in range(T) if a["ids"][p] != b["ids"][p]]; S = [a["last_subj_pos"]]
    Tm = [p for p in range(T) if p not in D]
    ca, cb = capture(a["ids"]), capture(b["ids"])
    ld = lambda lg: (lg[..., a["aid"]] - lg[..., b["aid"]]).float()
    ld_a, ld_b = ld(ca["logits"]).item(), ld(cb["logits"]).item()
    specs = {"KV13_all": {"kv13": allp}, "KV14_all": {"kv14": allp}, "KVBOTH_all": {"kv13": allp, "kv14": allp},
             "KV13_D": {"kv13": D}, "KV14_D": {"kv14": D}, "KV13_S": {"kv13": S}, "KV13_T": {"kv13": Tm}}
    specs = {k: {**v, "src": "a"} for k, v in specs.items()}
    specs["NULL"] = {"kv13": allp, "kv14": allp, "src": "b"}
    specs.update({f"ATTN_{L}": {"attn": L, "src": "a"} for L in LATE})
    specs.update({f"HEAD25_{h}": {"head": h, "src": "a"} for h in range(model.config.num_attention_heads)})
    names = list(specs)
    rec = ((ld(run_specs(b["ids"], [specs[n] for n in names], {"a": ca, "b": cb})) - ld_b) / (ld_a - ld_b)).tolist()
    out = dict(zip(names, rec))
    noise = run_specs(a["ids"], [{"attn": 25, "src": "b"}], {"b": cb})
    out["NOISE_ATTN25"] = ((ld(noise) - ld_b) / (ld_a - ld_b)).item()
    return {"clean": a["subj"], "corrupt": b["subj"], "T": T, "D": D, "S": S[0], "ld_clean": ld_a, "ld_corrupt": ld_b,
            "tokens": [tok.decode(t) for t in a["ids"]], "recovery": out}


t0 = time.time()
landm = make_items(LANDMARKS, LANDMARK_TPL, "The ")
caps = make_items(CAPITALS, CAPITAL_TPL, "The capital of ")
okL, chL = choose_pairs(landm); okC, chC = choose_pairs(caps)
plan = [("R1", "in_sample", p) for p in chL[:12]] + [("R1", "held_out", p) for p in chL[12:24]] + \
       [("R2", "held_out", p) for p in chC[:16]]
res = {"info": info, "n_correct": {"R1": [len(okL), len(landm)], "R2": [len(okC), len(caps)]},
       "items_R2": [{k: v for k, v in i.items() if k != "ids"} for i in caps], "pairs": []}
print("correct R1", len(okL), "/", len(landm), "R2", len(okC), "/", len(caps), "plan", len(plan), flush=True)
for rel, split, (a, b) in plan:
    r = run_pair(a, b); r.update(relation=rel, split=split); res["pairs"].append(r)
    json.dump(res, open(OUT, "w"), indent=1)
    rc = r["recovery"]
    print(rel, split, a["subj"], "->", b["subj"], {k: round(rc[k], 3) for k in ("KV13_all", "KV14_all", "ATTN_25", "NULL")},
          round(time.time() - t0), "s", flush=True)
res["wall_s"] = round(time.time() - t0); json.dump(res, open(OUT, "w"), indent=1); print("done", flush=True)
