import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute
from runner import Runner
T, B, K, NRAND = 30, 16, 5, 30
V = os.environ.get("VARIANT", "it"); OUT = os.path.join(os.path.dirname(__file__), f"result_{V}.json")
g = torch.Generator().manual_seed(0)
model, tok, info = load(V); use_fp32_compute(model); R = Runner(model)
H, L = model.config.num_attention_heads, model.config.num_hidden_layers
seq = torch.randint(1000, 30000, (B, T), generator=g)
ids = torch.cat([torch.full((B, 1), tok.bos_token_id), seq, seq], dim=1)
def losses(out):
    nll = -torch.log_softmax(out["logits"].float(), -1)[:, :-1].gather(-1, ids[:, 1:, None])[..., 0]
    return nll[:, 1:T].mean().item(), nll[:, T + 1:].mean().item()
t0 = time.time(); scores = torch.zeros(L, H); means = {}
q = torch.arange(T + 1, 2 * T + 1); kk = q - T + 1
def probs_hook(p, i): scores[i] = p[:, :, q, kk].mean(dim=(0, 2)).float()
def mean_hook(h, i): means[i] = h.mean(dim=(0, 1), keepdim=True).clone()
hooks = {("attn_probs", i): probs_hook for i in range(L)}; hooks.update({("head_in", i): mean_hook for i in range(L)})
l1, l2 = losses(R.forward(ids, hooks=hooks, attn="eager", logits="all"))
top = sorted([(s.item(), l, h) for l in range(L) for h, s in enumerate(scores[l])], reverse=True)[:K]
def ablate(heads):
    by = {}
    for l, h in heads: by.setdefault(l, []).append(h)
    def mk(hs):
        def f(x, i):
            x = x.clone(); x[:, :, hs] = means[i][:, :, hs]; return x
        return f
    a1, a2 = losses(R.forward(ids, hooks={("head_in", l): mk(hs) for l, hs in by.items()}, logits="all"))
    return a1 - l1, a2 - l2
res = {"info": info, "clean_nll": {"repeat1": l1, "repeat2": l2}, "scores": scores.tolist(),
       "top_heads": [{"layer": l, "head": h, "score": s} for s, l, h in top]}
res["top_ablation"] = dict(zip(["d_repeat1", "d_repeat2"], ablate([(l, h) for _, l, h in top])))
print("clean", l1, l2, "top", res["top_ablation"], round(time.time() - t0), flush=True)
rg = torch.Generator().manual_seed(1); allh = [(l, h) for l in range(L) for h in range(H)]; res["random_ablation"] = []
for r in range(NRAND):
    sel = [allh[j] for j in torch.randperm(len(allh), generator=rg)[:K].tolist()]
    d1, d2 = ablate(sel); res["random_ablation"].append({"heads": sel, "d_repeat1": d1, "d_repeat2": d2})
    json.dump(res, open(OUT, "w"), indent=1)
    print("rand", r, round(d1, 4), round(d2, 4), round(time.time() - t0), flush=True)
res["wall_s"] = round(time.time() - t0); json.dump(res, open(OUT, "w"), indent=1); print("done")
