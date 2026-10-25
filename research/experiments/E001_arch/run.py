import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode, _header, _tensor, ROOT, MODELS
from runner import Runner
torch.manual_seed(0)
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
prompts = ["The Eiffel Tower is located in the city of", "When Mary and John went to the store, John gave a drink to"]
res = {"load": info, "equivalence": [], "perturb": [], "speed": {}}
for p in prompts:
    ids = torch.tensor(encode(tok, p))
    with torch.no_grad():
        ref = model(input_ids=ids).logits[0, -1].float()
    a = R.forward(ids)["logits"][0]; b = R.forward(ids, last_only_from=15)["logits"][0]
    res["equivalence"].append({"prompt": p, "n": ids.shape[1], "runner_vs_hf_maxabs": (a - ref).abs().max().item(),
                               "lastonly_vs_full_maxabs": (b - a).abs().max().item(), "top1": tok.decode(a.argmax())})
    for L in [5, 13, 14, 15, 16, 25, 34]:
        noise = torch.randn(1, ids.shape[1], 1536) * 50.0; noise[:, -1] = 0
        o = R.forward(ids, hooks={("resid_pre", L): (lambda h, i, n=noise: h + n)})["logits"][0]
        res["perturb"].append({"prompt": p[:25], "layer": L, "maxabs_delta_last_logits": (o - a).abs().max().item()})
    print("prompt done", flush=True)
ids = torch.cat([torch.tensor(encode(tok, prompts[0]) * 8)] * 4, dim=1)[:, :40]
for tag, k in [("full", None), ("last_only_15", 15)]:
    R.forward(ids[:1, :6], last_only_from=k); t0 = time.time(); R.forward(ids, last_only_from=k); res["speed"][tag] = round(time.time() - t0, 2)
st = os.path.join(ROOT, MODELS["it"], "model.safetensors"); h, off = _header(st)
res["ckpt_kproj_stats"] = {L: {"std": round(t.std().item(), 5), "absmax": round(t.abs().max().item(), 4)} for L in [12, 13, 14, 15, 20, 34]
                           for t in [_tensor(st, h[f"model.language_model.layers.{L}.self_attn.k_proj.weight"], off).float()]}
res["unused_ckpt_tensors_for_shared_layers"] = sum(1 for k in h if k.startswith("model.language_model.layers.")
    and int(k.split(".")[3]) >= 15 and any(s in k for s in ("k_proj", "v_proj", "k_norm", "v_norm")))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=1)
print("done")
