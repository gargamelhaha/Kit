"""E001b: persistent-buffer fp32 compute must be bit-identical to per-call upcasting (process change PL-6)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch, torch.nn.functional as F
from torch import nn
import gemma_loader
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
ids = torch.tensor(encode(tok, "The Eiffel Tower is located in the city of"))
t0 = time.time(); a = R.forward(ids)["logits"][0]; ta = time.time() - t0
a15 = R.forward(ids, last_only_from=15)["logits"][0]
saved = {}
for m in model.modules():  # temporarily switch to unbuffered per-call upcasting
    if isinstance(m, nn.Linear) and m.weight.shape[0] <= 65536:
        saved[m] = m.forward; m.forward = (lambda mm: (lambda x: F.linear(x.float(), mm.weight.float())))(m)
t0 = time.time(); b = R.forward(ids)["logits"][0]; tb = time.time() - t0
for m, f in saved.items(): m.forward = f
res = {"buffered_vs_unbuffered_maxabs": (a - b).abs().max().item(), "lastonly_vs_full_maxabs": (a15 - a).abs().max().item(),
       "top1": tok.decode(a.argmax()), "s_buffered": round(ta, 2), "s_unbuffered": round(tb, 2),
       "rss_guard_peak_gb": round(getattr(gemma_loader.start_rss_guard, "peak", -1), 2)}
json.dump(res, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=1); print(res)
