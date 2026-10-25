"""Equivalence + speed test for Runner.prefix/suffix vs Runner.forward(last_only_from=15) (process change PL-12).
Workload = 3 E006 conditions on 2 pairs (short) + 1 long prompt (T=700), identical numerics required within 1e-4."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
from datasets import LANDMARK_TPL
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
a = encode(tok, LANDMARK_TPL.format("Eiffel Tower"))[0]; b = encode(tok, LANDMARK_TPL.format("Colosseum"))[0]
assert len(a) == len(b), (len(a), len(b))
long_txt = open(os.path.join(os.path.dirname(__file__), "..", "..", "LITERATURE_MAP.md")).read()
L = encode(tok, long_txt)[0][:700]
out = {"rows": []}
for name, (ca_ids, cb_ids) in {"short": (a, b), "long700": (L, L[:-1] + [a[-1]])}.items():
    ca = R.forward(torch.tensor([ca_ids]), last_only_from=15, return_shared=True, allow_long=True)
    attn25 = {}
    R.forward(torch.tensor([ca_ids]), hooks={("attn_out", 25): lambda o, i: attn25.setdefault(25, o[0, -1].clone())}, last_only_from=15, allow_long=True)
    def kv13(kv, i):
        if i != 13: return None
        k, v = kv[0].clone(), kv[1].clone(); k[0] = ca["shared"]["sliding_attention"][0][0]; v[0] = ca["shared"]["sliding_attention"][1][0]; return k, v
    def at25(o, i):
        o = o.clone(); o[1, -1] = attn25[25]; return o
    hooks = {("kv_store", 13): kv13, ("kv_store", 14): kv13, ("attn_out", 25): at25}
    t0 = time.perf_counter(); ref = R.forward(torch.tensor([cb_ids] * 3), hooks=hooks, last_only_from=15, allow_long=True)["logits"]; t_ref = time.perf_counter() - t0
    t0 = time.perf_counter(); st = R.prefix(torch.tensor([cb_ids])); t_pre = time.perf_counter() - t0
    t0 = time.perf_counter(); new = R.suffix(st, 3, hooks=hooks)["logits"]; t_suf = time.perf_counter() - t0
    row = {"case": name, "T": len(cb_ids), "max_abs_diff": float((ref - new).abs().max()), "top1_same": bool((ref.argmax(-1) == new.argmax(-1)).all()),
           "row_effects": [float((ref[i] - ref[2]).abs().max()) for i in range(2)],
           "forward_3cond_s": round(t_ref, 3), "prefix_s": round(t_pre, 3), "suffix_3cond_s": round(t_suf, 3)}
    out["rows"].append(row); print(json.dumps(row), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=1)
