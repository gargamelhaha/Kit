"""E010: weight-only INT8 / INT4(g32) quantisation quality + mechanism preservation (see PREREG.md)."""
import gc, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
import torch.nn.functional as F
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
from pairs import e006_plan
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_it.json")
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 2**20
lin = [(n, m) for n, m in model.model.named_modules() if isinstance(m, torch.nn.Linear)]
bf16_bytes = sum(m.weight.numel() * 2 for _, m in lin)
items = [json.loads(l) for l in open(os.path.join(D, "..", "..", "benchmarks", "factset_v1.jsonl"))]
test = [it for n, it in enumerate(items) if n % 2 == 1]
plan = e006_plan(R, tok); r2 = [p for rel, p in plan["held_out"] if rel == "R2"]


def factset_pass():
    out = []
    for it in test:
        ids = encode(tok, it["prompt"])[0]; tgt = tok(" " + it["answer"], add_special_tokens=False).input_ids[0]
        lg = R.forward(torch.tensor([ids]), last_only_from=15)["logits"][0].float()
        lp = torch.log_softmax(lg, -1); top = lp.topk(64)
        out.append({"id": it["id"], "top1": int(lp.argmax()), "correct": int(lp.argmax()) == tgt, "top_idx": top.indices.tolist(), "top_lp": top.values.tolist()})
    return out


def mech_pass():
    res = []
    for a, b in r2:
        ca = R.forward(torch.tensor([a["ids"]]), last_only_from=15, return_shared=True)
        at = {}
        R.forward(torch.tensor([a["ids"]]), hooks={("attn_out", 25): lambda o, i: at.__setitem__("a", o[0, -1].clone())}, last_only_from=15)
        cb = R.forward(torch.tensor([b["ids"]]), last_only_from=15)
        bt = {}
        R.forward(torch.tensor([b["ids"]]), hooks={("attn_out", 25): lambda o, i: bt.__setitem__("b", o[0, -1].clone())}, last_only_from=15)
        ld = lambda lg: (lg[..., a["aid"]] - lg[..., b["aid"]]).float()
        la, lb = ld(ca["logits"][0]).item(), ld(cb["logits"][0]).item()
        def kv(which):
            def h(kvs, i):
                if (i == 13 and which in ("13", "both")) or (i == 14 and which in ("14", "both")):
                    lt = "sliding_attention" if i == 13 else "full_attention"
                    return tuple(t.clone().copy_(ca["shared"][lt][j]) for j, t in enumerate(kvs))
            return h
        def put(vec):
            def h(o, i):
                o = o.clone(); o[:, -1] = vec; return o
            return h
        r = {}
        for name, hooks, ids, in_clean in (("KV13_all", {("kv_store", 13): kv("13"), ("kv_store", 14): kv("13")}, b["ids"], False),
                                           ("KV14_all", {("kv_store", 13): kv("14"), ("kv_store", 14): kv("14")}, b["ids"], False),
                                           ("ATTN_25", {("attn_out", 25): put(at["a"])}, b["ids"], False),
                                           ("NOISE_ATTN25", {("attn_out", 25): put(bt["b"])}, a["ids"], True)):
            lg = R.forward(torch.tensor([ids]), hooks=hooks, last_only_from=15)["logits"][0]
            r[name] = (ld(lg).item() - lb) / (la - lb)
        res.append({"pair": [a["subj"], b["subj"]], "ld_clean": la, "ld_corrupt": lb, **r})
    return res


@torch.no_grad()
def quantize(mode):
    for _, m in lin:
        W = m.weight.float()
        if mode == "int8":
            s = W.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
            q = torch.round(W / s).clamp(-127, 127).to(torch.int8)
            m._q, m._s = q, s; m.weight = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
            m.forward = (lambda mm: (lambda x: F.linear(x.float(), mm._q.float() * mm._s)))(m)
        elif mode == "int4":
            o, i = W.shape; G = W.view(o, i // 32, 32)
            s = G.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
            Wq = (torch.round(G / s).clamp(-7, 7) * s).view(o, i)
            m.weight.data.copy_(Wq.to(torch.bfloat16))  # quality-only arm: dequantised values held in bf16
        del W
    gc.collect()


res = {"info": info, "bf16_linear_bytes": bf16_bytes, "n_linear": len(lin)}
t0 = time.time()
gc.collect(); res["rss_bf16"] = rss()
res["BF16"] = {"factset": factset_pass(), "mech": mech_pass()}; json.dump(res, open(OUT, "w")); print("bf16 done", round(time.time() - t0), flush=True)
# order: INT8 (replaces weights by int8 storage) -> free -> reload bf16 -> INT4 (in place)
quantize("int8"); gc.collect(); res["rss_int8"] = rss()
res["int8_linear_bytes"] = sum(m._q.numel() + m._s.numel() * 4 for _, m in lin)
res["INT8"] = {"factset": factset_pass(), "mech": mech_pass()}; json.dump(res, open(OUT, "w")); print("int8 done", round(time.time() - t0), flush=True)
lin = None; del model, R; gc.collect()
res["rss_after_free"] = rss(); print("rss after free", res["rss_after_free"], flush=True)
if res["rss_after_free"] > 2.5:  # PL-16: never hold two models; skip INT4 rather than risk the box
    res["INT4"] = "skipped: previous model not freed"; json.dump(res, open(OUT, "w")); sys.exit(0)
model, tok, _ = load("it"); use_fp32_compute(model); R = Runner(model)
lin = [(n, m) for n, m in model.model.named_modules() if isinstance(m, torch.nn.Linear)]
quantize("int4"); res["int4_packed_linear_bytes_equiv"] = sum(m.weight.numel() // 2 + (m.weight.numel() // 32) * 2 for _, m in lin)
res["INT4"] = {"factset": factset_pass(), "mech": mech_pass()}; res["wall_s"] = round(time.time() - t0)
json.dump(res, open(OUT, "w")); print("done", round(time.time() - t0), flush=True)
