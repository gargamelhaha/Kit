"""E005: root-cause test for the sandbox resets (PL-4). Hypothesis: calling the chunked unembedding outside
torch.no_grad() on a requires_grad tied embedding retains fp32 weight chunks for autograd (~1.6 GB for a
262144x1536 matrix), pushing the model process over the 8 GiB box. Safe: no model is loaded (0.8 GB tensor)."""
import json, resource, sys, os, gc
import torch
def rss_now():
    return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
W = torch.nn.Parameter(torch.randn(262144, 1536, dtype=torch.bfloat16) * 0.02)
x = torch.randn(1, 1536)
def unembed(x):
    return torch.cat([torch.nn.functional.linear(x, W[r:r + 32768].float()) for r in range(0, W.shape[0], 32768)], dim=-1)
res = {"base_gb": round(rss_now(), 3)}
with torch.no_grad():
    lg = unembed(x)
res["after_no_grad_call_live_logits_gb"] = round(rss_now(), 3); del lg; gc.collect()
lg = unembed(x)
res["after_grad_call_live_logits_gb"] = round(rss_now(), 3)
res["grad_fn"] = str(lg.grad_fn)
del lg; gc.collect()
res["after_del_gb"] = round(rss_now(), 3)
W.requires_grad_(False)
lg = unembed(x)
res["after_requires_grad_false_call_gb"] = round(rss_now(), 3); del lg
# Is RLIMIT_DATA enforced here? Try to allocate 2 GB under a 1.5 GB-above-current limit.
cur = rss_now()
resource.setrlimit(resource.RLIMIT_DATA, (int((cur + 1.0) * 1e9), resource.RLIM_INFINITY))
try:
    big = torch.empty(int(2e9) // 4, dtype=torch.float32); big.fill_(1.0)
    res["rlimit_data_enforced"] = False
except (MemoryError, RuntimeError) as e:
    res["rlimit_data_enforced"] = True; res["rlimit_error"] = str(e)[:120]
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(__file__), "result.json"), "w"), indent=1)
