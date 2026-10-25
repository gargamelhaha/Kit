"""E007b (exploratory, post-hoc after the T=2048 FULL arm hit the 6.8 GB RSS guard): peak-RSS per arm, SKIP at T=2048."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_b_it.json")
import threading
def cur_rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 2**20
class Peak:  # gVisor exposes no VmHWM -> sample VmRSS every 5 ms
    def __enter__(self):
        self.peak, self.run = cur_rss(), True
        def loop():
            while self.run: self.peak = max(self.peak, cur_rss()); time.sleep(0.005)
        self.t = threading.Thread(target=loop, daemon=True); self.t.start(); return self
    def __exit__(self, *a): self.run = False; self.t.join()
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
text = "".join(open(os.path.join(D, "..", "..", p)).read() for p in ("LITERATURE_MAP.md", "meta_analysis/methods_table.md"))
base = encode(tok, text)[0]
res = {"rows": [], "peak_method": "VmRSS sampled every 5 ms (no VmHWM under gVisor)"}
for T, arm in [(256, "FULL"), (256, "SKIP"), (1024, "FULL"), (1024, "SKIP"), (2048, "SKIP")]:
    ids = torch.tensor([base[:T]]); kw = {} if arm == "FULL" else {"last_only_from": 15, "allow_long": True}
    before = cur_rss(); t0 = time.perf_counter()
    with Peak() as pk:
        R.forward(ids, logits="last", **kw)
    dt = time.perf_counter() - t0
    row = {"T": T, "arm": arm, "wall_s": round(dt, 2), "rss_before_gb": round(before, 3), "peak_gb": round(pk.peak, 3),
           "peak_minus_before_gb": round(pk.peak - before, 3)}
    res["rows"].append(row); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(row), flush=True)
print("done", flush=True)
