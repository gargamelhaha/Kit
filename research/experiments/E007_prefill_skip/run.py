"""E007: prefill time FULL vs SKIP (KV-shared layers at final position only). See PREREG.md."""
import json, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_it.json")
model, tok, info = load("it"); use_fp32_compute(model); R = Runner(model)
text = "".join(open(os.path.join(D, "..", "..", p)).read() for p in ("LITERATURE_MAP.md", "meta_analysis/methods_table.md"))
base = encode(tok, text)[0]
res = {"info": info, "torch_threads": torch.get_num_threads(), "n_text_tokens": len(base), "rows": []}
print("text tokens", len(base), flush=True)
for T in (64, 256, 512, 1024, 2048):
    ids = torch.tensor([base[:T]])
    run = {"FULL": lambda: R.forward(ids, logits="last"),
           "SKIP": lambda: R.forward(ids, logits="last", last_only_from=15, allow_long=True)}
    outs = {k: f()["logits"][0] for k, f in run.items()}  # warm-up + equivalence
    times = {"FULL": [], "SKIP": []}
    for _ in range(3):
        for k in ("FULL", "SKIP"):
            t0 = time.perf_counter(); run[k](); times[k].append(time.perf_counter() - t0)
    a, b = outs["FULL"], outs["SKIP"]
    row = {"T": T, "full_s": [round(x, 3) for x in times["FULL"]], "skip_s": [round(x, 3) for x in times["SKIP"]],
           "speedup_median": round(statistics.median(times["FULL"]) / statistics.median(times["SKIP"]), 3),
           "max_abs_dlogit": float((a - b).abs().max()), "top1_same": bool(a.argmax() == b.argmax()),
           "top10_overlap": len(set(a.topk(10).indices.tolist()) & set(b.topk(10).indices.tolist()))}
    res["rows"].append(row); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps(row), flush=True)
print("done", flush=True)
