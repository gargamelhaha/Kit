"""E000: load check, tokenizer/BOS check, generation, bf16 vs fp32-compute agreement and speed."""
import json, os, platform, resource, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch, transformers
from gemma_loader import load, use_fp32_compute, encode
V = os.environ.get("VARIANT", "it"); torch.manual_seed(0)
rss = lambda: round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
out = {"env": {"torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version(),
               "threads": torch.get_num_threads(), "cpu_flags_bf16": "avx512_bf16" in open("/proc/cpuinfo").read()}}
model, tok, info = load(V); out["load"] = info; out["peak_rss_gb_after_load"] = rss()
s = "The Eiffel Tower is in Paris. Привет, мир! 12345"
raw = tok(s).input_ids
out["tokenizer"] = {"default_adds_bos": raw[0] == tok.bos_token_id, "roundtrip_ok": tok.decode(raw, skip_special_tokens=True) == s}
prompts = ["The capital of France is", "Two plus two equals", "def fibonacci(n):\n    if n <",
           "The mitochondrion is an organelle found in the cells of most eukaryotes. It generates most of the chemical energy needed to power the cell's biochemical reactions."]
def probe(tag):
    r = {}
    with torch.no_grad():
        for p in prompts:
            for bos in (False, True):
                ids = torch.tensor(encode(tok, p, add_bos=bos))
                o = model(input_ids=ids, labels=ids)
                r[f"{p[:24]}|bos={bos}"] = {"ppl": round(torch.exp(o.loss.float()).item(), 2),
                    "top3": [tok.decode(i) for i in o.logits[0, -1].topk(3).indices], "_last": o.logits[0, -1].float()}
        ids = torch.randint(1000, 200000, (16, 32)); ids[:, 0] = tok.bos_token_id
        model(input_ids=ids[:2, :8]); t0 = time.time(); model(input_ids=ids, logits_to_keep=1)
    return r, round(time.time() - t0, 2)
with torch.no_grad():
    if V == "it":
        enc = tok.apply_chat_template([{"role": "user", "content": "In one sentence: what is a sparse autoencoder?"}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
    else:
        enc = {"input_ids": torch.tensor(encode(tok, "Q: What is the capital of Japan?\nA:"))}
    t0 = time.time(); g = model.generate(**enc, max_new_tokens=32, do_sample=False); dt = time.time() - t0
    new = g[0, enc["input_ids"].shape[1]:]
    out["generate_bf16"] = {"new_tokens": len(new), "tok_per_s": round(len(new) / dt, 2), "text": tok.decode(new)}
bf, t_bf = probe("bf16"); use_fp32_compute(model); fp, t_fp = probe("fp32")
out["batch16x32_forward_s"] = {"bf16": t_bf, "fp32_compute": t_fp}
out["probes"] = {}
for k in bf:
    a, b = bf[k].pop("_last"), fp[k].pop("_last")
    out["probes"][k] = {"bf16": bf[k], "fp32": fp[k], "max_abs_logit_diff": round((a - b).abs().max().item(), 3), "top1_agree": int(a.argmax() == b.argmax())}
out["peak_rss_gb"] = rss()
json.dump(out, open(os.path.join(os.path.dirname(__file__), f"result_{V}.json"), "w"), ensure_ascii=False, indent=1)
print("done")
