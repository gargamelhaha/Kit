"""E007c: prefill skip in real generation modes + bf16-decode arm (see PREREG.md)."""
import json, os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from fastpaths import enable_prefill_skip, disable_prefill_skip, set_small_batch_bf16
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_it.json")
model, tok, info = load("it"); use_fp32_compute(model); set_small_batch_bf16(model, 0)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
text = open(os.path.join(D, "..", "..", "LITERATURE_MAP.md")).read() + open(os.path.join(D, "..", "..", "meta_analysis", "methods_table.md")).read()
base = encode(tok, text)[0]


def rss():
    for l in open("/proc/self/status"):
        if l.startswith("VmRSS"): return int(l.split()[1]) / 2**20


calls = []
orig_forward = model.forward
def timed_forward(*a, **k):
    t0 = time.perf_counter(); o = orig_forward(*a, **k); calls.append(time.perf_counter() - t0); return o
model.forward = timed_forward


def gen(ids, mask, G, past=None):
    calls.clear(); peak = [rss()]; run = [True]
    def mon():
        while run[0]: peak[0] = max(peak[0], rss()); time.sleep(0.01)
    th = threading.Thread(target=mon, daemon=True); th.start()
    t0 = time.perf_counter()
    o = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=G, min_new_tokens=G, do_sample=False,
                       output_scores=True, return_dict_in_generate=True, pad_token_id=PAD, past_key_values=past, use_cache=True)
    wall = time.perf_counter() - t0; run[0] = False; th.join()
    return {"seq": o.sequences[:, ids.shape[1]:], "scores": torch.stack(o.scores, 1).float(), "wall": wall,
            "ttft": calls[0] if calls else None, "decode_per_tok": (sum(calls[1:]) / max(1, len(calls) - 1)) if calls else None,
            "peak_rss": peak[0], "past": o.past_key_values, "full_seq": o.sequences}


def set_arm(arm):
    disable_prefill_skip(model); set_small_batch_bf16(model, 0)
    if arm in ("SKIP", "SKIP+BF16"): enable_prefill_skip(model)
    if arm == "SKIP+BF16": set_small_batch_bf16(model, 4)


def compare(ref, x):
    rows = []
    for r in range(ref["seq"].shape[0]):
        same = (ref["seq"][r] == x["seq"][r]).tolist()
        div = same.index(False) if False in same else len(same)
        n = max(1, div)
        rows.append({"identical": div == len(same), "first_div": div, "frac_identical_tokens": sum(same) / len(same),
                     "max_dscore_before_div": float((ref["scores"][r, :n] - x["scores"][r, :n]).abs().max())})
    return rows


def chat_ids(msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = ids["input_ids"] if isinstance(ids, dict) else ids
    if ids[0] != tok.bos_token_id: ids = [tok.bos_token_id] + ids
    return ids


conds = {}
conds["C1"] = [(torch.tensor([base[:128]]), None, 32)]
conds["C2"] = [(torch.tensor([base[:1024]]), None, 24)]
lens = [90, 160, 230, 300]; starts = [2000, 4000, 6000, 8000]
seqs = [[tok.bos_token_id] + base[s:s + n - 1] for s, n in zip(starts, lens)]
M = max(lens); ids3 = torch.tensor([[PAD] * (M - len(s)) + s for s in seqs]); mask3 = (torch.arange(M)[None] >= (M - torch.tensor(lens))[:, None]).long()
conds["C3"] = [(ids3, mask3, 24)]
QS = ["What is the capital of Australia? Answer in one sentence.", "Explain in two sentences what a transformer attention head does.",
      "Name three noble gases."]
conds["C4"] = [(torch.tensor([chat_ids([{"role": "user", "content": q}])]), None, 48) for q in QS]
res = {"info": info, "rows": []}
for cname, items in conds.items():
    for k, (ids, mask, G) in enumerate(items):
        outs = {}
        for arm in ("REF", "SKIP", "SKIP+BF16"):
            set_arm(arm); outs[arm] = gen(ids, mask, G)
            row = {"cond": cname, "item": k, "arm": arm, "T": ids.shape[1], "batch": ids.shape[0], "G": G,
                   "wall_s": round(outs[arm]["wall"], 2), "ttft_s": round(outs[arm]["ttft"], 3),
                   "decode_s_per_step": round(outs[arm]["decode_per_tok"], 3), "peak_rss_gb": round(outs[arm]["peak_rss"], 3),
                   "text": tok.decode(outs[arm]["seq"][0])[:200]}
            if arm != "REF": row["vs_ref"] = compare(outs["REF"], outs[arm])
            res["rows"].append(row); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({k2: v for k2, v in row.items() if k2 != "text"}), flush=True)
        if cname == "C4":  # C5: continue the conversation with the cached state of each arm
            follow = tok("<end_of_turn>\n<start_of_turn>user\nNow say it again in five words.<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False).input_ids
            o5 = {}
            for arm in ("REF", "SKIP", "SKIP+BF16"):
                set_arm(arm); prev = outs[arm]
                ids5 = torch.cat([prev["full_seq"], torch.tensor([follow])], 1)
                o5[arm] = gen(ids5, None, 24, past=prev["past"])
                row = {"cond": "C5", "item": k, "arm": arm, "T": ids5.shape[1], "batch": 1, "G": 24, "wall_s": round(o5[arm]["wall"], 2),
                       "ttft_s": round(o5[arm]["ttft"], 3), "decode_s_per_step": round(o5[arm]["decode_per_tok"], 3),
                       "peak_rss_gb": round(o5[arm]["peak_rss"], 3), "text": tok.decode(o5[arm]["seq"][0])[:200],
                       "note": "each arm continues its own cache; for SKIP vs REF identity implies the cache is exact"}
                if arm != "REF": row["vs_ref"] = compare(o5["REF"], o5[arm])
                res["rows"].append(row); json.dump(res, open(OUT, "w"), indent=1); print(json.dumps({k2: v for k2, v in row.items() if k2 != "text"}), flush=True)
print("done", flush=True)
