"""E007c: prefill skip in real generation modes + bf16-decode arm (see PREREG.md)."""
import json, os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from fastpaths import enable_prefill_skip, disable_prefill_skip, set_small_batch_bf16
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "result_it.json")
model, tok, info = load("it"); use_fp32_compute(model); set_small_batch_bf16(model, 0)
assert all(m.bias is None for m in model.modules() if isinstance(m, torch.nn.Linear)), "bf16 path assumes bias-free linears"
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


def cache_diff(p1, p2):
    """max |diff| over all cached K/V tensors (layers 0-14 write cache; verifies the skip leaves the cache unchanged)."""
    try:
        m = 0.0
        for l1, l2 in zip(p1.layers, p2.layers):
            for a_, b_ in ((getattr(l1, "keys", None), getattr(l2, "keys", None)), (getattr(l1, "values", None), getattr(l2, "values", None))):
                if a_ is not None and b_ is not None and a_.numel():
                    m = max(m, float((a_.float() - b_.float()).abs().max()))
        return m
    except Exception as e:
        return f"n/a: {e}"


def compare(ref, x):
    rows = []
    for r in range(ref["seq"].shape[0]):
        same = (ref["seq"][r] == x["seq"][r]).tolist()
        div = same.index(False) if False in same else len(same)
        n = max(1, div)
        rows.append({"identical": div == len(same), "first_div": div, "frac_identical_tokens_all": sum(same) / len(same),
                     "max_dscore_before_div": float((ref["scores"][r, :n] - x["scores"][r, :n]).abs().max())})
    return rows


def chat_ids(msgs):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = ids["input_ids"] if isinstance(ids, dict) else ids
    if ids[0] != tok.bos_token_id: ids = [tok.bos_token_id] + ids
    return ids


conds = {}
conds["C1"] = [(torch.tensor([[tok.bos_token_id] + base[s0:s0 + 127]]), None, 32) for s0 in (1, 3000, 9000)]
conds["C2"] = [(torch.tensor([base[:1024]]), None, 24)]
lens = [90, 160, 230, 300]; starts = [2000, 4000, 6000, 8000]
seqs = [[tok.bos_token_id] + base[s:s + n - 1] for s, n in zip(starts, lens)]
M = max(lens); ids3 = torch.tensor([[PAD] * (M - len(s)) + s for s in seqs]); mask3 = (torch.arange(M)[None] >= (M - torch.tensor(lens))[:, None]).long()
conds["C3"] = [(ids3, mask3, 24)]
conds["C3b"] = [(torch.tensor([[tok.bos_token_id] + base[s0:s0 + 199] for s0 in (2000, 4000, 6000, 8000)]), None, 24)]  # equal length, no padding
QS = ["What is the capital of Australia? Answer in one sentence.", "Explain in two sentences what a transformer attention head does.",
      "Name three noble gases."]
conds["C4"] = [(torch.tensor([chat_ids([{"role": "user", "content": q}])]), None, 48) for q in QS]
res = {"info": info, "rows": [], "changes": "after PREAUDIT-E007c: warm-up, 2 interleaved repeats, BF16 vs SKIP, cache diff, immutable C5 context"}
ARMS = ("REF", "SKIP", "SKIP+BF16")
set_arm("SKIP"); gen(torch.tensor([base[:40]]), None, 2); set_arm("REF"); gen(torch.tensor([base[:40]]), None, 2)  # warm-up
def summarize(cname, k, ids, G, outs, times, extra=None):
    for arm in ARMS:
        o = outs[arm]
        row = {"cond": cname, "item": k, "arm": arm, "T": ids.shape[1], "batch": ids.shape[0], "G": G,
               "wall_s": [round(t["wall"], 2) for t in times[arm]], "prefill_s": [round(t["ttft"], 3) for t in times[arm]],
               "decode_s_per_step": [round(t["decode_per_tok"], 3) for t in times[arm]], "peak_rss_gb": round(o["peak_rss"], 3),
               "text": tok.decode(o["seq"][0])[:200]}
        if arm == "SKIP": row["vs_ref"] = compare(outs["REF"], o); row["cache_maxdiff_vs_ref"] = cache_diff(outs["REF"]["past"], o["past"])
        if arm == "SKIP+BF16": row["vs_skip"] = compare(outs["SKIP"], o); row["vs_ref"] = compare(outs["REF"], o)
        if extra: row.update(extra)
        res["rows"].append(row); json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps({k2: v for k2, v in row.items() if k2 != "text"}), flush=True)
for cname, items in conds.items():
    for k, (ids, mask, G) in enumerate(items):
        outs, times = {}, {a: [] for a in ARMS}
        for order in (ARMS, ARMS[::-1]):
            for arm in order:
                set_arm(arm); o = gen(ids, mask, G); times[arm].append(o)
                if arm not in outs: outs[arm] = o
        summarize(cname, k, ids, G, outs, times)
        if cname == "C4":  # C5: identical immutable context (prompt + REF answer + follow-up) for every arm
            follow = tok("<end_of_turn>\n<start_of_turn>user\nNow say it again in five words.<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False).input_ids
            ctx = outs["REF"]["full_seq"][:, :-1]  # last generated token is not yet in any cache
            ids5 = torch.cat([outs["REF"]["full_seq"], torch.tensor([follow])], 1)
            o5, t5 = {}, {a: [] for a in ARMS}
            for arm in ARMS:
                set_arm(arm)
                with torch.no_grad():
                    past = model(input_ids=ctx, use_cache=True).past_key_values  # chunk 1 (prefill)
                o = gen(ids5, None, 24, past=past); o5[arm] = o; t5[arm].append(o)    # chunk 2 (continuation) + decode
            set_arm("REF"); full = gen(ids5, None, 24)                                 # control: full re-prefill, no cache reuse
            summarize("C5", k, ids5, 24, o5, t5, {"ref_cached_vs_full_reprefill": compare(full, o5["REF"])})
print("done", flush=True)
