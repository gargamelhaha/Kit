"""E009: per-item internal signals, lens distributions for DoLa-style contrast and early exit (see PREREG.md)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
V = os.environ.get("VARIANT", "it")
D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, f"result_{V}.jsonl")
model, tok, info = load(V); use_fp32_compute(model); R = Runner(model)
items = [json.loads(l) for l in open(os.path.join(D, "..", "..", "benchmarks", "factset_v1.jsonl"))]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT)}
f = open(OUT, "a"); t0 = time.time()
for n, it in enumerate(items):
    if it["id"] in done: continue
    ids = encode(tok, it["prompt"])[0]; tgt = tok(" " + it["answer"], add_special_tokens=False).input_ids[0]
    ent = {}
    def cap(p, i): ent[i] = p[0, 0, -1].clone()  # head 0, final query
    o = R.forward(torch.tensor([ids]), hooks={("attn_probs", 25): cap}, last_only_from=15, keep_resid=True, attn="eager", logits=None)  # final logits = lens row 35
    H = torch.stack([r[0, -1] for r in o["resid"]])            # [36, d]
    lp = torch.log_softmax(R.unembed(H).float(), -1)            # [36, V]
    fin = lp[-1]; top1 = int(fin.argmax())
    tops = lp.argmax(-1).tolist(); pmax = lp.max(-1).values.exp().tolist()
    plaus = (fin.exp() >= 0.1 * fin.exp().max()).nonzero().flatten()
    pf = fin.exp()
    jsd = []
    for k in range(36):
        pk = lp[k].exp(); m = 0.5 * (pf + pk)
        jsd.append(float(0.5 * (pf * (fin - m.log())).sum() + 0.5 * (pk * (lp[k] - m.log())).sum()))
    a = ent.get(25)
    rec = {"id": it["id"], "relation": it["relation"], "difficulty": it["difficulty"], "T": len(ids), "target": tgt,
           "target_str": it["answer"], "top1": top1, "top1_str": tok.decode(top1), "correct": top1 == tgt,
           "final_maxprob": pmax[-1], "lens_top1": tops, "lens_maxprob": [round(x, 5) for x in pmax],
           "target_rank_final": int((fin > fin[tgt]).sum()), "jsd": [round(x, 6) for x in jsd],
           "plausible": plaus.tolist()[:50], "plausible_lp": lp[:, plaus[:50]].tolist(), "target_lp": lp[:, tgt].tolist(),
           "attn25h0_entropy": float(-(a * (a + 1e-12).log()).sum()) if a is not None else None,
           "margin": float(fin.topk(2).values[0] - fin.topk(2).values[1])}
    f.write(json.dumps(rec) + "\n"); f.flush()
    if n % 25 == 0: print(n, "/", len(items), round(time.time() - t0), "s", flush=True)
print("done", flush=True)
