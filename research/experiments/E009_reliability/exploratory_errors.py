"""Exploratory (post hoc, not preregistered): error taxonomy and 'answer present in intermediate layers but lost at the output'."""
import json, os, collections
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
V = os.environ.get("VARIANT", "it")
recs = [json.loads(l) for l in open(os.path.join(D, f"result_{V}.jsonl"))]
FUNC = {"the", "a", "an", "located", "in", "of", "known", "called", "one", "not", "also", "that", "to", "at", "on", "by", "is", "was", "and", "from"}
def cat(r):
    s = r["top1_str"].strip().lower()
    if not s or not any(ch.isalnum() for ch in s): return "punct/whitespace"
    if s in FUNC: return "function word"
    if r["target_rank_final"] <= 2: return "content, target rank<=2"
    return "content, other"
err = [r for r in recs if not r["correct"]]
tax = collections.Counter(cat(r) for r in err)
lost = [r for r in err if any(t == r["target"] for t in r["lens_top1"][20:35])]
lost_content = [r for r in lost if cat(r) in ("function word", "punct/whitespace")]
peak = collections.Counter(max(range(20, 35), key=lambda k: r["target_lp"][k]) for r in lost)
# oracle: best single residual index (chosen on DEV parity) for answering
order = {json.loads(l)["id"]: n for n, l in enumerate(open(os.path.join(D, "..", "..", "benchmarks", "factset_v1.jsonl")))}
dev = [r for r in recs if order[r["id"]] % 2 == 0]; test = [r for r in recs if order[r["id"]] % 2 == 1]
acc_k = {k: float(np.mean([r["lens_top1"][k] == r["target"] for r in dev])) for k in range(20, 36)}
kbest = max(acc_k, key=acc_k.get)
# content-restricted final answer: skip function words / punctuation at the final layer (pick best non-function token among plausible)
MD = {"**", "*", "#", "##"}
md_err = sum(1 for r in err if r["top1_str"].strip() in MD)
out = {"markdown_top1_errors": md_err, "markdown_share_of_errors": md_err / max(1, len(err)), "n": len(recs), "n_errors": len(err), "error_taxonomy": dict(tax),
       "errors_with_target_top1_at_some_layer_20_34": len(lost), "of_which_final_is_function_or_punct": len(lost_content),
       "layer_where_target_peaks_in_lost": dict(sorted(peak.items())),
       "dev_acc_by_residual_index": {k: round(v, 3) for k, v in acc_k.items()}, "best_index_on_dev": kbest,
       "test_acc_final": float(np.mean([r["correct"] for r in test])), "test_acc_best_index": float(np.mean([r["lens_top1"][kbest] == r["target"] for r in test])),
       "test_acc_index33": float(np.mean([r["lens_top1"][33] == r["target"] for r in test])),
       "examples_lost": [(r["id"], r["target_str"], r["top1_str"], [k for k in range(20, 35) if r["lens_top1"][k] == r["target"]][:6]) for r in lost[:12]],
       "acc_by_relation": {rel: round(float(np.mean([r["correct"] for r in recs if r["relation"] == rel])), 2) for rel in sorted({r["relation"] for r in recs})}}
json.dump(out, open(os.path.join(D, f"exploratory_errors_{V}.json"), "w"), indent=1)
print(json.dumps(out, indent=1)[:4500])
