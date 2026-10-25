import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
import torch
from gemma_loader import load, use_fp32_compute, encode
from runner import Runner
from datasets import LANDMARKS, CAPITALS, LANDMARK_TPL, CAPITAL_TPL
V = os.environ.get("VARIANT", "it")
items = [(LANDMARK_TPL.format(s), c, "landmark") for s, c in LANDMARKS] + [(CAPITAL_TPL.format(s), c, "capital") for s, c in CAPITALS]
model, tok, info = load(V); use_fp32_compute(model); R = Runner(model)
out = {"info": info, "items": []}; t0 = time.time()
for prompt, city, kind in items:
    aid = tok(" " + city, add_special_tokens=False).input_ids[0]
    r = R.forward(torch.tensor(encode(tok, prompt)), last_only_from=15, keep_resid=True)
    ranks, probs, top1 = [], [], []
    for h in r["resid"]:
        lg = R.unembed(h[:, -1])[0]
        ranks.append(int((lg > lg[aid]).sum().item()) + 1); probs.append(torch.softmax(lg, -1)[aid].item()); top1.append(tok.decode(lg.argmax()))
    out["items"].append({"prompt": prompt, "answer": city, "kind": kind, "correct": int(ranks[-1] == 1), "ranks": ranks, "probs": probs, "top1": top1})
    json.dump(out, open(os.path.join(os.path.dirname(__file__), f"result_{V}.json"), "w"), indent=1, ensure_ascii=False)
    print(len(out["items"]), round(time.time() - t0), flush=True)
out["wall_s"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(os.path.dirname(__file__), f"result_{V}.json"), "w"), indent=1, ensure_ascii=False)
print("done")
