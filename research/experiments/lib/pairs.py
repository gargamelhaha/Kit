"""Shared item/pair selection for factual-recall patching experiments (E006 procedure). Imported, not copied (AUDIT-003)."""
import itertools
import torch
from gemma_loader import encode
from datasets import LANDMARKS, LANDMARK_TPL, CAPITALS, CAPITAL_TPL
REL = {"R1": (LANDMARKS, LANDMARK_TPL, "The "), "R2": (CAPITALS, CAPITAL_TPL, "The capital of ")}


def make_items(R, tok, rel):
    src, tpl, prefix = REL[rel]
    items = []
    for s, c in src:
        ids = encode(tok, tpl.format(s))[0]
        start = len(encode(tok, prefix.rstrip())[0])  # first subject token (after BOS + prefix words)
        last = len(encode(tok, prefix + s)[0]) - 1
        items.append({"rel": rel, "subj": s, "city": c, "ids": ids, "aid": tok(" " + c, add_special_tokens=False).input_ids[0],
                      "last_subj_pos": last, "subj_span": list(range(start, last + 1))})
    for it in items:
        it["correct"] = int(R.forward(torch.tensor([it["ids"]]), last_only_from=15)["logits"][0].argmax().item() == it["aid"])
    return items


def choose_pairs(items):
    ok = [i for i in items if i["correct"]]
    pairs = [(a, b) for a, b in itertools.permutations(ok, 2) if len(a["ids"]) == len(b["ids"]) and a["aid"] != b["aid"]
             and a["last_subj_pos"] == b["last_subj_pos"]]
    pairs = [pairs[j] for j in torch.randperm(len(pairs), generator=torch.Generator().manual_seed(0)).tolist()]
    seen, chosen = set(), []
    for a, b in pairs:
        k = frozenset((a["subj"], b["subj"]))
        if k not in seen:
            seen.add(k); chosen.append((a, b))
    return ok, chosen


def e006_plan(R, tok):
    iL, iC = make_items(R, tok, "R1"), make_items(R, tok, "R2")
    okL, chL = choose_pairs(iL); okC, chC = choose_pairs(iC)
    return {"held_out": [("R1", p) for p in chL[12:24]] + [("R2", p) for p in chC[:16]], "in_sample": [("R1", p) for p in chL[:12]],
            "ok": {"R1": okL, "R2": okC}}
