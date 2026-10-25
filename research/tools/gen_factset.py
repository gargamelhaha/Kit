"""Build a factual-recall item set with natural model errors (for E009: calibration / error detection / decoding-time fixes).

Two model families each generate completion-style items; each item is then verified by the *other* family; an item is kept
only if both the generator and the cross-verifier agree that the answer is correct and unambiguous. Output:
research/benchmarks/factset_v1.jsonl  {id, relation, prompt, answer, difficulty, generator, verifier}
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__)); from llm_pool import Pool
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RELATIONS = ["country -> capital city", "chemical element -> chemical symbol (answer is the symbol)", "famous book -> author surname",
             "painting or sculpture -> artist surname", "company -> country of headquarters", "city -> country",
             "scientist -> field (one word, e.g. physics)", "planet or moon -> planet it orbits / or star system object", "language -> country where it is most spoken",
             "musical instrument -> instrument family (e.g. string)", "animal -> class (mammal, bird, reptile, fish, amphibian, insect)",
             "currency -> country", "river -> continent", "sport -> number of players per team (as a word)", "inventor -> invention (one word)",
             "US state -> capital city", "mountain -> continent", "composer -> nationality adjective", "dish -> country of origin", "university -> city"]
GEN = """Write 24 factual completion items for the relation "{rel}" for testing a small language model.
Mix difficulty: 8 easy (well known), 8 medium, 8 hard (obscure but unambiguous and verifiable).
Each item is an English sentence prefix whose correct continuation is ONE common English word (the answer), e.g.
{{"prompt": "The capital of Australia is", "answer": "Canberra", "difficulty": "easy"}}.
Rules: the answer must be unique and uncontroversial; no dates or numbers written as digits; the prefix must not contain the answer;
prefix must end right before the answer (no trailing space). Return ONLY a JSON list."""
VER = """For each numbered item decide whether the answer is the correct, unique, uncontroversial single-word continuation.
Return ONLY a JSON list of objects {{"i": <number>, "ok": true|false, "note": "<=10 words"}}.
{items}"""


def js(s, open_="[", close="]"):
    m = re.search(re.escape(open_) + r".*" + re.escape(close), s or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def main():
    pool = Pool(tag="factset")
    reqs = [{"model": m, "max_tokens": 16000, "reasoning_effort": "low", "meta": {"rel": r},
             "messages": [{"role": "user", "content": GEN.format(rel=r)}]} for r in RELATIONS for m in ("deepseek", "minimax")]
    outs = pool.map(reqs)
    items = []
    for rq, o in zip(reqs, outs):
        for it in js(o.get("content")) or []:
            if isinstance(it, dict) and it.get("prompt") and it.get("answer") and " " not in it["answer"].strip():
                items.append({"relation": rq["meta"]["rel"], "prompt": it["prompt"].rstrip(), "answer": it["answer"].strip(),
                              "difficulty": it.get("difficulty", "?"), "generator": rq["model"]})
    other = {"deepseek": "minimax", "minimax": "deepseek", "free-deepseek-4.1-flash": "minimax", "minimax-m3.1-flash": "deepseek"}
    batches = {}
    for k, it in enumerate(items):
        batches.setdefault((other[it["generator"]], it["relation"]), []).append(k)
    vreqs, keys = [], []
    for (m, rel), ks in batches.items():
        txt = "\n".join(f"{n}. {items[k]['prompt']} ___ -> {items[k]['answer']}" for n, k in enumerate(ks))
        vreqs.append({"model": m, "max_tokens": 12000, "reasoning_effort": "low", "meta": {"verify": rel},
                      "messages": [{"role": "user", "content": VER.format(items=txt)}]}); keys.append(ks)
    for ks, o, rq in zip(keys, pool.map(vreqs), vreqs):
        for v in js(o.get("content")) or []:
            if isinstance(v, dict) and isinstance(v.get("i"), int) and 0 <= v["i"] < len(ks):
                items[ks[v["i"]]]["verified"] = bool(v.get("ok")); items[ks[v["i"]]]["verifier"] = rq["model"]
    seen, kept = set(), []
    for it in items:
        key = it["prompt"].lower()
        if it.get("verified") and key not in seen:
            seen.add(key); kept.append(it)
    os.makedirs(os.path.join(ROOT, "benchmarks"), exist_ok=True)
    with open(os.path.join(ROOT, "benchmarks", "factset_v1.jsonl"), "w") as f:
        for n, it in enumerate(kept):
            f.write(json.dumps({"id": f"F{n:04d}", **it}, ensure_ascii=False) + "\n")
    stats = {"generated": len(items), "verified_true": sum(1 for i in items if i.get("verified")),
             "verified_false": sum(1 for i in items if i.get("verified") is False),
             "unverified": sum(1 for i in items if "verified" not in i), "kept_unique": len(kept)}
    json.dump(stats, open(os.path.join(ROOT, "benchmarks", "factset_v1_stats.json"), "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
