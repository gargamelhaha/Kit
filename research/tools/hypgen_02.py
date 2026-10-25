import json, os, sys, re
sys.path.insert(0, "tools"); from llm_pool import Pool
ctx = open("models/MODELS.md").read().split("## Host constraints")[0]
kb = open("SCIENTIFIC_KNOWLEDGE.md").read()[:30000]
hyp = open("HYPOTHESIS_REGISTER.md").read(); oq = open("OPEN_QUESTIONS.md").read()
tools = """Tooling: Runner hooks resid_pre/resid_post/attn_out/mlp_out/ple_out/head_in (per-head o_proj input)/attn_probs/kv_store
(replace the K/V that layers 13/14 store for layers 15-34); last_only_from=15 (exact). Batched patching ~0.7 s per condition
for 12-30 token prompts on CPU (2 vCPU, 8 GiB, fp32 compute). Both it and base checkpoints available. No GPU, no SAE
training at scale, no fine-tuning of the full model (memory)."""
areas_old = ["mechanisms/circuits (attention heads, MLP, residual, KV-sharing)", "knowledge storage & retrieval (facts vs context vs association)",
         "generalization vs memorization / transfer", "computational efficiency without quality loss (prefill, decode, KV cache, precision)",
         "knowledge editing & interference", "confidence / calibration / error detection from internals",
         "radically new principle for discovering regularities beyond hypothesis enumeration"]
areas = ["mechanism: why one layer (25) and one head (0) among 20 KV-shared layers; what computes the late-attention-independent half",
         "IT vs base: late-block answer-form shift (H-017) and what instruction tuning changed in the circuit",
         "efficiency without quality loss: prefill/decode/memory on CPU for this architecture (each proposal MUST include a cost model showing the saving is physically possible, e.g. decode already processes one position)",
         "reliability: knowing when the model is wrong, with metrics robust to answer-form artefacts",
         "transfer of mechanisms to other architectures (MoE, SSM/Mamba, MLA, sliding-window/KV-sharing models) testable on CPU with small public models",
         "knowledge editing/forgetting localized to the layer-13 K/V subject representation",
         "a radically new discovery principle (not enumerating hypotheses) that can be tested on existing data first"]
P = f"""You are the Hypothesis Generator of an autonomous interpretability lab. Using ONLY the context below, propose
7 new falsifiable hypotheses about Gemma 4 E2B, one per area: {areas}.
For each give JSON fields: area, statement, why_plausible (cite K-/H- ids from context), competing_explanations (>=2),
discriminating_experiment (concrete, feasible with the tooling), numeric_prediction (what result supports vs refutes),
cost_hours_cpu, value (1-5: how much the answer changes our understanding), novelty_risk (what prior work might already show it).
Do not repeat registered hypotheses H-001..H-018. Return ONLY a JSON list.

{tools}

ARCHITECTURE:
{ctx}

VERIFIED KNOWLEDGE:
{kb}

HYPOTHESES ALREADY REGISTERED:
{hyp}

OPEN QUESTIONS:
{oq}"""
outs = Pool(tag="hypgen").map([{"model": m, "max_tokens": 40000, "reasoning_effort": "medium", "meta": {"task": "hypgen"}, "messages": [{"role": "user", "content": P}]} for m in ("deepseek", "minimax")])
rep = {}
for m, o in zip(("deepseek", "minimax"), outs):
    mm = re.search(r"\[.*\]", o.get("content", ""), re.S)
    try: rep[m] = json.loads(mm.group(0))
    except Exception: rep[m] = {"raw": o.get("content", ""), "error": o.get("error")}
    print(m, o.get("usage"), type(rep[m]).__name__, len(rep[m]) if isinstance(rep[m], list) else "")
json.dump(rep, open("agents/reports/HYPGEN-02_raw.json", "w"), indent=1, ensure_ascii=False)
