# HYPGEN-02 — second generation round (cycle 3), coordinator screening
Input: updated KB (K-0001…K-0010), register H-001…H-018, 7 target areas incl. an explicit demand for cost models on efficiency
proposals and an explicit VOI computation. Raw: `HYPGEN-02_raw.json` (14 proposals, 2 calls).

| proposal | decision |
|---|---|
| DS: decode cache storing only layers 0–14 K/V and running 15–34 at one position (−57% KV memory, ≥20% faster decode) | **rejected, physically void**: HF already stores K/V only for layers 0–14 (shared layers never write cache) and decode already runs one position. Same error class as HYPGEN-01 despite an explicit cost-model instruction |
| MM: row-sliced PLE access | rejected: already implemented and measured (PL-13 PreadEmbedding; ple_pread_bench.json) |
| DS + MM: IT vs base — late MLP/PLE pathway (blocks 26–34) implements the answer-form shift; cross-patching base→IT late outputs restores the factual token | **accepted → H-019** (mechanism behind H-017) |
| DS: answer-form-robust error decomposition (target top-1 at some residual 20–34 vs never) | accepted as an analysis variant for H-018 (zero model cost) |
| DS + MM: transfer principle — evidence written at the last layer able to alter shared state (KV-sharing source, MLA latent, SSM state) and extracted by a later reader | **accepted → H-020** (cross-architecture program; needs small public MLA/SSM/MoE models) |
| MM: intervention-fingerprint clustering of prompts | merged into H-014 (component 1 of the E006 matrix already separated multi-token R1 from R2 subjects) |
| DS: "shared-state intervention aliasing" (interventions equivalent iff same Δ in shared K/V + final query) | noted; follows from the architecture for layers ≥ 15 (K-0002) — a computational shortcut rather than a hypothesis |
| DS/MM: head-25.0 routing fidelity as a reliability signal; layer-13 rank-one edits for forgetting; PLE-gate explanation of why layer 25 | deferred (cost 12–48 h, or presupposes unverified PLE role H-006) |

## Agent utility
14 proposals → 2 registered, 1 analysis variant, 1 merged, 2 rejected as void/redundant (both efficiency proposals). Instruction
compliance: VOI field omitted by both models; MiniMax misspelled required keys in 6/7 items. Lesson (kept from cycle 2, now
twice confirmed): generator efficiency proposals need an automatic architectural sanity check before registration.
