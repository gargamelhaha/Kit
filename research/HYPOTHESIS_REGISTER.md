# Hypothesis register
Preliminary hypotheses, ideas and open interpretations. Not verified knowledge. Machine-readable twin: `hypotheses.jsonl`.

| id | statement (short) | status | test |
|---|---|---|---|
| H-001 | In Gemma 4 E2B all cross-position information that reaches the final position is carried by layer-13/14 K/V; layers 15–34 act per-position | EXPERIMENTALLY_SUPPORTED (deterministic consequence of the code, confirmed by E001) | E001 |
| H-002 | Factual recall in E2B shows the ROME two-site pattern (last subject token mid-layers; final token late layers) | INCONCLUSIVE — late site 12/12; early site P1 failed (8/12); control not a zero null (AUDIT-001) | E004 → E004b |
| H-003 | Because subject-token states are only visible to later positions through layer ≤14 K/V, subject "enrichment" relevant for recall must complete by layer 14 (≤40% depth) | INCONCLUSIVE — superseded by H-008 (E006: the relevant K/V is layer 13's; enrichment itself not measured) | E004 P3, E006 |
| H-004 | E2B contains induction heads that are causally necessary for repeated-sequence copying | CANDIDATE | E003 (preregistered) |
| H-005 | Logit-lens answer emergence occurs in the last third of layers in E2B (as in other LMs) | EXPERIMENTALLY_SUPPORTED → K-0005 | E002 |
| H-006 | PLE token-identity injection lets late layers re-read token identity, so residual-only interventions under-estimate token effects | CANDIDATE | ablate/patch PLE inputs vs residual (planned) |
| H-007 | The 60 k/v tensors stored for KV-shared layers are untrained remnants (absmax ≈ 0.06 vs 0.23–0.35 in trained k_proj) | CANDIDATE | compare with init distribution; check base checkpoint |
| H-008 | Subject info reaches the final position via layer-13 sliding K/V, not layer-14 full K/V | EXPERIMENTALLY_SUPPORTED → K-0006 (E006: 0.986 vs 0.017, 28 held-out pairs) | E006 |
| H-009 | Subject→final transfer happens in block-25 attention | EXPERIMENTALLY_SUPPORTED → K-0007 (sufficient 0.757, layer-specific; **not necessary**: mean ablation retains 0.84, E006b) | E006, E006b |
| H-010 | Head 25.0 is the dominant extraction head | EXPERIMENTALLY_SUPPORTED (E006b: subject attention 0.88 vs 0.17 uniform; only head whose corruption matters, 0.37 vs ≥0.98); not in KB (no head mean-ablation, no audit) | E006, E006b |
| H-011 | Subject >512 tokens before the query cannot use the block-25/layer-13-sliding route → reroute via layer-14 full K/V or fail | CANDIDATE | E008: distance sweep across the sliding window |
| H-012 | Multi-token subjects: answer information in layer-13 K/V is spread over the subject span (KV13_S 0.39 for R1 vs 0.98 for R2) | CANDIDATE | new multi-token relation set |
| H-013 | In-context associations use the same layer-13 K/V → block-25 route as parametric facts (MiniMax) vs. layer-14 full K/V (DeepSeek) — competing generator predictions | CANDIDATE | E008 |
| H-014 | Causal-effect spectroscopy: single-site effect matrices are low-rank and modular | INCONCLUSIVE — prereg criteria pass nominally (0.83 vs null p99 0.80) but driven by one high-variance column; standardised: 0.40 vs 0.21 (structure, not low rank) | meta_analysis/H014_spectroscopy |
| H-015 | A backup route carries subject information when block-25 subject content is mean-ablated | REFUTED (single-layer form): no second KV-shared layer matters (min 0.74); all-20-layer ablation retains 0.46 → H-016 | E006c |
| H-016 | Half of the effect enters the final position via layers 0–14 attention and is read out without KV-shared attention | CANDIDATE | E006d: early+late final-position attention ablation |
| H-017 | IT's final block shifts output from the factual token to chat-style formatting (' **', first names); absent in base | REFUTED (E009-base: late answer loss +3.3 pp and 148 lost answers in base too; only markdown is IT-specific) | E009 on base checkpoint (prereg in E009_reliability/PREREG_base.md) |
| H-018 | Internal stability signals detect errors better than softmax confidence | INCONCLUSIVE — IT: clustered CI includes 0; base: 0.857 vs 0.772, clustered CI [0.016, 0.154]; first-stable-layer best single signal in both | E009 |
| H-019 | IT answer-form shift lives in late MLP/PLE outputs (blocks 26–34), not the retrieval route; base→IT late cross-patch restores the factual token | CANDIDATE | cross-checkpoint patching |
| H-020 | Transfer: subject evidence is written at the last writer of shared state (KV source / MLA latent / SSM state) and read later | CANDIDATE | small public MLA/SSM/MoE models on CPU |
