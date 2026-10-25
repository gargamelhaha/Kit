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
| H-009 | Subject→final transfer happens in block-25 attention | EXPERIMENTALLY_SUPPORTED → K-0007 (E006: sufficient 0.757, noising retains 0.227, max in 28/28) | E006 |
| H-010 | Head 25.0 is the dominant extraction head (exploratory HEAD25_0 = 0.48; other heads ≤ 0.007) | CANDIDATE | head-level noising + attention pattern (E008) |
| H-011 | Subject >512 tokens before the query cannot use the block-25/layer-13-sliding route → reroute via layer-14 full K/V or fail | CANDIDATE | E008: distance sweep across the sliding window |
| H-012 | Multi-token subjects: answer information in layer-13 K/V is spread over the subject span (KV13_S 0.39 for R1 vs 0.98 for R2) | CANDIDATE | new multi-token relation set |
