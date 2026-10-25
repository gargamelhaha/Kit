# Hypothesis register
Preliminary hypotheses, ideas and open interpretations. Not verified knowledge. Machine-readable twin: `hypotheses.jsonl`.

| id | statement (short) | status | test |
|---|---|---|---|
| H-001 | In Gemma 4 E2B all cross-position information that reaches the final position is carried by layer-13/14 K/V; layers 15–34 act per-position | EXPERIMENTALLY_SUPPORTED (deterministic consequence of the code, confirmed by E001) | E001 |
| H-002 | Factual recall in E2B shows the ROME two-site pattern (last subject token mid-layers; final token late layers) | INCONCLUSIVE — late site 12/12; early site P1 failed (8/12); control not a zero null (AUDIT-001) | E004 → E004b |
| H-003 | Because subject-token states are only visible to later positions through layer ≤14 K/V, subject "enrichment" relevant for recall must complete by layer 14 (≤40% depth) | INCONCLUSIVE — superseded by H-008 | E004 P3 |
| H-004 | E2B contains induction heads that are causally necessary for repeated-sequence copying | CANDIDATE | E003 (preregistered) |
| H-005 | Logit-lens answer emergence occurs in the last third of layers in E2B (as in other LMs) | EXPERIMENTALLY_SUPPORTED → K-0005 | E002 |
| H-006 | PLE token-identity injection lets late layers re-read token identity, so residual-only interventions under-estimate token effects | CANDIDATE | ablate/patch PLE inputs vs residual (planned) |
| H-007 | The 60 k/v tensors stored for KV-shared layers are untrained remnants (absmax ≈ 0.06 vs 0.23–0.35 in trained k_proj) | CANDIDATE | compare with init distribution; check base checkpoint |
| H-008 | Subject info reaches the final position via layer-13 sliding K/V, not layer-14 full K/V (E004: recovery 0.595 at L13 vs 0.010 at L14) | CANDIDATE | E006: patch K/V of layer 13 vs 14 only |
| H-009 | Subject→final transfer completes in block 25 (final-position recovery 0.056 before vs 0.846 after) | CANDIDATE | ablate block-25 attention at final position |
