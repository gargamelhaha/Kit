# Hypothesis register
Preliminary hypotheses, ideas and open interpretations. Not verified knowledge. Machine-readable twin: `hypotheses.jsonl`.

| id | statement (short) | status | test |
|---|---|---|---|
| H-001 | In Gemma 4 E2B all cross-position information that reaches the final position is carried by layer-13/14 K/V; layers 15–34 act per-position | EXPERIMENTALLY_SUPPORTED (deterministic consequence of the code, confirmed by E001) | E001 |
| H-002 | Factual recall in E2B shows the ROME two-site pattern (last subject token mid-layers; final token late layers) | CANDIDATE | E004 (preregistered) |
| H-003 | Because subject-token states are only visible to later positions through layer ≤14 K/V, subject "enrichment" relevant for recall must complete by layer 14 (≤40% depth) | CANDIDATE | E004 P3 + follow-up K/V patching at layers 13/14 |
| H-004 | E2B contains induction heads that are causally necessary for repeated-sequence copying | CANDIDATE | E003 (preregistered) |
| H-005 | Logit-lens answer emergence occurs in the last third of layers in E2B (as in other LMs) | CANDIDATE | E002 (preregistered) |
| H-006 | PLE token-identity injection lets late layers re-read token identity, so residual-only interventions under-estimate token effects | CANDIDATE | ablate/patch PLE inputs vs residual (planned) |
| H-007 | The 60 k/v tensors stored for KV-shared layers are untrained remnants (absmax ≈ 0.06 vs 0.23–0.35 in trained k_proj) | CANDIDATE | compare with init distribution; check base checkpoint |
