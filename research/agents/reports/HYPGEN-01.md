# HYPGEN-01 — Hypothesis generation by two external model families (coordinator review)
Task packet: `tools/hypgen_01.py` (context = MODELS.md architecture, SCIENTIFIC_KNOWLEDGE.md, HYPOTHESIS_REGISTER.md,
OPEN_QUESTIONS.md, tooling limits; one hypothesis per area × 7 areas; JSON with competing explanations, discriminating
experiment, numeric prediction, cost, value). Raw: `HYPGEN-01_raw.json` (final run, both models),
`HYPGEN-01_raw_run1_deepseek_only.json` (first run; MiniMax failed 3× at the 300 s gateway / reasoning budget, PL-9).

## Coordinator screening (accepted → register; rejected → reason)
| source | proposal (short) | decision |
|---|---|---|
| DeepSeek run1 | skip layers 15–34 at decode for speed | **rejected — invalid**: with a KV cache each decode step already processes one position; no saving exists |
| MiniMax | cache only layer-13/14 K/V (≈57 % less KV memory) | rejected — already how the architecture/HF cache work (shared layers never write cache); no new knowledge |
| DeepSeek vs MiniMax | in-context associations: layer-14 full K/V (DS) vs same layer-13 route (MM) | **accepted as H-013** — genuine competing predictions → E008 |
| DeepSeek | "causal-effect spectroscopy": low-rank modular structure of intervention-effect matrices | accepted as H-014 (≈0 CPU: re-analysis of E004/E006) |
| MiniMax | block-25 retrieval is a redundant multi-head ensemble | **contradicted by E006b** (only head 0 matters under noising: 0.37 vs ≥0.98) — logged, not registered |
| both | internal error detectors (logit-lens margin at index 28, block-25 entropy, post-25 residual) beat softmax confidence | deferred: needs a prompt set with ≥30 natural errors (current sets 60/62 correct) |
| both | knowledge editing via layer-13 subject K/V vs late MLP, interference on same-relation facts | deferred to weight-edit study (activation patching has no cross-prompt interference by construction) |
| DeepSeek | late double-wide MLPs 26–30 as readout | deferred (mlp_out sweep cheap; combine with H-015) |

## Agent utility
Proposals: 21 (7 DS run1 + 7 DS + 7 MM). Accepted to register: 2 (+1 merged into H-015 design). Invalid: 1 (DS). Redundant
with known architecture: 1 (MM). Contradicted by data within the same cycle: 1 (MM). Cost: 9 ok calls, ≈ 59k completion tokens.
Same-cycle lesson: generators propose plausible but sometimes physically impossible efficiency gains — every efficiency
hypothesis needs a cost-model check before it is registered.
