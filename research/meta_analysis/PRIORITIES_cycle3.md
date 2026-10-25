# Cycle 3 priorities (value-of-information ranking)
Written 2026-10-25 at cycle start, before any cycle-3 data. VOI ≈ P(result changes a claim/decision) × importance ÷ cost.
Importance 1–5 (scientific + engineering reach); P from current uncertainty; cost in CPU-hours on this box *after* the
prefix-cache acceleration planned below (without it, every row except D/G costs 3–10×).

| # | line | question | P(changes) | imp. | cost h | VOI | decision |
|---|---|---|---|---|---|---|---|
| A | **Process: prefix cache** | all interventions at layers ≥15 only need the last-position suffix → reuse layers 0–14 | 0.9 | 5 (multiplies every later exp.) | 0.3 | 15 | **do first** |
| B | **E008 sliding-window rerouting + in-context route** (H-011, H-013) | what happens when the subject is >512 tokens back? same route for prompt-defined facts? | 0.7 (two generators predict opposite) | 5 (mechanism + long-context reliability) | 0.5 | 7 | **do** |
| C | **Transfer it → base** (K-0006/K-0007 on `gemma-4-E2B`) | is the route a pretraining property or IT artefact? | 0.4 | 4 (cross-model transfer is a program goal) | 0.4 | 4 | **do** (download in background) |
| D | **H-014 causal-effect spectroscopy** on existing matrices | is the effect matrix low-rank/modular? | 0.5 | 3 (method) | 0.05 | 30 | **do** (no model) — high VOI but low importance ceiling |
| E | **E006c backup route** (H-015) + head-0 mean ablation (H-010 → KB) | what keeps 0.84 of recall without block-25 content? | 0.6 | 4 | 0.3 | 8 | **do** in the E008 process |
| F | AUDIT-003 (E006b, E007) | external check of K-0007 revision, K-0008 | 0.3 | 3 | API only | high | **do** (parallel) |
| G | Deep-read LIT-06 top sources (KV sharing, Mamba factual flow, MLA) | novelty of K-0006/7; transfer to SSMs | 0.3 | 3 | API only | high | **do** (parallel) |
| H | Internal error detectors / calibration | do internal signals beat softmax confidence? | 0.5 | 4 | 1.0 + data | 2 | next cycle (needs a set with natural errors; generate it now via API) |
| I | Quantization/precision effect on the route (engineering quality) | does int8/bf16 alter K-0006 mechanisms or just logits? | 0.3 | 3 | 0.5 | 1.8 | later |
| J | Knowledge editing via layer-13 K/V vs weights | interference | 0.5 | 5 | 3+ (weight edits on CPU) | 0.8 | later |

Process experiment (from cycle-2 review): **pre-run dual-family audit of the E008 PREREG**; metric = accepted design changes and
how many of the cycle-2 post-hoc issue types it catches.
