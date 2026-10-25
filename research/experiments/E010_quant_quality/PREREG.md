# E010 — Weight-only quantisation: memory saving vs quality loss, including preservation of a known mechanism
Preregistered 2026-10-25 (cycle 3) before data. Status: registered.

## Question
"Quantisation changed only the rounding, not the architecture" — does it preserve (1) outputs, (2) factual accuracy and
(3) the *internal mechanism* (K-0006/K-0007 route), and how much memory does it save on this CPU path?

## Arms (all decoder Linear weights; embeddings, PLE table, norms untouched; compute stays fp32 via per-call dequantisation)
BF16 (reference, current checkpoint) · INT8 (symmetric, per-output-channel scale) · INT4 (symmetric, group size 32 along the input
dimension, stored in an int8 container: quality only, memory reported as the packed 4-bit equivalent).
Storage is real for INT8 (int8 tensors + fp32 scales replace bf16 weights in RAM); measured RSS reported.

## Data / metrics
(a) factset_v1 TEST half (396 items, odd index): top-1 agreement with BF16, accuracy delta (first-token, as E009), mean KL(BF16 || arm)
of the final distribution. (b) Mechanism: the 16 R2 held-out pairs of E006 (country→capital): KV13_all, KV14_all, ATTN_25 recovery and
NOISE_ATTN25 retained, compared with BF16 run in the same process. (c) Weight bytes and process RSS.

## Predictions
- Q1 INT8: top-1 agreement ≥ 98%, |Δaccuracy| ≤ 1 point, mean KL ≤ 0.01 nats; mechanism medians within ±0.05 of BF16.
- Q2 INT4 (g32): top-1 agreement ≥ 90%, |Δaccuracy| ≤ 3 points; mechanism *qualitatively* preserved (KV13_all ≥ 0.8, KV14_all ≤ 0.2,
  ATTN_25 largest effect) even if numbers move by > 0.05 — i.e. output degradation exceeds mechanism degradation.
- Q3 memory: INT8 reduces decoder-linear weight bytes by ≥ 45% vs BF16 and process RSS by ≥ 1.0 GB.
Exploratory: which item types flip under INT4 (relation, BF16 confidence).
