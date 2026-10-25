# E007 — Prefill cost of computing KV-shared layers at every position (engineering)
Preregistered 2026-10-25 (cycle 2) before data. Status: registered.

## Question
HF transformers 5.20.2 runs all 35 decoder layers over every prompt position during prefill. In Gemma 4 E2B,
layers 15–34 write no K/V, so their outputs at non-final positions are never read (K-0001/K-0002, T ≤ 512 tested).
How much prefill time does computing them only at the final position save, is it exact beyond the 512-token sliding
window, and how does the saving compare with the cost model and with Google's claim for Gemma 3n
("KV cache sharing … 2x improvement on prefill", developer blog, not yet verified against a primary measurement)?

## Cost model (computed from config before running)
Per-token linear-layer parameters: layers 0–14 = 563.9 M, layers 15–34 = 1299.2 M (double-wide MLP, 512-d global heads).
For linear FLOPs, full / last-only → 3.30× as T → ∞. Attention-score FLOPs (∝ T²) and the 262k-vocab unembedding of
one position are equal in both arms, so measured speedup should be < 3.30 and approach it from below as T grows,
until T² attention in layers 0–14 dominates.

## Design
- Model `google/gemma-4-E2B-it`, `use_fp32_compute`, batch 1, SDPA, `Runner.forward(logits="last")`.
- Inputs: BOS + the first T tokens of a fixed natural-English text (concatenated repository markdown,
  `research/LITERATURE_MAP.md` + `research/meta_analysis/methods_table.md`), T ∈ {64, 256, 512, 1024, 2048}.
- Arms: FULL (`last_only_from=None`) vs SKIP (`last_only_from=15, allow_long=True`). 1 warm-up, then 3 timed repeats per arm,
  arms interleaved (FULL, SKIP, FULL, …); report median wall time.
- Equivalence: final-position logits max|Δ|, top-1 and top-10 set agreement between arms.

## Predictions
- P1 exactness: max|Δlogit| ≤ 1e-3 and identical top-1 at every T, including T > 512 (sliding window).
- P2 speed: SKIP speedup ≥ 2.0× for every T ≥ 256.
- P3 cost model: speedup ≤ 3.30× at every T (if exceeded, the model is wrong, e.g. memory effects).
- P4 trend: speedup at T=1024 ≥ speedup at T=64.

## Limitations stated in advance
CPU only (2 vCPU, fp32 compute); no KV-cache decode or batched serving measured; energy not measurable in this sandbox
(no RAPL/power counters exposed); timing noise from a shared host.
