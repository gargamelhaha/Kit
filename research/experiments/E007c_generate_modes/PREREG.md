# E007c — Does the prefill skip (K-0008) save time in real generation modes without quality loss? + bf16 decode arm
Preregistered 2026-10-25 (cycle 3) before data. Status: registered.

## Question
K-0008 was measured on a single forward pass returning final-position logits. Real use is `model.generate` with a KV cache,
batching with padding, chat templates and multi-turn continuation. (a) Is `fastpaths.enable_prefill_skip` exact in these modes?
(b) What is the end-to-end gain (time-to-first-token TTFT, total time) when decode steps — which the skip cannot accelerate — are
included? (c) Separate precision arm: `set_small_batch_bf16(rows<=4)` for decode steps (a numerics change like quantisation):
speed gain vs quality loss.

## Arms
REF = stock HF generate (fp32 compute); SKIP = REF + prefill skip; SKIP+BF16 = SKIP + bf16 matmuls for inputs with ≤ 4 rows.
Greedy decoding, fixed length (min_new_tokens = max_new_tokens = G), `output_scores=True`.

## Conditions (gemma-4-E2B-it, BOS always present)
| id | mode | T (prompt) | batch | G |
|---|---|---|---|---|
| C1 | raw text continuation | 128 | 1 | 32 |
| C2 | raw text, beyond sliding window | 1024 | 1 | 24 |
| C3 | 4 prompts of different length (≈90–300), left-padded with attention mask | mixed | 4 | 24 |
| C4 | chat template, 3 factual/explanatory questions (run one by one) | ≈20–30 | 1 | 48 |
| C5 | multi-turn: C4 answer + new user turn appended, continued with the existing `past_key_values` (chunked prefill of the new turn) | +≈20 | 1 | 24 |

## Metrics
Token identity of generated sequences vs REF (per row); first divergence position; max|Δ| of step scores vs REF over the
steps before the first divergence; TTFT (time of the first model call), total wall time; sampled peak RSS.

## Predictions
- P1 exactness: SKIP generates token-identical sequences to REF in every condition and row, with max|Δscore| ≤ 1e-3.
- P2 TTFT: SKIP/REF TTFT speedup ≥ 2× in C2 and C3, ≥ 1.5× in C1.
- P3 end-to-end (expected small where decode dominates): total-time speedup in C1 ≤ 1.5× (decode-dominated) and ≥ 1.3× in C2.
- P4 bf16 decode arm: decode time per token improves ≥ 1.3× vs SKIP; quality: ≥ 90% of generated tokens identical to REF before
  first divergence averaged over conditions — reported as the cost of the precision change, not assumed harmless.
- Mode limit stated in advance: the skip is not applicable to scoring/perplexity (non-final logits) — not tested, by design.

## Failure handling
Any SKIP divergence → inspect masks/position handling for that mode; keep outputs; claim scope restricted to passing modes.
