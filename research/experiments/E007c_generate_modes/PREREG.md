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

## Changes after pre-audit (PREAUDIT-E007c, before any E007c data)
Accepted: (1) C5 uses one immutable context (prompt + REF answer + follow-up) for all arms; chunk-1 prefill then cached
continuation; control = REF full re-prefill without cache reuse. (2) BF16 effect measured vs SKIP (precision effect), and vs REF
separately. (3) Warm-up + 2 interleaved repeats per arm (orders REF,SKIP,BF16 and reversed); report all values. (4) Premise check:
max |diff| of all cached K/V tensors SKIP vs REF. (5) C1 has 3 prompts; C3b = equal-length batch without padding (separates
batching from padding). (6) Assert bias-free linears for the bf16 path. (7) Metric renamed: `prefill_s` (first model call),
`frac_identical_tokens_all` over all G positions. Rejected: CUDA sync (CPU only); right padding (not a supported HF generation
mode); no-op-hook arm (hook overhead = 20 Python calls per prefill, negligible vs seconds); human quality evaluation (greedy
token identity is a stricter criterion than task quality for the exactness claim; for the bf16 arm task quality is not measured
and this is stated). P3's C1 bound is uninformative (pre-audit): reported descriptively only.
- Run restarted ~12:48Z: first run compared processed scores (EOS = -inf under min_new_tokens -> NaN diffs); now raw logits (output_logits). Aborted partial output kept as *_aborted_nan.*
- Cost cut at ~13:20Z (decided from run time, without seeing outcomes of the dropped cells): fp32 REF decode costs ~3 s/token,
  projected E007c wall ≈ 100 min. C1 reduced to its first prompt (already complete), C4 G 48 → 32, C4/C5 one repetition (timing
  from a single run, flagged). Completed rows are kept (resume); partial run copy in result_it_part1.json.
- PL-15 (13:25Z): the run died on the RSS guard in C2 REF. Root cause: the timing wrapper replaced `model.forward` with
  `(*a, **k)`, hiding `logits_to_keep` from `generate()`, which then computed logits for all prompt positions (+1.07 GB at T=1024;
  REF prefill also paid ≈ 3% extra time at T=128). Fixed with functools.wraps + assertion; all rows so far moved to
  `biased_run1/` (kept, not analysed); E007c is rerun from scratch after E008.
- PL-17 (14:52Z): second RSS-guard exit in C2 REF (6.80 GB) after the PL-15 fix. Changes: malloc_trim(0) + RSS logged before each generate (tests heap-fragmentation hypothesis), C2 moved last, RSS guard 7.2 GB for this run only (memwatch 1 GB floor remains). Completed C1 rows kept (resume).
- PL-17 root cause (16:15Z, third RSS-guard exit, 7.23 GB in C3): the harness kept complete outputs (scores [b, G, 262144] fp32 ≈ 100 MB/generation at b=4, plus KV caches) for every repeat. Fix: one full output per arm, timings only for repeats; malloc_trim per item. Fragmentation hypothesis not needed to explain the growth.
