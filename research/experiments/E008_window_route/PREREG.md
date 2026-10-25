# E008 — Routes for distant subjects (sliding-window boundary) and for in-context associations (H-011, H-013)
Preregistered 2026-10-25 (cycle 3) before data. Status: registered; pre-run audit by two API families pending (cycle-3
process experiment) — any design change it causes is listed in §Changes after pre-audit.

## Background
K-0006/K-0007: on short prompts the subject reaches the answer via layer-13 stored K/V read by block-25 attention (head 0).
Layer 13 is a sliding-window layer (512): the final query of layers reading its K/V cannot attend to keys more than 511
positions back. Layer-14 K/V (full attention, read by blocks 19, 24, 29, 34) has no such limit.

## Part A — distance sweep (H-011)
Prompt = BOS + "Note: the landmark in this report is the {S}." + F + " The landmark named in the note is located in the city of"
where F is a fixed neutral filler (repository prose, identical for both prompts of a pair) of length chosen so that the distance
from the last subject token to the final position is d ∈ {≈40 (near), ≈400 (inside window), ≈700 (outside window)}.
Pairs: landmark subjects whose note sentences have equal token length (E006 selection rule), up to 8 pairs with both clean
items answered correctly at d≈40; the same pairs at all d. Correctness at each d recorded for every item (behaviour).
Interventions (corrupt run b, clean a, recovery as in E006): KV13_D, KV14_D (subject span), KV13_near (all positions within
the last 512 that are not in D), KV14_all, KV13_all, ATTN_L for L = 15..34 (final-position attention contribution), NULL.

Predictions
- A1 (window, mechanical): at d≈700, KV13_D |recovery| ≤ 0.05 (keys outside the window are masked).
- A2 (H-011 rerouting): at d≈700, among items still answered correctly, KV14_D median ≥ 0.5 and the largest ATTN_L is a
  full-attention block (19/24/29/34).
- A3 (inside window, same route as K-0006): at d≈40 and d≈400, KV13_D ≥ 0.5 and ATTN_25 is the largest ATTN_L in ≥ 50% of pairs.
- A4 (behaviour): accuracy at d≈700 is lower than at d≈40 (reported with exact counts; no threshold).
Competing outcome recorded in advance: if at d≈700 both KV13_D and KV14_D are small but recall is correct, information was
copied to nearer positions before layer 13 (test: KV13_near ≥ 0.5).

## Part B — in-context association route (H-013; competing predictions from two generator families)
Prompt = BOS + "In this game, the secret word for the {N} box is {W}." + short filler (≈30 tokens) + " The secret word for the {N} box is"
with single-token W (common nouns) and N ∈ colours; corrupt = same N, different W (equal length). D = positions of W.
Interventions: KV13_D, KV14_D, KV13_all, KV14_all, ATTN_L (L = 15..34), NULL. Pairs: up to 16 with both items correct.
- B-MM (MiniMax HYPGEN-01): KV13_D ≥ 0.7 and KV14_D ≤ 0.2 (same route as parametric facts).
- B-DS (DeepSeek HYPGEN-01): KV14_D ≥ 0.5 (in-context binding through full-attention K/V).
Both cannot hold; result decides between them or refutes both.

## Analysis
Medians + subject-clustered bootstrap CIs (10 000, seed 0), per d. Fewer than 6 usable pairs in a cell → INCONCLUSIVE for that cell.
NULL must be |r| < 1e-4. Exploratory: per-head attention mass on D in the block with the largest ATTN_L.

## Changes after pre-audit (PREAUDIT-E008: DeepSeek 17 issues, MiniMax 15; before any E008 data) — this section supersedes the above where they differ
1. **Distance no longer confounded with length**: every Part A/B prompt has the same filler budget (720 filler tokens) and
   query; only the position of the note/binding sentence inside the filler moves. Exact distances (last subject token → final
   position) d ∈ {60, 400, 505, 525, 700}; 505/525 bracket the 512 window (keys with distance ≥ 512 are masked for layer-13 K/V
   readers; recorded from the mask). Filler A (5 distances) and an independently generated filler B (d ∈ {60, 700}) —
   `benchmarks/fillers_v1.json`, both checked against all subject/answer names.
2. **No outcome selection**: pairs are drawn (seed 0) from landmark items answered correctly in the *short* E006 template; Part A
   uses exactly N = 10 pairs; all pairs enter route analyses at every d. Accuracy is reported for all 20 items at every d.
   Usable cell ⇔ |LD_a − LD_b| ≥ 2 (pre-specified denominator floor); denominators reported.
3. **A1 is a mechanical check** (mask semantics), not evidence for H-011.
4. **Positions**: KV{13,14}_D (subject span), KV{13,14}_BETWEEN (strictly between note and query template),
   KV{13,14}_TPL (query template incl. final), KV13_all, KV14_all, KVBOTH_all, NULL, ATTN_L (15..34).
5. **Part B redesigned** so copying is insufficient: three bindings in one sentence ("the red box holds the word W1, the blue …
   W2, and the green … W3"), the query names one colour; corrupt = different word for the queried colour only. Exactly N = 12
   items (queried colour rotates), at d ∈ {60, 700} with the same 720-token filler budget, so layer-13 access is manipulated
   (inside vs outside its window) while layer-14 access is constant. Positions patched: W (D), queried colour (N), the queried
   clause, all positions after the binding sentence, both K/V sources separately and jointly.
6. **Route classification rule** (per cell, medians): L13-only (KV13 ≥ 0.5, KV14 < 0.2); L14-only (KV14 ≥ 0.5, KV13 < 0.2);
   mixed (both ≥ 0.2); redundant (both < 0.5 but KVBOTH ≥ 0.5); neither (KVBOTH < 0.5). The generator predictions map to
   L13-only (MiniMax) vs L14-only (DeepSeek) at d = 60; at d = 700 H-011 predicts L14-only.
7. **Decision rule**: a prediction is "supported" when the median passes and the subject-clustered (Part B: item-clustered) 95%
   bootstrap CI lies on the predicted side of the threshold; "weak" when only the median passes; cells with < 6 usable pairs are
   INCONCLUSIVE. d = 60 is a positive control (replication of K-0006 in a long prompt), not a new test.
Rejected: several fillers per pair beyond A/B and power analysis (CPU budget: each long-prompt cell costs ≈ 35 s; N fixed in
advance instead); nonce words for Part B (single-token common nouns used; prior bias measured by the corrupt run).
