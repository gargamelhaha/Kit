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

## Changes after pre-audit
(filled in after the audit, before running)
