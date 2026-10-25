# E006 — Which shared K/V carries subject information to the answer position? (H-008, H-009)
Preregistered 2026-10-25 (cycle 2) before any E006 data were produced. Author: coordinator. Status: registered.

## Question
In Gemma 4 E2B layers 15–34 have no K/V projections; they read the K/V stored by layer 13 (sliding window) and
layer 14 (full attention). E004 found that patching the clean last-subject residual into the corrupt run at
`resid_pre` of layer 13 recovers 0.595 of the logit difference but at layer 14 only 0.010, and that final-position
recovery jumps from 0.056 (entering block 25) to 0.846 (entering block 26). Two hypotheses follow:
- **H-008**: the subject information that determines the answer reaches the final position through the layer-13
  (sliding) shared K/V, not through the layer-14 (full) shared K/V.
- **H-009**: the attention of block 25 (a sliding, KV-shared layer) is where this transfer happens.

## Data
- Same model/code path as E004: `google/gemma-4-E2B-it`, `use_fp32_compute`, BOS, `last_only_from=15`.
- Relation R1 (landmarks, `LANDMARK_TPL`) and relation R2 (capitals, `CAPITAL_TPL`, never used in any prior experiment).
- Items whose clean top-1 is the answer token. Pairs: equal token length, equal last-subject position, different
  answer token; each unordered pair once; seed-0 shuffle (E004 procedure).
- **Held-out set (confirmatory)**: R1 pairs *not* among the 12 analysed in E004 (up to 12) + up to 16 R2 pairs.
  **In-sample set**: the 12 E004 pairs (reported separately; they generated the hypotheses).

## Interventions (corrupt prompt b, clean prompt a; recovery = (LD_patched − LD_b)/(LD_a − LD_b), LD = logit(ans_a) − logit(ans_b))
Let D = positions where the token ids of a and b differ (the subject span), S = last subject position.
| id | intervention on the corrupt run |
|---|---|
| KV13_all / KV14_all / KVBOTH_all | replace layer-13 / layer-14 / both stored K/V at all positions by the clean ones |
| KV13_D / KV14_D | same, only at positions D |
| KV13_S | layer-13 K/V only at S |
| KV13_T | layer-13 K/V at all positions not in D (template tokens, incl. final) |
| ATTN_L (L = 15..34) | replace the attention contribution (post-attn norm) of block L at the final position by the clean one |
| HEAD25_h (h = 0..7) | replace head h's o_proj input in block 25 at the final position by the clean one |
| NOISE_ATTN25 | *clean* run with block-25 attention contribution at the final position taken from the corrupt run; retained = (LD − LD_b)/(LD_a − LD_b) |
| NULL | corrupt run with K/V replaced by its own K/V (must give recovery 0 within 1e-4) |

## Predictions (held-out set; medians over pairs)
- P1 (H-008): KV13_all ≥ 0.8 and KV14_all ≤ 0.2; and in ≥70% of pairs KV13_all − KV14_all ≥ 0.5.
- P2 (completeness of the K/V route, H-001 consequence): KVBOTH_all ≥ 0.9.
- P3 (localisation): KV13_D ≥ 0.7 and KV13_T ≤ 0.3.
- P4 (H-009 sufficiency): ATTN_25 ≥ 0.5, and ATTN_25 is the largest ATTN_L in ≥50% of pairs.
- P5 (H-009 necessity): NOISE_ATTN25 retained ≤ 0.5.
- Control: NULL |recovery| < 1e-4 in every pair (otherwise the run is invalid).
H-008 is supported if P1 holds, refuted if KV14_all ≥ KV13_all in the median. H-009 is supported if P4 and P5 hold;
"partially supported (sufficient, redundant)" if P4 holds but P5 fails; refuted if P4 fails.

## Analysis
Per-condition median, mean and 95% bootstrap CI (10 000 resamples over pairs, seed 0) for held-out, in-sample and
per relation. HEAD25 and ATTN_L sweeps beyond P4 are exploratory. Script: `analyze.py` → `analysis.json`.

## Failure handling
Run errors or a failed NULL control → fix and rerun, keeping the failed outputs. Fewer than 12 held-out pairs → report
as underpowered and keep the status at most INCONCLUSIVE.
