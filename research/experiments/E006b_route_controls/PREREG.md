# E006b — Specificity controls for the block-25 route (AUDIT-002 follow-ups)
Preregistered 2026-10-25 (cycle 2) after E006 and AUDIT-002, before any E006b data. Status: registered.

## Questions (raised independently by both AUDIT-002 auditors)
1. Does block-25 attention at the final position actually read the subject positions (vs. merely writing the answer)?
2. Is the E006 noising effect specific to block 25, or would corrupting any late attention layer disrupt recall?
3. Is the effect subject-specific information, or a generic disruption of the block-25 output statistics?
4. Which block-25 heads are necessary (single-head noising)?

## Data
The 28 held-out pairs of E006 (same selection code; R1 12 + R2 16). Clean = a, corrupt = b. Retained = (LD − LD_b)/(LD_a − LD_b).

## Measurements (all in the clean run of a unless stated)
| id | measurement |
|---|---|
| MASS25_h | attention probability of block-25 head h at the final query on subject positions D (eager attention); baseline = |D|/T |
| MASS_L | head-averaged attention mass on D at the final query for every KV-shared layer L = 15..34 |
| NOISE_ATTN_L (L = 15..34) | block-L attention contribution at the final position replaced by the corrupt run's |
| MEAN_ATTN25 | block-25 attention contribution replaced by its mean over all held-out clean prompts of the same relation |
| NOISE_HEAD25_h | head h's o_proj input in block 25 replaced by the corrupt run's |

## Predictions (held-out medians)
- Q1: MASS25 of the head with the largest E006 HEAD25 recovery (head 0) ≥ 0.3 and ≥ 3× the uniform baseline |D|/T.
- Q2: NOISE_ATTN_25 retained ≤ 0.5 while NOISE_ATTN_L retained ≥ 0.8 for every other L (specificity of the layer).
- Q3: MEAN_ATTN25 retained ≤ 0.6 (removing subject-specific content without changing typical statistics hurts recall).
- Q4 (exploratory): NOISE_HEAD25_0 is the lowest retained among the eight heads.
Failure handling as in E006 (fewer than 12 usable pairs → INCONCLUSIVE; errors → rerun keeping outputs).
