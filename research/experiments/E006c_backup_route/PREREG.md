# E006c — Corrected controls for K-0007 and search for the backup route (AUDIT-003, H-010, H-015)
Preregistered 2026-10-25 (cycle 3) before data. Status: registered.

## Fixes vs E006b (AUDIT-003)
All reference vectors and all patched runs use SDPA (attention masses come from a separate eager run, never patched); the
eager/SDPA discrepancy of E006b is quantified. Subject span = verified token offsets of the subject. Mean ablations use the
**leave-one-out** mean over the other correctly answered prompts of the same relation (excluding a and b) and a **norm-matched**
variant. Pair selection imported from `lib/pairs.py` and checked to equal E006's held-out pairs.

## Data
E006 held-out pairs (12 R1 + 16 R2). Clean run a; retained = (LD − LD_b)/(LD_a − LD_b) in the clean run after the ablation.

## Conditions (block-25 = attention contribution at the final position; heads = o_proj input of block 25 at the final position)
B25_NOISE (from b), B25_MEAN (LOO mean), B25_MEANNM (LOO mean rescaled to ‖a‖), B25_ZERO, H0_NOISE, H0_MEAN, H1to7_NOISE (heads 1–7
from b, head 0 clean), HALL_NOISE; JOINT_25_L = B25_MEAN + mean-ablation of block L's attention (L ∈ 15..34, L≠25);
LATE_ALL_MEAN = mean-ablation of all 20 KV-shared attention contributions; MASS (eager): head-averaged and per-head attention on the
verified subject span at block 25.

## Predictions (held-out medians)
- C1 consistency: |LD_eager − LD_sdpa| ≤ 1e-3 for all clean/corrupt prompts (E006b conclusions numerically unaffected).
- C2 non-necessity robust: B25_MEAN ≥ 0.7 and B25_MEANNM ≥ 0.7. (If < 0.6, K-0007's non-necessity statement is withdrawn.)
- C3 head 0: H1to7_NOISE ≥ 0.9 and H0_NOISE ≤ 0.6; H0_MEAN reported.
- C4 backup route (H-015): min over L of JOINT_25_L ≤ 0.5, and LATE_ALL_MEAN ≤ 0.3.
- C5 verified-span attention: block-25 head-0 mass on the verified subject span ≥ 0.5 (paired ratio to uniform ≥ 3 in ≥ 80% of pairs).
Bootstrap: subject-clustered, stratified by relation (10 000, seed 0).
