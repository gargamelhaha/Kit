# E006d — Where does the late-attention-independent half come from? (H-016)
Preregistered 2026-10-25 (cycle 3) before data. Status: registered.
Data: E006 held-out pairs (12 R1 + 16 R2), clean run a, retained = (LD − LD_b)/(LD_a − LD_b). Final-position ablations with
leave-one-out (LOO) means over the other correct prompts of the same relation (excluding a and b), SDPA throughout.
| id | ablation at the final position |
|---|---|
| LATE_ATTN_MEAN | attention output of layers 15–34 → LOO mean (replicates E006c LATE_ALL_MEAN) |
| EARLY_ATTN_MEAN | attention output of layers 0–14 → LOO mean |
| ALL_ATTN_MEAN | attention output of all 35 layers → LOO mean |
| LATE_ATTN_RESAMPLE | layers 15–34 attention ← a random other prompt of the same relation with a different answer (seed 0) |
| LATE_MLP_MEAN | MLP output of layers 15–34 → LOO mean |
| ALL_ATTN_MEAN+LATE_MLP_MEAN | both |
Predictions (H-016): P1 ALL_ATTN_MEAN ≤ 0.15; P2 EARLY_ATTN_MEAN ≥ 0.7; P3 LATE_MLP_MEAN ≥ 0.5 alone (MLPs read out, they don't
carry subject identity). Alternative "mean vectors carry relation priors": LATE_ATTN_RESAMPLE ≪ LATE_ATTN_MEAN (resampling a
wrong-answer prompt should push towards b) — reported. Analysis: medians + relation-stratified subject-cluster CIs (lib/stats.py).
