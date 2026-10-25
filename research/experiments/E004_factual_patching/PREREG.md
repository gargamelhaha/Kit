# E004 — Causal tracing of factual recall with counterfactual activation patching
Replication target: Meng et al. 2022 (early site at last subject token in middle layers; late site at final token),
using counterfactual patching with a logit-difference metric (as recommended by Zhang & Nanda 2024; Heimersheim &
Nanda 2024) instead of Gaussian-noise corruption. Registered 2026-10-25 before data collection.
**Data.** Landmark prompts from `lib/datasets.py`; pairs (clean, corrupt) with equal token length, equal last-subject
position and different answer tokens; items whose clean top-1 is not the answer are excluded before patching.
Up to 12 pairs, chosen by a seeded shuffle (seed 0).
**Intervention.** Run the corrupt prompt; replace resid_pre at (layer L, position p) with the clean run's value.
Layers 0–14 at every position; layers 15–34 at the final position only (other positions are provably inert, E001).
**Metric.** LD = logit(clean answer) − logit(corrupt answer); recovery = (LD_patched − LD_corrupt)/(LD_clean − LD_corrupt).
**Predictions.** P1: at the last subject token, max over L ≤ 14 of recovery ≥ 0.5 in ≥ 70% of pairs. P2: at the final
position, recovery ≥ 0.5 for some L ≥ 15 in ≥ 70% of pairs. P3: at the last subject token, recovery at L = 14 is lower
than the max over L ≤ 14 (median over pairs). P4: median recovery of non-subject template tokens (positions after the
subject, excluding the final position) < 0.2 at every L ≤ 14.
**Control.** Patch from a third prompt with the same length and a different answer (last-subject position, L ≤ 14,
and final position, L ≥ 15); expected |recovery| < 0.1 (median).
