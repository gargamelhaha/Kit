# E002 — Logit lens on Gemma 4 E2B (replication of nostalgebraist 2020; cf. Belrose et al. 2023)
Registered 2026-10-25 before data collection.
**Data.** `lib/datasets.py`: 32 landmark prompts ("The {X} is located in the city of") + 30 capital prompts
("The capital of {X} is"). Only items whose final top-1 is the answer's first token are analysed (fixed criterion).
**Method.** Final position; decode each resid_pre (0..34) and the final residual via the model's own final RMSNorm →
tied unembedding → softcap 30. Record rank and probability of the answer token. IT and base checkpoints.
**Predictions.** P1: median first layer at which the answer is top-1 is ≥ 24 (last third). P2: median answer rank at
layers ≤ 5 is > 1000. P3: once top-1, the answer stays top-1 in ≥ 80% of subsequent layers (median over items).
**Limitation.** Logit lens is biased in early layers (Belrose et al. 2023); no tuned lens in this cycle.
