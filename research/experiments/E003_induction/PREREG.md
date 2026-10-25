# E003 — Induction heads in Gemma 4 E2B (replication of Olsson et al. 2022; Elhage et al. 2021)
Registered 2026-10-25 before data collection.
**Data.** 16 sequences: BOS + 30 random token ids (uniform in [1000, 30000), seed 0) repeated twice (61 tokens).
**Measures.** (1) mean NLL on repeat 1 vs repeat 2; (2) induction score per head = mean attention probability from
position i in repeat 2 to position i−30+1; (3) causal: mean-ablate head outputs (o_proj input replaced by its clean
mean over batch×positions) for (a) top-5 heads by induction score and (b) 30 random 5-head sets (seed 1).
**Predictions.** P1: repeat-2 NLL < 0.25 × repeat-1 NLL. P2: ≥1 head with induction score > 0.3. P3: top-5 ablation
raises repeat-2 NLL more than all 30 random sets (p < 1/31) and by ≥3× the random median. P4: repeat-1 ΔNLL from
top-5 ablation < 25% of its repeat-2 ΔNLL.
**Exploratory.** Location of induction heads relative to the KV-shared block (layers 15–34).
**Failure handling.** No post-hoc change of k or ablation type; variants only as labelled exploratory follow-ups.
