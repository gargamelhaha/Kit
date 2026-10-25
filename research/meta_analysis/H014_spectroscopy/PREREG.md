# H-014 test — Causal-effect spectroscopy on existing intervention matrices (no new model runs)
Preregistered 2026-10-25 (cycle 3) before computing any decomposition. Status: registered.

Data: E006 recovery matrices (rows = pairs: IT 40 + base 40; columns = 44 interventions, NULL excluded). Rows centred by
column means (variation across pairs; uncentred low rank would be trivially driven by column means — the artefact the
hypothesis register flagged). SVD; explained variance of top-k components.
Null: 1 000 matrices with each column independently permuted across rows (keeps marginals, destroys co-variation).
H-014 predictions: (i) top-3 components explain ≥ 80% of centred variance; (ii) this exceeds the 99th percentile of the null;
(iii) |loadings| of component 1/2 separate site groups {KV13_* and ATTN_25/HEAD25_0} from {KV14_*, other ATTN_L} (group mean
|loading| ratio ≥ 3); (iv) transfer: components fit on IT rows explain ≥ 70% of the centred variance of base rows (vs null).
If (i) holds but (ii) fails, the low rank is a marginal artefact → H-014 REFUTED for this data.
