# AUDIT-005 — E006c/E006d and the corrected K-0007
Date 2026-10-25. DeepSeek returned an empty answer (content '', no error; reasoning budget presumably exhausted at 60k) — no
output to evaluate. MiniMax: "material correction required". Raw: `audits/AUDIT-005_raw.json`.

| issue (MiniMax) | decision |
|---|---|
| "critical": head-mass extraction `ea['probs'][:, span]` mis-indexes a 4-D tensor | **rejected — false positive**: `capture()` stores `p[0, :, -1]` (shape [heads, keys]) — run.py line 22 — so `[:, span].sum(-1)` yields the 8 per-head masses on the final query; result shape matches |
| "result inconsistent with code version" | rejected (follows from the false positive); results were produced by the committed code (git history) |
| "the same" (0.464 vs 0.460) needs a paired contrast | **accepted**: paired difference 0.002 [−0.006, 0.006] (relation-stratified subject-cluster bootstrap) |
| "neutral point" should be called the operational all-attention baseline and explained (≈ 0.46, not 0) | accepted |
| 63% normalised: give formula + CI | accepted: 0.67 [0.55, 0.80] per pair (E006d/normalized_block25.json) |
| "only head" — heads 1–7 tested jointly; R1/R2 heterogeneity | accepted (wording) |
| "subject info could travel via residual/MLP pathways" | rejected as stated: attention is the only cross-position path; MLP ablation (0.58) is reported as readout dependence — premise now explicit in K-0007 |
| resample filter compares first answer tokens | minor, accepted as limitation (answers in these relations have distinct first tokens) |
| base results not supplied / "still running" | packet scope; base E006 analysis exists (analysis_base.json); wording clarified |

Agent utility: MiniMax 4 accepted wording/statistics items, 2 false positives (one labelled critical). DeepSeek 0 (empty).
Lesson: auditor claims of "critical bugs" must be verified against the code before acting — this one would have triggered a
needless rerun.
