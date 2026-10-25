# AUDIT-004 — E009 (reliability) and E006c (K-0007) by two non-Anthropic families
Date 2026-10-25. Tool `tools/audit_exp.py`; raw `audits/AUDIT-004_raw.json`. Verdicts: DS "partially supported; major revisions";
MM "narrow IT result reasonably supported; broad wording not warranted".

| issue | found by | effect after fix |
|---|---|---|
| Cluster bootstrap collapsed subject multiplicities (`set(rng.choice(...))`) in E006/E006c analyses — CIs in K-0006/K-0007 invalid | DS | fixed in shared `lib/stats.py` (weights = mult(clean)·mult(corrupt), relation-stratified); CIs wider, every threshold still met |
| AUROC tie handling (double argsort) | DS + MM | average ranks; S2 0.760→0.757, S3 0.777→0.775, combined unchanged |
| DoLa plausible set truncated to first 50 token *ids* | DS + MM | real code defect; affected **0** items (no item had ≥ 50 plausible tokens; final top-1 always stored) → K-0009 valid |
| item-iid bootstrap ignores relation clustering | DS | relation-clustered CI for combined − maxprob [−0.03, 0.23] includes 0 → error-detection result moved to H-018 (not KB) |
| head-averaged attention mass missing (prereg) | MM | added: 0.31 |
| K-0007 wording too broad; R1/R2 heterogeneity | both | caveats added |
| "base results not auditable" | MM | rejected: base artifacts exist but were outside this packet (audited in E006 analysis) |
| unstable sigmoid; hard-coded 8 heads; rounding of saved signals | DS/MM | minor; noted, no effect on conclusions |

Agent utility: 4 valid major issues (2 found by both), 1 changed a claim's destination (H-018 instead of KB). This audit caught
a statistics bug that had passed AUDIT-002 and AUDIT-003 — evidence that audits need the analysis code, not only results.
