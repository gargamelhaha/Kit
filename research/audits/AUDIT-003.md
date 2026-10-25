# AUDIT-003 — Independent audit of E006b and E007 and of claims K-0007, K-0008
Date 2026-10-25. Auditors: DeepSeek 4.1 Flash, MiniMax M3.1 Flash (lab API), given PREREGs, code, analyses, raw-result excerpts
and the current KB wording; no desired conclusion. Tool: `tools/audit_exp.py`; raw: `audits/AUDIT-003_raw.json`.

## Accepted issues → actions
| issue (auditor) | action |
|---|---|
| E006b reference vectors captured with **eager** attention, patched runs with **SDPA** (DS) | quantify in E006c (all conditions in one attention mode); K-0007 caveat added |
| "subject span" = differing token positions, not verified subject offsets (both) | E006c uses offsets from tokenising the subject; K-0007 now says "positions where the prompts differ" |
| mean ablation leave-one-in, not norm-matched → weak intervention (DS) | E006c: leave-one-out mean, norm-matched mean, zero and resample ablations |
| head-0 result selection-sensitive, multi-head interactions untested (both) | E006c: joint corruption of heads 1–7 with head 0 clean, and head-0 mean ablation |
| per-layer medians vs per-pair minima (layer 31 pair 0.85) (DS) | K-0007 wording now gives both |
| Q1 should be paired per prompt; clustering also by relation (DS) | E006c analysis: paired ratios, relation-stratified bootstrap |
| K-0008: "identical" → numerical equivalence; top-10 *set*; remove unverified Google statement; T=2048 exploratory (both) | done in KB |
| E007 inputs deviate from prereg file list (DS) | recorded in K-0008 uncertainty |
| "E006 artifacts missing" (MM) | **rejected**: E006 files exist; the audit packet only contained E006b/E007 by design (E006 audited in AUDIT-002) |

## Agent utility
DS: 1 implementation mismatch (eager/SDPA) not caught by the author or AUDIT-002; 1 mean-ablation weakness. MM: wording/scope items,
1 rejected (missing-artifact claim due to packet scope). Cost: 2 calls.
