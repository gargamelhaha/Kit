# AUDIT-002 — Independent audit of E006 (K/V route, block-25 attention)
Date 2026-10-25. Auditors: DeepSeek 4.1 Flash (`free-deepseek-4.1-flash`) and MiniMax M3.1 Flash (`minimax-m3.1-flash`) via the lab API —
two model families different from the author's (Anthropic). Input: PREREG.md, run.py, analyze.py, runner.py, architecture facts
and per-pair raw results; **no** statement of the author's conclusion. Prompt + raw replies: `tools/audit_e006.py`,
`audits/AUDIT-002_raw.json` (first DeepSeek attempt exhausted 32k tokens on reasoning with an empty answer — rerun with 60k/medium).

## Verdicts
- Both: P1–P5 and NULL pass; data support layer-13 K/V sufficiency + localisation to the subject span and strong block-25
  involvement. Neither found an implementation bug in hooks/indices.
- MiniMax: analyze.py did not *enforce* the NULL-invalidation and underpowered rules (did not bind in this run). **Fixed.**
- Both: pairs share subjects → pair bootstrap anti-conservative. **Done:** subject-clustered bootstrap — every threshold still holds
  (KV13_all [0.947, 0.995], KV14_all [0.011, 0.025], ATTN_25 [0.654, 0.826], NOISE_ATTN25 [0.155, 0.306]).
- DeepSeek: held-out R1 not filtered explicitly against E004 pairs. **Checked:** 0 held-out pairs equal an in-sample pair, but all 12
  R1 held-out pairs share a subject with in-sample pairs → the fully independent confirmation is R2 (16 capital pairs, never used
  before), which gives the same pattern (KV13_all 0.982, KV14_all 0.017, ATTN_25 0.747, NOISE 0.211).

## Alternative explanations raised (→ action)
1. Prompts are < 512 tokens, so layer 13's sliding window is effectively full: the result is "layer-13 vs layer-14 stored K/V",
   **not** "sliding vs full". → K-0006 worded accordingly; H-011 / E008 tests beyond the window.
2. Block 25 might write the answer rather than read the subject. → E006b MASS25 (attention mass on subject positions).
3. Noising effect could be generic disruption. → E006b NOISE_ATTN_L sweep + MEAN_ATTN25.
4. HEAD25_0 could be an artefact of non-additive post-attention RMSNorm. → E006b single-head noising; H-010 stays CANDIDATE.

## Overclaims to avoid (adopted)
Block 25 is not shown to be necessary in every pair (max retained 0.602); recovery is unbounded (>1 occurs); no generalisation
beyond two relations, this checkpoint and short prompts.

## Agent utility
Unique valid issues: MiniMax 1 code-rule defect + 1 statistics issue; DeepSeek 1 split issue + the sliding-window confound (most
important). Both converged on 3 shared items. Cost: 4 calls, ≈ 110k completion tokens, ≈ 12 min wall.
