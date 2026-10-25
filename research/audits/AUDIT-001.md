# AUDIT-001 — independent audit of cycle-1 code, E001/E001b, E002, E004 and K-0001..K-0004

- Auditor: subagent task `subagent_62dad3c4b6f33a23863d26af`, role Independent Auditor, model id `gpt-6-astra` (provider: OpenAI — a different family from the
  coordinator's Anthropic model), 2026-10-25. Given methods, code and data; not told the desired conclusions.
- Provenance note: the auditor ran in read-only review mode and returned its findings as a final message instead of writing
  this file. The coordinator transcribed the message below **verbatim** (no edits), then recorded actions taken.

## Auditor's report (verbatim)
> Reviewed all named artifacts and installed Transformers source using lightweight JSON checks. No files changed or model reruns performed.
>
> **Major findings**
> 1. `/tmp/hoplite/workspace/research/experiments/E004_factual_patching/PREREG.md:15` — Third-prompt substitution is not a zero-recovery null: neutral LD=0 implies median recovery **0.499**. Observed median |recovery| **0.146** cannot establish contamination. Preserve the failed threshold; add identity and answer-suppression controls. P1 also fails: **8/12 pairs**.
> 2. `/tmp/hoplite/workspace/research/scientific_knowledge.jsonl:4` — K-0004 understates maximum logit deviation: **2.321**, not ~0.9. Correct before retaining EXPERIMENTALLY_SUPPORTED.
> 3. `/tmp/hoplite/workspace/research/experiments/run_many.py:53` — Exceptions are swallowed without a failing process exit. Automation can treat failed experiments as successful. Aggregate failures and exit nonzero.
>
> **Minor findings**
> 4. `/tmp/hoplite/workspace/research/experiments/analyze_cycle1.py:23` — Persistence includes the first successful position despite “subsequent” preregistration. Exclude it; current median remains 1.
> 5. `/tmp/hoplite/workspace/research/experiments/lib/runner.py:82` — Missing `last_only_from >= first_shared` guard permits incorrect KV construction.
>
> Mask/rotary/PLE slicing, patch indices, serial buffer reuse, and unembedding appear consistent. E002 recomputes to median first-top1 index **28**, CI **[28,29]**. E004’s 25→26 jump implicates **block 25**, not 26; the 13→14 discontinuity admits frozen-KV explanations.
>
> Retain K-0001–0003 provisionally with narrower wording: sampled perturbations, observed error ≈3.624e−5, and prompt—not answer—perplexity. Batched equivalence and mechanistic follow-ups remain unverified.

## Actions taken by the coordinator
| finding | action | status |
|---|---|---|
| 1 control not a zero null; P1 fails | Failed threshold kept as failed in analysis; E004 interpretation rewritten; follow-up E004b (identity control, separate clean-answer restoration vs corrupt-answer suppression) registered in replication plan / state | done (follow-up open) |
| 2 K-0004 understated | statement corrected (0.42–0.65 with BOS, up to 2.32 without) | done |
| 3 run_many swallows failures | failures aggregated; exits 1 | done |
| 4 persistence definition | analysis excludes first top-1 position; P3 median unchanged (1.0) | done |
| 5 last_only guard | `ValueError` if `last_only_from < 15` | done |
| wording K-0001..K-0003 | narrowed as recommended | done |
| "block 25 not 26" | adopted in H-009 wording | done |
