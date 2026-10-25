---
name: meta-research
description: Mandatory procedure for measuring, auditing and accelerating this lab's own research process — durability, bottleneck profiling, root-cause analysis, value-of-information experiment choice, meta-analysis discipline and agent-utility accounting. Use at the start and end of every research cycle and whenever a step is slow, fails, or yields a surprising result.
---

# Meta-research skill (always on)

The research process is itself an object of study. Every cycle must leave the process measurably better, or
record why it could not. Process changes are hypotheses: validate them like any claim (before/after on an identical
workload that was not used to design the change).

## 0. Durability first
- Commit and push after every completed step (`git add -A && git commit && git push`). Cycle-1 lesson: the sandbox
  was lost 37 min in and *all* unpushed work (code, results, logs) vanished; recovery cost ≈ 30 min.
- Experiments write partial results incrementally; the queue (`experiments/queue.sh`) runs one model process at a time.
- `research/tools/memwatch.sh` must run while experiments run (kills experiments before the box limit is hit).

## 1. Instrument everything
- Every experiment runs through `research/tools/procmetrics.py`, appending to `research/logs/process_metrics.jsonl`
  `{ts, cycle, step, kind, wall_s, cpu_s, peak_rss_gb, outcome, artifact, notes}`.
- Progress lines use `flush=True`.

## 2. Find the bottleneck before optimizing
1. Rank steps by wall time × expected repetitions next cycle.
2. Write the cost model (FLOPs/token × tokens ÷ measured FLOP/s; bytes ÷ bandwidth). A large gap between model and
   measurement *is* the problem to explain.
3. Microbenchmark the suspected primitive in isolation, one variable at a time.

## 3. Root cause, not symptom
Keep asking "why" until the answer is a property of hardware, algorithm, data or specification. Record each case in
`research/meta_analysis/PROCESS_LEDGER.md` as symptom → evidence → root cause → fix → measured effect → status.
Prefer fixes that also remove work (e.g. proving parts of a computation inert) over fixes that only make it faster.

## 4. Choose experiments by value of information
`VOI ≈ P(result changes a decision or a claim status) × importance ÷ cost_hours`. Prefer experiments that discriminate
between live competing explanations. Preregister (`PREREG.md`: question, data, metric, numeric predictions, controls,
failure handling) before running; label post-hoc analyses as exploratory.

## 5. Meta-analysis discipline
- Fix inclusion criteria, effect measure and pooling rule before looking at results; never pool incompatible metrics.
- Report n, effect size, uncertainty (bootstrap CI / exact test) and heterogeneity; keep incompatible studies separate
  and explain disagreement (model, data, metric, tool version, controls).
- Same lab/codebase ≠ independent replication. Abstract-only reading ≠ evidence extraction.
- Literature saturation: stop a search line when ≥10 consecutive queries add <10% new relevant items; log it.

## 6. Agent utility accounting
Per subagent task: unique verified contributions, errors caught by audit, cost, share of claims surviving audit.
Same-model subagents are logical roles, not independent judges; for audits of important claims use a different model
family and give it method + data, not the desired conclusion. Stop agents that repeat work or miss acceptance criteria.

## 6b. External LLM API as a worker pool (cycle 2)
- Use `research/tools/llm_pool.py` only: cross-process cap of 6 in-flight calls (server limit, PL-7), SSE streaming (300 s
  gateway cut, PL-9), response cache, call log `logs/llm_calls.jsonl`. Credentials stay in `~/.config/lab/api.env`.
- Budget hidden reasoning: long-input tasks need `max_tokens` ≥ 32k (extraction) / 60k + `reasoning_effort: medium` (audits).
- Two families are available (DeepSeek, MiniMax) — use both for audits and screening and report agreement (κ); they are
  different from the coordinator's family, so they count as independent *reviewers*, not as independent *evidence*.
- Never let API models assert literature facts from memory: give them the source text and machine-verify every quote
  (`tools/deep_read.py`). Abstract screening (`tools/arxiv_scout.py`) is triage only.
- Agent-utility per task is computed from `llm_calls.jsonl` (calls, tokens, latency, error rate) plus verified-quote rate /
  issues accepted after audit.

## 3b. Root-cause heuristics learned
- Same failure across independent agents/runs → suspect the shared input or pipeline (PL-8).
- Failure at a suspiciously round number (300 s, exactly max_tokens) → a limit somewhere, find whose (PL-9).
- A status line that never shows failure is itself a suspect (PL-10): cross-check with an independent log.

## 5b. Metric neutral points (cycle 3, PL-18)
For any normalised effect metric (recovery, retained fraction) measure its empirical neutral point with an ablation that removes all
information of interest, and interpret effects relative to it. Lessons stored only in audit files get lost: promote them to this skill.

## 7. Process-change acceptance
Accept a change only with (a) identical-workload before/after numbers, (b) output equivalence within a stated tolerance
(or a deliberate correctness fix), (c) a PROCESS_LEDGER entry.

## 8. End-of-cycle review → `research/meta_analysis/cycle_<n>_review.md`
Time breakdown; top bottleneck and root cause; accelerations achieved (numbers); failures and root causes; agent
utility table; the process experiment for the next cycle.
