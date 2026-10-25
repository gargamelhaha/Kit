# Cycle 2 review (2026-10-25, ≈ 80 min wall)

## What was done (evidence in git history of branch hoplite/ambrakia-a2a0753c)
- Workspace recovery: Axg history imported (read-only source repo), venv + weights re-created in ≈ 12 min (download 607 s, sha256 = MODELS.md).
- External LLM API integrated as a worker pool (`tools/llm_pool.py`): measured cap, cross-process limiter, streaming, cache, call log.
- Experiments: E006 (K/V route, block 25; all 5 preregistered predictions pass), E006b (specificity; Q3 fails → K-0007 corrected),
  E007 (prefill skip; exact, 1.8–3.3×), E007b (exploratory memory). New knowledge K-0006, K-0007, K-0008.
- Literature: 25 priority sources deep-read with dual-model, quote-verified extraction (reading debt list of LITERATURE_MAP
  cleared at machine level); LIT-06 (20 logged arXiv queries, 125 new candidates, 40 included by both screeners, κ = 0.733).
- Audit: AUDIT-002 by two non-Anthropic families (no desired conclusion given) → 3 fixes, 4 alternative explanations, 2 tested in E006b.
- Hypotheses: H-008/H-009/H-010 supported, H-011…H-015 registered (two from external generators, one invalid proposal rejected).

## Time breakdown (process_metrics.jsonl + llm_calls.jsonl)
| step | wall | note |
|---|---|---|
| env + weights | ≈ 12 min | parallel with API work |
| E006 | 22.5 min | 40 pairs × 45 conditions; 0.7 s/condition |
| E007 + E007b | 13.6 + 3.7 min | T=2048 FULL hit RSS guard |
| E006b | 12.6 min | 28 pairs × 29 conditions + eager-attention captures |
| API: deep_read | 52 calls, 1.06 M prompt / 0.79 M completion tokens; p50 latency 83 s (DS) / 169 s (MM) |
| API: LIT-06 screening | 250 calls, p50 5–9 s |
| API: audit + hypgen | 13 calls, ≈ 140k completion tokens |
Model process and API work overlapped all cycle (model CPU-bound, API I/O-bound): no step waited on the other.

## Top bottleneck and root cause
API throughput collapsed twice (78 × HTTP 429; 300 s 504s). Root causes: per-key server cap vs per-process limiter (PL-7b);
nginx 300 s cut on non-streaming responses + reasoning eating the token budget (PL-9). After fixes: 0 × 429 and no 504 in the
final hypgen/audit runs. Remaining model-side bottleneck: patching conditions at ≈ 0.7 s each (fp32 CPU GEMMs).

## Accelerations achieved (numbers)
- Literature evidence extraction: 25 full papers in ≈ 25 min wall, 95–96 % of 1 668 quotes machine-verified (vs cycle 1: 2 papers full text).
- Verification re-run after PL-8 without any API cost (cache + `--reverify`).
- Prefill for long prompts: 2.4–3.3× (K-0008) — usable for all future final-position experiments at T > 512 (`allow_long`).

## Failures and root causes
PL-7b, PL-8 (pdftotext layout), PL-9, PL-10 (queue exit code), E007 T=2048 FULL OOM-guard (PL-11, not a reset).
Analysis bug: numpy bool JSON serialisation (twice) — trivial, fixed. Prereg error: E007 cost model ignored skipped T² attention.

## Agent utility table
| agent / role | output | verified / accepted | errors caught | cost |
|---|---|---|---|---|
| DeepSeek extractor | 776 quotes / 25 papers | 96.4 % quote-verified | — | 26 calls |
| MiniMax extractor | 892 quotes / 24 parsed | 95.1 % | 1 unparsable JSON (ROME) | 26 calls |
| screeners (both) | 125 candidates | κ = 0.733; 40 joint includes | — | 250 calls |
| DeepSeek auditor | 1 split issue + sliding-window confound | both accepted, confound decisive for K-0006 wording | — | 2 calls |
| MiniMax auditor | rule-enforcement bug + clustering issue | both accepted, fixed | — | 2 calls |
| generators | 21 proposals | 2 registered | 1 invalid, 1 redundant, 1 contradicted | 9 calls |

## Process experiment for next cycle
Test whether *dual-family audit before running* (on the PREREG) catches design flaws that the post-hoc audit caught here
(sliding-window confound, mean-ablation control). Metric: number of accepted design changes per prereg, and the
fraction of post-run audit issues that a pre-run audit would have prevented. Workload: E008 prereg.
