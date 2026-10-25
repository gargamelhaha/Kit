# Process ledger (meta-research skill §3)
Format: symptom → evidence → root cause → fix → measured effect → status.

## PL-1 Slow batched inference
- Symptom: 0.57 tok/s generation; 90.9 s per 16×32-token forward (session 1, bf16).
- Evidence: microbenchmark of a 1536→6144 linear (session 1, not archived — numbers from the run transcript):
  n=32 tokens: fp32 104 GFLOP/s vs bf16 8.4 GFLOP/s; n=1: bf16 faster (memory-bound). `/proc/cpuinfo`: avx512f, no avx512_bf16.
- Root cause: no native bf16 GEMM on this CPU → emulated bf16 for multi-token batches.
- Fix: `use_fp32_compute` (bf16 storage, per-call fp32 upcast). Measured 90.9 → 29.0 s (session 1, 3.1×) and
  107 → 63.8 s (E000 archived run, 1.7×, CPU shared with 4 subagents). Also lowers numerical error (bf16↔fp32 last-token
  logit max|Δ| up to 0.88; top-1 always agreed, `E000_smoke/result_it.json`).
- Status: accepted. Generation (n=1) still uses bf16.

## PL-2 Patching sweeps too expensive
- Root cause (architectural): layers 15–34 read only K/V produced by layers 13/14 → non-final positions after layer 14
  cannot influence final-position outputs.
- Evidence: E001 — noise (σ=50) on all non-final positions at resid_pre of layers 15/16/25/34 gives max|Δlogit| = 0.0 exactly,
  vs 17.7–36.4 at layers 5/13/14 (two prompts; identical numbers in two separate runs).
- Fix: `Runner(last_only_from=15)`; exact to 3.6e-5. Speed 64.3 → 29.8 s on an 8×40 batch (2.16×); a patching grid shrinks
  from 35·T to 15·T + 20 conditions.
- Status: accepted.

## PL-3 Nonsense predictions without BOS
- Evidence: E000 — "The capital of France is": ppl 2.3e7 and top-1 " France" without BOS vs ppl 257 and " Paris" with BOS.
- Root cause: the tokenizer config does not add BOS (`add_bos_token` unset); Gemma is trained with BOS.
- Fix: `encode()` helper prepends BOS. Status: accepted (same pitfall independently reported for Gemma 4 12B-it, LIT-03).

## PL-4 Work lost to sandbox resets (×3)
- Symptom: sandbox generations lost at ~06:13, ~07:04, ~07:13 UTC; all unpushed files gone each time.
- Evidence: uptime reset; empty worktree; each loss 1–4 min after starting a 5–7 GB model process while 3–5 subagents ran
  in the same 8 GiB sandbox. Hypothesis "memory accumulates across successive model processes" was refuted by reset #3
  (first model process in a fresh sandbox).
- Root cause of the *loss*: no commit/push between steps (confirmed). Root cause of the *resets*: unresolved; leading
  hypothesis = combined memory of model process + concurrent subagent processes exceeding the box (container killed, no
  in-sandbox OOM trace). Reported to the platform.
- Fixes: (1) push after every step + auto-push per experiment (`run_many.py`); (2) single long-lived model process;
  (3) run model work only when no subagents are active; (4) request 16 GiB once subagent leases clear.
- Measured cost: ≈ 70 min of rework across three resets (re-download 3–5 min, reinstall ~2 min, re-running experiments).
- Status: mitigations (1)–(3) in place; (4) pending.

## PL-5 Memory-mapping all weights regressed speed
- Evidence: E000 attempt with all weights memory-mapped: 616 s wall, 70.9 s CPU (≈11% CPU utilization).
- Root cause: weight pages repeatedly faulted from the sandbox filesystem (I/O-bound).
- Fix: only the PLE table stays mapped (row lookups); decoder + embedding copied to RAM; unembedding computed in 32k-row
  chunks to avoid a 1.6 GB fp32 transient. Status: accepted.
