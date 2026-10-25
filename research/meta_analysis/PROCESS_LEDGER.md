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

## PL-4 Work lost to sandbox resets (×4)
- Symptom: sandbox generations lost at ~06:13, ~07:04, ~07:13, ~07:47 UTC; all unpushed files gone each time.
- Key regularity (found on reset #4): **every** reset happened while E002 (logit lens) was running — E000/E001 never
  triggered one. Reset #4 came ≈9 min into E002 (mid-run, not at load). The sandbox box is 8 GiB and cannot be raised
  (resize refused: "exceeds current resource ceilings").
- Evidence: uptime reset; empty worktree; each loss 1–4 min after starting a 5–7 GB model process while 3–5 subagents ran
  in the same 8 GiB sandbox. Hypothesis "memory accumulates across successive model processes" was refuted by reset #3
  (first model process in a fresh sandbox).
- Root cause of the *loss*: no commit/push between steps (confirmed). Root cause of the *resets*: unresolved; leading
  hypothesis = combined memory of model process + concurrent subagent processes exceeding the box (container killed, no
  in-sandbox OOM trace). Reported to the platform.
- Fixes: (1) push after every step + auto-push per experiment (`run_many.py`); (2) single long-lived model process;
  (3) run model work only when no subagents are active; (4) request 16 GiB once subagent leases clear.
- Measured cost: ≈ 70 min of rework across three resets (re-download 3–5 min, reinstall ~2 min, re-running experiments).
- Status: mitigations (1)–(3) in place; (4) impossible (ceiling). See PL-6 for the E002-specific analysis.
- Also observed: the platform's headless Chrome (agent-browser, 17 processes when subagents use web tools) shares the
  box; it is stopped before model runs (its true memory share is not measurable here: no PSS under gVisor).

## PL-5 Memory-mapping all weights regressed speed
- Evidence: E000 attempt with all weights memory-mapped: 616 s wall, 70.9 s CPU (≈11% CPU utilization).
- Root cause: weight pages repeatedly faulted from the sandbox filesystem (I/O-bound).
- Fix: only the PLE table stays mapped (row lookups); decoder + embedding copied to RAM; unembedding computed in 32k-row
  chunks to avoid a 1.6 GB fp32 transient. Status: accepted.

## PL-6 E002-specific resets: hypotheses tested
- H-a "unembedding outside no_grad retains fp32 weight copies for autograd" — **refuted** by E005 (RSS unchanged:
  1.143 GB with and without grad; linear's weight-grad needs only the input). Kept fix anyway: params `requires_grad=False`.
- Side finding (E005): RLIMIT_DATA is not enforced in this sandbox → OS-level per-process limits cannot protect it.
- H-b (leading, unconfirmed): allocation churn. E002 performs ≈18k allocate/free cycles of 200 MB (fp32 unembedding
  chunks: 62 items × 36 layers × 8 chunks) plus ≈15k cycles of ≤75 MB; E001 ≈2 orders of magnitude fewer. Under gVisor
  freed anonymous memory may not be returned to the host promptly, so host-level usage can exceed the box while in-sandbox
  RSS looks fine.
- Fix (removes the suspected mechanism and bounds memory): one persistent fp32 buffer per weight shape (~0.5 GB total) —
  zero per-call allocations for weights; in-process RSS guard thread (exit at 6.8 GB, logs to memwatch.log); E002 writes
  results after every item. Equivalence of the buffered path is tested in E001b (must be bit-identical).
- Status: mitigated; root cause remains a hypothesis unless E002 now completes repeatedly (weak evidence) — a direct
  churn test would risk another reset and is deferred.

## PL-7 External LLM API: real parallelism limit (cycle 2)
- Symptom: the API was described as "no limits"; a naive pool at concurrency 8 lost 7/16 requests.
- Evidence: `logs/llm_ramp.json` — MiniMax c=1 0.30 req/s, c=4 1.24 req/s (p50 3.1 s), c=8 → HTTP 429
  `"Too many parallel requests: up to 6 at a time"`; DeepSeek c=1 had one 120 s read timeout (heavy latency tail).
- Root cause: server-side per-key cap of 6 in-flight requests (a specification property, not a client bug).
- Fix: process-wide `BoundedSemaphore(6)` in `tools/llm_pool.py`, 429 → wait 2 s without consuming a retry, 300 s timeout.
  Measured: requested concurrency 12 → 12/12 ok in 7.1 s (before the fix: 9/16 ok at c=8).
- Second root cause found on first real task: both models spent the entire `max_tokens=8000` on hidden reasoning for a
  21k-token paper (completion_tokens = reasoning_tokens = 8000, empty answer). Fix: 32k budget for extraction tasks.
- Status: accepted. Ceiling for the lab = 6 concurrent calls shared by all processes using the key.
- PL-7b (same cycle): the first fix used a per-process semaphore, but three tools ran at once → up to 18 in flight → 78 × HTTP 429
  in 10 min. Root cause: the cap is per key, the limiter per process. Fix: cross-process slots (`flock` on 6 lock files).
  Verified: 3 processes × 20 threads → max 6 concurrent slots (test in session log); no 429 since.

## PL-8 Deep-read quote verification failed for 5/25 papers for *both* extractors
- Symptom: verified-quote fraction 0.05–0.37 for LIT02-065, LIT02-047, LIT02-128, LIT05-046, LIT01-040 for both models; others 0.86–1.0.
- Key regularity: failure was per *paper*, not per model → suspect the shared input, not the agents.
- Evidence: unverified quotes were correct sentences; in the `pdftotext -layout` text their words were interleaved with the
  neighbouring column ("we analyze two auto-regressive decoder**normalization, aℓ,j ∈** only gpt lms…").
- Root cause: layout mode preserves 2-column geometry line by line. Fix: reading-order extraction (no `-layout`) + verification
  against both texts; no new API calls (`deep_read.py --reverify`). Effect: all 25 papers ≥ 0.76, pooled verified 0.964 (DeepSeek,
  776 quotes) / 0.951 (MiniMax, 892 quotes). Status: accepted.
- General rule: when independent agents fail on the same item, inspect the shared pipeline first.

## PL-9 Long API generations failed at exactly ~300 s
- Symptom: MiniMax hypgen/audit calls errored at 300.3–301 s even after raising the client timeout to 1800 s; a DeepSeek audit
  returned an empty answer with completion_tokens = reasoning_tokens = 32 000.
- Root causes: (a) server-side nginx gateway timeout of 300 s for non-streaming responses (HTTP 504 page); (b) hidden reasoning
  consumes the whole `max_tokens` budget on 25k-token audit prompts.
- Fixes: SSE streaming in `llm_pool` (connection stays alive; verified on both models); audits use 60k tokens + `reasoning_effort: medium`.
- Status: accepted (streaming); budget rule added to the skill.

## PL-10 Queue logged failed runs as exit=0
- Symptom: E007 hit the RSS guard (rc=3 in process_metrics.jsonl) but `queue_cycle2.log` said `exit=0`.
- Root cause: `echo "[$(date …)] … exit=$?"` — the command substitution runs first and resets `$?`.
- Fix: `rc=$?` captured right after the run. Status: accepted (cycle-1 queue logs may contain the same error; process_metrics.jsonl is authoritative).

## PL-11 RSS guard converted a would-be sandbox reset into a clean failure
- E007 FULL at T=2048 grew past 6.8 GB; the in-process guard (PL-6) exited with rc=3, logged, and the queue continued. No reset
  in cycle 2 (vs 4 in cycle 1) with up to 3 concurrent API tools + 1 model process. Evidence that the guard, not luck, prevented
  the reset: the guard log line at 10:32:57Z with RSS 6.81 GB.

## PL-12 Prefix cache for suffix-only interventions (cycle 3)
- Change: `Runner.prefix()` computes layers 0–14 once; `Runner.suffix(st, n, hooks)` evaluates n conditions on layers 15–34 at
  the final position (kv_store / resid / attn hooks). Valid for any intervention at layer ≥ 15 or on the stored K/V.
- Equivalence (E000_prefix_equiv): max|Δlogit| 4.9e-5 (T=10), 0 (T=700) vs `forward(last_only_from=15)`; intervention effects 2–38 logits.
- Speed (identical workload, 3 conditions): T=10 4.19 s → 0.88 + 3.89 s (no gain: suffix dominated by per-call fp32 weight upcast);
  T=700 46.9 s → 13.0 s once + 3.0 s (≈ 1 s/condition vs 15.6 s). Status: accepted for long-context experiments (E008).

## PL-13 First forward after sandbox restart: RSS 10.35 GB → RSS guard (root cause: whole-file mmap residency)
- Symptom: E000 died twice at the first forward (RSS 10.35/10.36 GB, 13–25 s CPU); same code peaked at 6.15 GB in cycle 2.
- Hypothesis 1 "concurrent 10 GB base download" — **refuted** (reproduced with nothing else running).
- Bisection: load 5.61 GB → first forward → 10.31 GB. 10.3 ≈ 5.6 + 4.7 GB (PLE table). Microtest: np.memmap of the PLE
  tensor, reading **one** row: RSS 0.07 → 4.77 GB in 13 ms (no real I/O).
- Root cause: in the restored sandbox (overlay root FS under gVisor) a file mmap becomes fully resident on first touch; the
  loader's design assumption (mmap materialises only touched rows) is a property of the host FS, not guaranteed.
- Fix: `PreadEmbedding` — PLE rows read with `os.pread` (17.9 KB/token). Verified bit-identical (E006 pair ld 37.920166015625 /
  −35.74284744262695 before = after); RSS after forward 6.19 GB. Status: accepted. Lesson: don't rely on mmap laziness in sandboxes.
