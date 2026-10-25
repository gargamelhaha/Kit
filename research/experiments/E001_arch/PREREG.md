# E001 — Architecture verification and the KV-sharing prediction
Registered 2026-10-25 (re-registered verbatim after the sandbox loss; the original run's numbers are quoted in
PROCESS_LEDGER as unarchived and are superseded by this run).

Source facts: config `num_kv_shared_layers=20` (35 layers); transformers 5.20.2 `Gemma4TextAttention` gives layers
>= 15 no k/v projections and makes them read `shared_kv_states[layer_type]`, stored by the last non-shared layer of the
same type (13 sliding, 14 full).

**P-ARCH-1 (deterministic prediction).** Adding large noise to resid_pre at all non-final positions of any layer >= 15
leaves final-position logits *bit-identical*; the same noise at layers <= 14 changes them.
**P-ARCH-2.** Running layers >= 15 on the final position only reproduces final logits (max |Δ| < 1e-3) and is faster.
**P-ARCH-3.** The custom Runner reproduces HF `model()` final logits (max |Δ| < 1e-3).
**Also recorded.** Statistics of checkpoint k/v tensors stored for KV-shared layers (unused by the model code).
