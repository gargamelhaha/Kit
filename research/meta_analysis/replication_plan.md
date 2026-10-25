# Replication and new-experiment plan (ranked) — Gemma 4 E2B and small CPU-trainable architectures

Owner: Synthesis & Meta-Analysis role (proposal only; the coordinator decides, and every experiment needs a `PREREG.md` before data collection and an independent audit before any status above EXPERIMENTALLY_SUPPORTED).
Inputs: `LITERATURE_MAP.md`, `meta_analysis/methods_table.md`, `meta_analysis/findings_map.md`, LIT-02 §8, LIT-01 §8, LIT-03 §7–8, LIT-05 §8.
**Already running/done — not re-proposed:** E001 (KV-sharing inertness), E002 (logit lens), E003 (induction heads, mean-ablation vs random heads), E004 (counterfactual residual patching, landmark→city). Follow-ups to them are proposed below.
Ids E006… / A001… / M001 are provisional (E005 is taken by `E005_mem_rootcause`).

## Conventions and honesty notes
- **Cost unit.** 1 batched forward of ~300 tokens (e.g. 8×40) on this box ≈ **30–60 s** (ledger: 64.3 s full, 29.8 s with `last_only_from=15`). "FB" = one such batch. Costs are my arithmetic from that figure and unmeasured otherwise; backward-pass cost is *not* measured (assumed ≤3× forward). Implementation time is not counted.
- **Scoring (my heuristic, not a source).** `score = decision-impact (1–5) × discriminating power (1–5) / mid-estimate hours`. It rewards cheap experiments; it is a tie-breaker, not a verdict. Small-architecture runs score high mainly because they are cheap; they are ranked within their own track and interleaved by CPU availability (one model process at a time; tiny runs steal CPU from E2B runs, so run them in E2B-idle gaps).
- **Thresholds marked "(proposed)"** are my suggestions to be fixed in `PREREG.md` before data; they are not from a source.
- **Statistics:** bootstrap CIs over prompts/seeds; ≥30 prompt pairs for E2B follow-ups (E004 uses ≤12, which gives wide CIs); report all conditions, not only the successful one; no pooling across different metrics (see `findings_map.md`). Always record dtype, template on/off (LIT02-127), BOS, seeds, tokenizer single-token checks (LIT-02 §8).
- **Reading dependencies** are listed per experiment (see `LITERATURE_MAP.md` (d)). A design that rests on an abstract-only source is flagged.

---

## Ranking overview

Ranks are **within track**, by the score below (cost-sensitive; Gemma track is the primary mission, Track A is the second programme and is cheap). The last column is the recommended global slot, which interleaves tracks because CPU is shared (see "Suggested schedule").

| track rank | id | experiment | cost (mid) | score | gate / depends on | global slot |
|---|---|---|---|---|---|---|
| **Gemma 1** | **E006** | K/V(13,14) shared-copy patching: where does context→answer information cross? | ~0.4 h | 50 | E004 pair set | 2 |
| **Gemma 2** | **E007** | PLE patching/ablation: does residual-only patching under-estimate? (H-006) | ~0.75 h | 21 | E004 pair set | 3 |
| Gemma (free) | **E014** | Weight-level check of "untrained K/V remnants" (H-007) | ~0.15 h | — | none (no forward) | 1 |
| **Gemma 3** | **E008** | Attribution patching vs exact patching on E2B | ~1 h | 20 | E004 exact sweep results | 5 |
| **Gemma 4** | **E009** | Component necessity in KV-shared layers 15–34 (attn vs MLP vs PLE) | ~1.25 h | 13 | E004 pair set; E001 runner | 6 |
| Gemma 5 | **E010** | Chat-template sensitivity of E002/E004 | ~1 h | 9 | E002/E004 outputs | 11 |
| Gemma 6 | **E011** | Validate community E2B SAEs (FVU, ΔCE, L0 vs random decoder) | ~3 h | 5 | **gate for all SAE/feature work**; control run on Gemma 2 2B | 13 |
| Gemma 7 | **E012** | Refusal direction: single vs multi, template on/off | ~1.25 h | 4.8 | read LIT02-088 | 14 |
| Gemma 8 | **E013** | Tuned vs logit lens on a few layers | ~1.25 h | 4.8 | E002; read LIT02-047 | 15 |
| **Arch 1** | **A001** | Optimizer/arch claims vs tuning protocol at tiny scale (LIT-05 exp. 3) | ~0.5 h | 32 | none; builds the shared tiny-training harness | 4 |
| Arch 2 | **A002** | HRM-style hierarchy vs compute-matched refinement (LIT-05 exp. 2) | ~0.5 h | 24 | A001 harness | 7 |
| Arch 3 | **A003** | Recall vs state size on MQAR (LIT-05 exp. 1) | ~0.5 h | 24 | A001 harness; read LIT05-013 | 8 |
| Arch 4 | **A004** | Self-correction vs equal-compute sampling (LIT-05 exp. 4) | ~0.5 h | 24 | A001 harness | 9 |
| Arch 5 | **A005** | Plasticity loss in a tiny continual learner (my extension, not in LIT-05 §8) | ~0.5 h | 18 | A001 harness | 12 |
| Arch 6 | **A006** | Diffusion vs AR under repeated data (LIT-05 exp. 5) | ~0.5 h | 12 | A001 harness | 16 |
| Process | **M001** | Seeded-fault audit of the auditor step | ~1 h agent time | 16 | code of completed experiments | 10 |

**Overall top five (my judgement):** E006, E007, A001, E008, E009 (E014 is free and goes first). E011 is expensive but gates all dictionary work, so schedule it deliberately rather than by score.

Contingent follow-ups (only if the parent experiment gives the stated result): E003-F1 (composition test for induction), E004-F1 (30-pair rerun if E004 is under-powered), IOI-style head patching with AtP prefilter (needs E008), SAE-feature steering (needs E011).

---

## Track G — Gemma 4 E2B

### E006 — K/V(13,14) shared-copy patching (follow-up to E004 and H-003)
- **Hypothesis.** Two rival accounts of how subject information reaches the answer. **H_late:** the answer-relevant subject information is read from the subject token's K/V (layers 13/14) by late attention at the final position (the "late extraction" site of LIT02-065/LIT02-002 reinterpreted for KV sharing). **H_early:** the information has already been moved into the final position's residual by layer 14, so late attention contributes little.
- **Protocol.** Use E004's filtered pairs (extend to ≥30; keep clean top-1 = answer filter; verify the final token is identical in clean/corrupt, otherwise patch PLE inputs too). Cache the clean run's shared K/V written by layer 13 (read by sliding layers 15–18, 20–23, 25–28, 30–33) and layer 14 (read by full layers 19, 24, 29, 34). In the corrupt run, patch **only the copy read by layers ≥15** (via `shared_kv_states`), leaving layers 13/14's own attention untouched; second condition also patches their own use. Conditions: (a) last-subject-position, (b) all subject-token positions, (c) post-subject template positions, (d) K only vs V only, (e) layer-13 copy only vs layer-14 copy only, (f) **positive control**: all positions' shared K/V + final-position resid_pre[15] ⇒ expected recovery 1 up to fp error (check ~3.6e-5, as in E001), (g) negative control: shared K/V from a third prompt with a different answer.
- **Metric.** LD recovery as in E004 (logit(clean answer) − logit(corrupt answer), normalised), pre-softcap and post-softcap LD both reported.
- **Predictions (proposed).** H_late: (a)/(b) recovery median ≥0.5 and (c) <0.2, (f) ≈1, (g) |recovery| <0.1. H_early: (a)/(b) <0.2 while E004's final-position resid_pre[15] recovery ≥0.8. Intermediate values are reported as "mixed", not forced into either.
- **Controls.** (f), (g), random-position patch, patch with K/V from the corrupt run itself (null).
- **Cost.** ~8 conditions × 30 pairs = 240 item-runs ≈ 30 batches ≈ 15–30 min (full forwards; layers 0–14 can be shared across conditions if cached, then cheaper).
- **Value of information.** Decides whether E2B has a *clean architectural bottleneck* for context→answer information at two layers; if so, all later cross-position circuit work can patch two K/V tensors instead of sweeping residuals over 15 layers × positions, and H-003 becomes testable at its source. If H_early holds, the "late extraction" part of factual-recall staging does not transfer to E2B. Novelty must be checked: no interpretability study of cross-layer KV sharing was found (LIT-03; absence is not proof) — read CLA/YOCO (LIT03-019, LIT03-020) and the Gemma 3n docs before claiming novelty.

### E007 — PLE patching/ablation (H-006)
- **Hypothesis.** Because PLE re-injects token identity into every layer, residual-only patching at layer L leaves corrupt token identity in layers >L and under-estimates the effect of the subject token. **Alternative:** PLE carries only low-level lexical information and the gap is ≈0 for answer-relevant content.
- **Protocol.** E004 pairs (≥30). Conditions at the last-subject position for L=0…14: (A) resid-only (reuse E004), (B) resid + clean `per_layer_inputs` for layers ≥L at that position, (C) PLE-only patch (clean PLE inputs, no residual patch). Separate ablation arm on ~30 neutral sentences: zero / mean-over-tokens / token-shuffled `per_layer_inputs` for layers ≥15, ≥L for several L, and all layers; measure ΔNLL. HF accepts `per_layer_inputs` only together with `inputs_embeds`, and the lookup needs `input_ids` (LIT-03 §4.2): implement via the runner, not by monkey-patching the embedding.
- **Metric.** Recovery (A vs B vs C) per L with CIs; ΔNLL per ablation type.
- **Predictions (proposed).** B ≥ A at all L ≤14 and the gap grows with L; C >0 and increasing with L; PLE ablation ΔNLL exceeds that of an equal-norm random perturbation of the same channels.
- **Controls.** Patch PLE from a third prompt; random-channel perturbation with equal norm; same-token patch (no change expected).
- **Cost.** (B) 30 pairs × 15 L = 450 + (C) ~30 + ablations ~150 item-runs ≈ 80 batches ≈ 40–80 min.
- **Value of information.** If the gap is large, **every residual-only conclusion in E2B (E004, steering, refusal ablation, probes) needs PLE patched alongside**; protocol change for the whole lab. If ≈0, residual-only patching is validated and PLE can be dropped from future designs. Also answers OPEN_QUESTIONS #3 in part.

### E008 — Attribution patching vs exact patching on E2B (process-speed experiment)
- **Hypothesis.** First-order attribution patching (gradient × activation difference; EAP-IG variant) ranks E2B sites like exact patching at layers ≤14 and at the final position for layers ≥15; it fails (false negatives) at sites with saturation (e.g. near the softcap) — the failure mode reported in LIT02-013 (abstract-level, single preprint).
- **Protocol.** Use E004's exact per-site recovery as ground truth (same pairs; if E004 stored only aggregates, rerun a subset). For each pair compute one corrupt forward + backward of LD w.r.t. resid_pre at all sites and w.r.t. K/V of layers 13/14; AtP estimate = (clean − corrupt activation)·gradient. Variants: pre- vs post-softcap metric; IG with ~5 steps. fp32 compute, weights frozen.
- **Metric.** Spearman ρ and top-10 site recall per pair; sign agreement; regression slope AtP vs exact; bootstrap CI over pairs.
- **Predictions (proposed).** Median ρ ≥0.8 and top-10 recall ≥0.8; falsified if median ρ <0.5. 
- **Controls.** Site-shuffled permutation null; gradient of an unrelated metric (clean-answer logprob only); layer-matched random ordering.
- **Cost.** Plain AtP: 30 pairs × (1 fwd + 1 bwd ≈ 4 forward-equivalents) ≈ 120 item-equivalents ≈ 15 batches ≈ 8–15 min if batch 8 fits for backward (otherwise ×2–4); the IG variant (×5 steps) ≈ 1–1.5 h. Mid estimate ~1 h. Backward memory/time on CPU unmeasured — pilot one pair first.
- **Value of information.** For a 40-token prompt an exact sweep needs ≈ 15 layers × 40 positions + 20 final-position sites ≈ 620 patched forwards; AtP needs ~1 forward + 1 backward (≈150–200× fewer forward-equivalents) [derived]. If AtP is reliable it removes the lab's slowest recurring step (30–60 s per forward) for all later sweeps and head/neuron-level screening; if not, we budget exact sweeps only at targeted sites. Reading dependency: LIT02-008, LIT01-090, LIT02-013 are abstract-only.

### E009 — Component necessity in the KV-shared layers 15–34 (open question 2)
- **Hypothesis.** Layers 15–34 act per position on fixed K/V; **H_readout:** for landmark→city answers, single-layer ablation of the **attention** output matters less than ablating the same layer's **MLP** output (median over layers) — consistent with late layers doing answer formation rather than information routing. **Rival:** late attention does real extraction (as in LIT02-065) and attention ablations matter as much as MLP.
- **Protocol.** Final-position-only ablations (exact, other positions are inert by K1′) at each of layers 15–34: attention output (post-`post_attention_layernorm`), MLP output (post-`post_feedforward_layernorm`), PLE output (`post_per_layer_input_norm`). Ablation types: mean (over a reference set), resample (from another prompt), zero on a 5-layer subset only (zero is OOD given `layer_scalar`; report anyway because ablation type changes conclusions, LIT02-022). 12–30 prompts from the E004 set.
- **Metric.** Normalised Δ in logit difference (clean answer vs counterfactual) and Δ in answer probability; effect per component with CIs.
- **Controls.** Equal-norm random-direction ablation; ablating a randomly chosen layer-matched component; clean run.
- **Cost.** 20 layers × 3 components × 2 ablations × 12 prompts = 1,440 item-runs ≈ 180 batches ≈ 1.5–3 h conservatively; much less if layers 0–14 are cached once per prompt and only layers 15–34 at the final position are recomputed (unmeasured) [derived].
- **Value of information.** Tells where to look for circuits in 57% of the network (20 of 35 layers) and whether the late-attention extraction hypothesis from other models transfers. Decision: whether later effort targets late attention, late MLPs, or PLE.

### E014 — Weight check of H-007 (no forward pass)
- **Hypothesis.** The k/v tensors stored for KV-shared layers (60 tensors, absmax ≈0.06 vs 0.23–0.35 in trained `k_proj`) are untrained remnants. **Protocol.** Read the tensors directly from safetensors (one tensor at a time), compare per-tensor statistics with trained `k_proj`/`v_proj` of layers 0–14 and with a freshly initialised layer of the same shape; repeat for the base checkpoint if present. **Metric/controls.** Std/absmax/kurtosis, cosine between IT and base tensors, random-init reference. **Cost.** minutes, <1 GiB RAM if read tensor-wise (memory-map; do not load the model). **Value.** Cheap check of whether the checkpoint was ever trained in an unshared configuration; informs how literally to treat "20 layers share K/V" (decision: low-stakes, but free). *Note:* LIT-03 reports that layers 15–34 have no k_proj/v_proj in the code; reconcile with H-007's "60 stored tensors" first.

### E010 — Chat-template sensitivity of E002/E004
- **Hypothesis.** Template choice matters for internal results, as LIT02-127 reports for refusal (65/75 vs 0/75; single source). **Protocol.** Rerun E002's lens (62 prompts) and E004's key sites (last-subject all L≤14; final position L≥15; ≥12 pairs) with the E2B-IT chat template vs raw text with BOS; same filtered items. **Metric.** Shift in median first-answer-top-1 layer; change in the layer of maximal recovery; recovery curves with CIs. **Decision rule (proposed).** Template is a factor if the median first-top-1 layer shifts by ≥2 layers or the argmax layer of recovery shifts by ≥2. **Controls.** Raw text with and without BOS (BOS omission produced nonsense, PL-3). **Cost.** E002 ≈ 8 batches (≈4–8 min); E004 sites ≈ 50 batches (≈25–50 min). **Value.** Decides whether all raw-prompt results can be reported as "E2B-IT" results or only as "raw-prompt" results; cheap insurance against a systematic confound.

### E011 — Validate the community E2B SAEs before using them
- **Hypothesis.** `decoderesearch/gemma-4-saes` (E2B **base**, L6/L17/L28, width 65,536, BatchTopK k=100; LIT03-026) reconstruct E2B-IT activations and preserve next-token loss clearly better than a random-decoder baseline (LIT01-046). **Alternative:** base→IT transfer is poor, or hook/BOS/template conventions differ, or PLE/`layer_scalar` scale breaks the expected input norm (K5).
- **Protocol.** Step 0 (control): run the identical metric code on **Gemma 2 2B + a Gemma Scope SAE** at one layer and compare with the paper's reported numbers (needs a Gemma 2 2B download and a read of LIT01-100's metric definitions — disk and bandwidth not yet checked). Step 1: cache ~10k tokens of mixed text (ctx ≤128, BOS prepended) on E2B-IT, then base (one model at a time). Step 2: per layer FVU, L0, dead-latent fraction, ΔCE spliced vs clean and vs zero/mean ablation; baselines: random decoder with matched L0, shuffled activations. Step 3 (no training): fraction of latents aligned with single MLP neurons (LIT01-071 cosine thresholds). Step 4: template on/off. For L17 and L28 the splice affects only position-local computation, so layers 17–34 can be recomputed from cached L17 inputs and cached 13/14 K/V (≈ half a forward) [derived].
- **Metric.** FVU; ΔCE-recovered fraction = 1 − (CE_spliced − CE_clean)/(CE_zero − CE_clean); L0. Pass criterion to pre-register (proposed): ΔCE-recovered ≥0.8 and clearly above the random-decoder baseline at the same L0.
- **Cost.** 10k tokens ≈ 33 batches ≈ 17–33 min per full pass; ~7 passes per checkpoint (clean, 3 splices, 3 zero-ablations, partly half-cost) ≈ 1.5–3 h for IT; ≈ 3–6 h with base; plus step 0.
- **Value of information.** Gate for every feature-level analysis on E2B (steering with SAE latents, entity directions LIT01-063, neuron-aligned latents). If it fails, the lab drops dictionary methods for E2B (training SAEs on CPU is infeasible within budget; LIT-01 advises deferring transcoders/CLTs) and stays with patching, neuron basis and probes.

### E012 — Refusal direction: single vs multiple, template on/off (tests contested C5)
- **Hypothesis.** One difference-in-means direction mediates the refusal-token log-odds shift (LIT02-088) vs a k=1…5 subspace does better (LIT02-089, -090). **Protocol.** ~100 public harmful and ~100 harmless instructions; refusal-token log-odds and prefix match only — **no harmful completions generated or stored**. Direction at a layer sweep; activation ablation (layers ≤14 all positions, ≥15 final position only, per K1′) and addition; k=1…5; with/without chat template. **Metric.** Δ refusal log-odds. **Controls.** Random direction, label-shuffled direction. **Cost.** collection ≈ 25 batches (13–25 min); ablation conditions ≈ 60 batches (30–60 min). **Value.** Adjudicates a contested claim on a new architecture and quantifies the template confound (S1); lower impact on core lab questions. Reading dependency: LIT02-088 abstract-only — read first. Safety-relevant content: keep to logits.

### E013 — Tuned vs logit lens on a few layers
- **Hypothesis.** Affine/tuned translators at 3 layers (e.g. 4, 10, 14) agree with the final distribution much better than the plain logit lens at early layers, and the two coincide late. **Protocol.** 20k tokens of activations at the 3 layers (≈123 MB per layer in fp32); train per-layer affine translators on KL to the final softcapped distribution; evaluate on held-out tokens. **Metric.** KL and top-1 agreement vs identity (logit lens) and random-init translators; E002 items re-decoded. **Cost.** 20k tokens ≈ 67 batches ≈ 35–70 min + minutes of training. **Value.** Decides whether E002's early-layer statements (answer rank at layers ≤5 etc.) are interpretable or lens artefacts; low priority if E002's claims are restricted to late layers. Reading dependency: LIT02-047 abstract-only.

---

## Track A — small CPU-trainable architectures (from LIT-05 §8, plus one extension)

Common caveats: LIT-05's estimate of ≤30 CPU-min per experiment is the scout's and was **not benchmarked** (run a timing pilot, cut seeds if a run exceeds ~2 min); sources are abstract-level; seeds ≥3 with 95% CIs; every arm gets the same tuning budget (the core lesson of LIT05-081, LIT05-067, LIT05-150); evaluate at the end of the schedule, not mid-training (LIT05-081 warns rankings flip); label-shuffled sanity run in each. Run one process at a time and outside E2B runs. A001 should produce the shared harness (tiny GPT/recurrent module, LR-grid runner, seed/CI reporting, FLOP accounting) reused by A002–A006.

### A001 — Optimizer speedup vs tuning protocol (LIT-05 exp. 3)
- **Hypothesis.** Muon's headline ~2× over AdamW (LIT05-080) shrinks to 1.0×–1.5× when both are tuned (LIT05-081-like) while an untuned-vs-tuned comparison shows >1.5×. **Protocol.** ≈1–2M-parameter GPT on a small fixed corpus; AdamW vs Muon (Newton–Schulz, weight decay, update scaling as in LIT05-080), 5-point LR grid each, 3 seeds, 1–2k steps; control arm: AdamW at Muon's best LR. **Metric.** Tokens to a pre-set end-of-training validation loss L\*. **Falsified if** tuned-vs-tuned speedup ≥1.8×. **Cost.** ~30 CPU-min. **Value.** Fixes the lab's default optimizer and tuning protocol for all later tiny-architecture work, and is a direct test of the "gains shrink under tuning" pattern (F10). Reading dependency: LIT05-080 abstract-only.

### A002 — HRM-style hierarchy vs compute-matched refinement (LIT-05 exp. 2)
- **Hypothesis.** The hierarchy adds little beyond weight-tied refinement with deep supervision (LIT05-150, -047, -048 vs LIT05-045). **Protocol.** Procedural 6×6 (or 4×4) mini-Sudoku, ~0.3–0.5M params; arms: (a) two-level H/L recurrence with deep supervision, (b) TRM-style single-network recursion, (c) parameter-matched one-pass Transformer, (d) Transformer with the same number of weight-tied refinement passes and deep supervision (compute-matched). Split by puzzle (no leakage), 5 seeds. **Metric.** Held-out exact match vs training FLOPs. **Predictions (LIT-05).** d ≈ a within 3 pp; a, b, d > c. **Falsified if** a exceeds d by ≥5 pp with non-overlapping CIs. **Cost.** ~30 CPU-min. **Value.** Separates "architecture" from "outer loop + augmentation" on a controlled toy; informs whether recursive-refinement learners are worth building for later from-scratch work. Reading dependency: LIT05-046 and LIT05-045 abstract-only.

### A003 — Recall vs state size (LIT-05 exp. 1)
- **Hypothesis.** Gated-delta > plain linear attention at matched state size; a 3:1 linear:attention hybrid is within 5 pp of attention (LIT05-008, -013, -022, -025). **Protocol.** ~0.5–1M-param, 2-layer, d=64 models (softmax attention, plain linear attention, gated delta-rule recurrence in naive PyTorch, 3:1 hybrid) on MQAR with sequence length ≤128. **Metric.** Exact accuracy vs number of KV pairs, 3 seeds. **Controls.** Parameter-matched; label-shuffled; fixed length. **Falsified if** hybrid >5 pp below attention with non-overlapping CIs or delta ≈ plain linear. **Cost.** ~30 CPU-min (time pilot first). **Value.** Chooses the sequence mixer for later tiny learners; checks F8's mitigation claims at toy scale. Reading dependency: LIT05-013 abstract-only.

### A004 — Self-correction vs equal-compute sampling (LIT-05 exp. 4)
- **Hypothesis.** Without an external verifier, self-refine gives ≤2 pp over equal-compute majority vote; a separately trained verifier helps in proportion to its generation–verification gap (LIT05-112; contested C20). **Protocol.** Tiny Transformer on 3-digit addition/sorting; arms: self-refine (trained on self-generated corrections), best-of-n by own likelihood, best-of-n with a trained verifier, oracle verifier (upper bound), random pick (lower bound), majority vote at equal n; 5 seeds. **Metric.** Accuracy vs sample budget. **Falsified if** self-refine beats equal-compute majority vote by ≥5 pp without an external verifier. **Cost.** ~30 CPU-min. **Value.** Decides whether "self-reflection" components deserve a place in the lab's own learners or whether compute is better spent on sampling plus verification.

### A005 — Plasticity loss in a tiny continual learner (extension; not in LIT-05 §8)
- **Hypothesis.** A small network trained on a sequence of synthetic tasks loses ability to learn later tasks (plasticity loss, LIT05-140), and a simple mitigation (e.g. continual-backprop-style reinitialisation or L2-to-init) restores it; whether this transfers to LLM scale is argued only at abstract level (LIT05-138, -139). **Protocol.** 20 sequential synthetic classification/regression tasks; arms: baseline Adam, + weight decay, + a continual-backprop-style mitigation; 3–5 seeds. **Metric.** Accuracy/loss on task k after fixed steps vs k. **Controls.** Fresh-initialised network per task (upper bound), shuffled task order. **Cost.** ~30 CPU-min. **Value.** The lab's second programme aims at systems that learn from scratch in arbitrary environments; the registry has only 3 "continual" titles and nothing on local-learning rules, predictive coding or neuroevolution (see `LITERATURE_MAP.md` (c)). This is the cheapest probe that tells whether a lifelong-learner harness must include plasticity maintenance. Weak literature basis: read LIT05-140 before designing.

### A006 — Diffusion vs AR under repeated data (LIT-05 exp. 5)
- **Hypothesis.** AR wins at 1 epoch; masked diffusion overtakes by ≤64 epochs on a small corpus (LIT05-062, -063). **Protocol.** 1–2M-param masked-diffusion vs AR Transformer, same width, ~2–5 MB text, 1/4/16/64 epochs. **Metric.** Held-out AR NLL vs diffusion ELBO (ELBO upper-bounds NLL; ELBO > NLL results are inconclusive). **Controls.** LR sweep (3 values), 3 seeds, matched dropout/weight decay. **Falsified if** AR NLL stays below diffusion ELBO at 64 epochs with non-overlapping CIs and no overfitting-control explanation applies. **Cost.** ~30 CPU-min (scale tokens to fit). **Value.** Lowest decision impact: only matters if data-limited regimes are the lab's setting.

---

## Track M — research-process experiment

### M001 — Seeded-fault audit of the independent-auditor step
- **Hypothesis.** An auditor that shares weights with the author (same-model roles collapse, LIT04-143; evaluator self-preference, LIT04-109; missed 6.5% of confirmed hacks in LIT04-016) misses a substantial share of injected faults. **Protocol.** Take 2–3 completed experiment scripts/result files; inject N≈10 known faults (off-by-one in position index, wrong BOS handling, leaked clean activation into the "corrupt" run, metric sign flip, normalisation using the wrong denominator, silently reduced sample, post-hoc threshold change); give the auditor the methodology and raw outputs without the author's conclusion (LIT-04 §8b); log detections; compare "blind" vs "told the fault classes to look for". **Metric.** Detection rate by fault class with Wilson interval (n is small; report it as such). **Controls.** Unmodified artefacts (false-positive rate). **Cost.** ≈1 h of agent time, ~0 CPU. **Value.** Sets how much weight the audit gate may carry when upgrading statuses and which fault classes need automated tests instead of review. This is a lab-internal measurement; it cannot show independence between same-weights roles, only measure their miss rate.

---

## Suggested schedule (one model process at a time)
1. E014 (free), then E006 and E007 back-to-back on the E004 pair set (≈1–1.5 h total; reuse cached clean runs).
2. In the next E2B-idle window: A001 (builds the harness).
3. E008 (pilot one pair first; stop if backward does not fit in RAM) then E009.
4. A002, A003, A004 in further idle windows; M001 once E006/E007 code exists (the cleanest seeded-fault targets).
5. E010, A005, then E011 (long; start only with a free ~3 h block and after the Gemma 2 2B control is possible), A006, E012, E013 on demand.
All of the above need: `PREREG.md` committed first, an auditor pass on the runner code (K/V override and `per_layer_inputs` paths are the highest-risk implementations), commit and push after each step (PL-4).
