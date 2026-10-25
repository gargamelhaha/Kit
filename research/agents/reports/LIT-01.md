# LIT-01 report — dictionary learning, SAEs, transcoders, crosscoders, attribution graphs

Role: Literature Scout + Evidence Extractor. Date: 2026-10-25. Deliverables: `research/evidence/LIT-01_sources.csv` (128 rows), `research/logs/search_LIT-01.jsonl` (111 lines), this report.
Source IDs are `LIT01-nnn` (the CSV is authoritative). Claims are attributed to authors ("authors report"), not asserted as established.

## 1. What was done

- Searched arXiv (API), OpenAlex (API, until its free daily budget ran out), Semantic Scholar (API, citation lists), DataCite (DOI check) and the web (WebSearch/WebFetch), covering the topics in the task packet.
- Built a 128-row source table:
  - 60 rows have relevance ≥4.
  - 30 rows are flagged critique/negative/replication.
  - 9 rows are from 2022–2023, 32 from 2024, 30 from 2025 and 57 from 2026.
- Ran backward and forward citation search for 4 anchors, each in both directions:
  - Transcoders (LIT01-068)
  - Sparse Feature Circuits (LIT01-069)
  - Gemma Scope (LIT01-100)
  - Scaling & evaluating SAEs (LIT01-012)
  - Toy Models (LIT01-001) was forward only.
- Read 28 sources partially, using WebFetch extraction of arXiv HTML or site pages. These include the key critiques and the Anthropic circuit-tracing/biology papers.
- **Incident (sandbox reset):** mid-task the sandbox was reset and my scratch files and first log were lost.
  - I re-verified every arXiv id (131 DataCite DOI lookups, bulk arXiv `id_list` calls) and re-ran the citation and topical searches.
  - The log marks entries rebuilt from my transcript with `"reconstructed": true`.
  - Reconstructed entries have approximate timestamps and sometimes `n_results: null`.
  - The pre-reset OpenAlex data was not re-fetched (see the OpenAlex budget limit below). Its counts rest on what I saw in tool output.
- **OpenAlex limit:** OpenAlex returned HTTP 429 ("free daily budget used up for shared IP") for the rest of the task. I did not create an API key, because that would need a user email. Backward/forward search therefore used Semantic Scholar, which returns at most 100 rows per call, unsorted by relevance.

## 2. Sources / files checked

- All 128 CSV rows. Read depth: **28 partial, 93 abstract_only, 7 metadata_only, 0 full_text.**
- No source was read as verbatim full text. "partial" means a small-model extraction of the page or arXiv HTML; I did not read the full text myself.
- Every arXiv id (112 rows) was confirmed by an arXiv API call, with title, authors and v1 date taken from it. Its DOI `10.48550/arXiv.<id>` was confirmed by DataCite. Venue status comes only from arXiv `journal_ref` or from OpenAlex output I saw before the budget ran out.
- Web-only rows have no DOI, because none was verified: Anthropic Transformer Circuits pages, the GDM blog post, Gemma Scope 2 and Hugging Face repos. They carry a URL.
- Not readable or not read:
  - The Gemma Scope 2 technical paper PDF (unreadable binary in WebFetch).
  - The OpenReview "Use SAEs to Discover Unknown Concepts…" page (blocked by a browser-verification page; only the abstract snippet is known).
  - `arxiv.org/html/2209.10652` returned 404; I used the transformer-circuits.pub version of Toy Models.
  - "Progress on Attention" and "Towards Monosemanticity" (search snippets only).

## 3. Actions actually run

| Catalog | Log lines | Results returned | Note |
|---|---|---|---|
| arXiv API | 25 live + 23 reconstructed | 399 live | Topical queries, `id_list` verification, version/journal_ref pass; several HTTP 429s retried with backoff |
| OpenAlex | 12 reconstructed | 143 | Title lookups, DOI-batch lookup, 8 topical searches; all pre-reset; budget exhausted afterwards |
| Semantic Scholar | 11 live | 805 rows | 9 successful calls: forward and backward lists for 4 anchors, plus forward for Toy Models; 2 RATELIMIT failures were retried |
| DataCite | 3 live | 131 DOI lookups | All resolved |
| WebSearch | 6 reconstructed | 48 | |
| WebFetch | 9 live + 22 reconstructed | | Includes 1 HTTP 404, 1 unreadable PDF and 1 blocked page |

Failed requests that returned an error were mostly not logged.

## 4. Results

### 4.1 Thematic map (source IDs in parentheses)

1. **Superposition theory.**
   - Toy Models (001) report that sparsity is necessary for superposition, and describe feature geometries and limited computation in superposition. The authors say it is unclear what generalises to real networks.
   - A 2024–2026 theory cluster (e.g. 002–008) extends this: computation in superposition, spectral and frame-theoretic geometry, and compressed computation. I read abstracts only.
2. **SAEs as the working tool.**
   - Foundational: Cunningham (011), TopK scaling (012), Gated (013), JumpReLU (014), BatchTopK, Matryoshka, and Claude 3 Sonnet scale (022).
   - Public weights: Gemma Scope for Gemma 2 (100), Llama Scope (101), and Gemma Scope 2 for Gemma 3 (121).
3. **SAE evaluation.**
   - SAEBench (028), ground-truth board-game evaluation (029), supervised evaluation (030), and the 2026 audit of the benchmarks themselves (032).
   - Autointerp and intervention-identification critiques (033, 035, 109).
4. **SAE critiques and negative results.**
   - Random baselines (046, 047), probing (048), steering (049, with a rebuttal in 050), seed dependence (052, 053), canonical units (051), absorption (054), and hedging (055).
   - The GDM negative-results post (120).
   - Do SAEs find reasoning features (065), and do they generalise (066).
5. **Transcoders, cross-layer transcoders and attribution graphs.**
   - Transcoder circuits (068), sparse feature circuits (069), the Circuit Tracing methods paper (115), Biology of an LLM (116), the QK-attribution paper (117), and the circuit-tracer library (122).
   - Open replications on small models (073, 074, 075) and the "neurons in disguise" critique (071).
6. **Neuron-basis and component circuits as competitors.** RelP/neuron-basis circuits (072) and circuit consistency/specificity (076).
7. **Crosscoders and model diffing.** Crosscoders (114), the Anthropic diffing note (118), later variants (2026), and LLM-baseline competitiveness (094).
8. **Method-level meta-critique.**
   - Identification assumptions (043).
   - Circuit extraction and comparison choices (077, 078, 105–111).
   - These are mostly 2026 abstract-level items.
9. **Gemma 4 E2B tooling.** Community SAEs exist (124–127) but their quality is unverified. Gemma Scope 2 covers Gemma 3, not Gemma 4.

### 4.2 Methods table

| Method | What it measures / provides | Known limitations (as reported by authors) | Key refs |
|---|---|---|---|
| SAE (ReLU/TopK/JumpReLU/Gated/Matryoshka) | Sparse linear dictionary of activations | Seed dependence (42% shared in one setting); absorption/splitting; dark matter; dictionary not unique; random baselines competitive on several metrics | 011, 012, 014, 028, 046, 051, 052, 054 |
| Transcoder / skip-transcoder | Sparse replacement for an MLP's input→output map | Many features align with single neurons (late layers); first-order faithfulness only | 068, 070, 071, 121 |
| Cross-layer transcoder + attribution graph | Prompt-specific replacement model with linear edges | Attention not explained; error nodes; hypotheses not proofs; graphs informative for ~25% of prompts tried | 115, 116 |
| QK attribution | Attention scores as feature-pair interactions | Qualitative only; no RoPE handling; competition among positions unexplained | 117 |
| Crosscoder / diffing | Shared and exclusive features across models | Exclusive features denser and more polysemantic; simple LLM baselines competitive | 114, 118, 094 |
| Neuron-basis circuits (EAP-IG, RelP) | Circuits over MLP activations | Circuits still large; component circuits consistent but not task-specific | 072, 076 |
| Probing, DiffMean, prompting baselines | Dense supervised alternatives | They need known concepts; prompting wins on steering in AxBench | 048, 049, 120 |
| Random-component and synthetic ground-truth checks | Sanity checks for SAE validity | Synthetic features assumed independent; only standard SAEs tested | 032, 046, 047 |

### 4.3 Confirmed vs contested

**Reported by several independent sources** (they are not independent replications of one experiment; each used a different setup):

- SAE features depend on the random seed:
  - 052: 42% of latents shared in one Pythia-160M setting; 30% in a Llama-3-8B FFN setting.
  - 053: mean reappearance ~0.75 over 96 seeds.
  - 051: stitching, composition and meta-latents show non-atomic dictionaries.
  - 010 reports instability theoretically (abstract only).
- Absorption and splitting exist in the SAEs tested: 054 reports absorption in every LLM SAE tested.
- Reconstruction and proxy metrics do not guarantee feature quality:
  - 028 and 032 report that SAE proxy metrics are poor predictors of downstream results.
  - 046 and 047 report that random baselines and random transformers score close to trained ones.
- Attribution graphs are approximate hypotheses. The Anthropic authors themselves list missing attention circuits, error nodes and unfaithfulness risk (115, 116).

**Contested or unresolved:**

- SAE utility for probing and steering:
  - Negative: 048, 049, 120.
  - A rebuttal of 049 reports that supervised feature labelling makes SAE steering competitive with LoRA on some forums, though prompting is still stronger (050, no exact numbers in my extraction).
  - 128 argues for a "discover, not act" split (snippet only).
- Transcoder value:
  - 070 (abstract) says transcoders beat SAEs.
  - 071 reports many transcoder features are neurons, and that thresholded neurons can serve as a circuit basis.
  - 072 reports neuron-basis circuits are competitive.
  - 076 reports a consistency/specificity trade-off for both components and neurons.
- Poetry-planning mechanism:
  - 116 reports newline-localised planning in Claude 3.5 Haiku.
  - 073 does not recover the newline mechanism in small open models (the author says this is not a disproof).
  - 074 reports latent planning that grows with scale in Qwen-3, but weak backward planning.
  - 075 (abstract only) tests rhyming planning across scales.
- Model diffing: 094 reports that an LLM-text baseline is competitive with SAE-based diffing on its tasks.

### 4.4 Open problems

- Ground truth for real-LLM features and a validated SAE-quality metric. 032 says the best audited metric still cannot distinguish similar variants.
- Whether features are the right unit, versus neurons, subspaces (053), manifolds or non-linear features (abstract-level 2026 items).
- Attention and QK circuits at scale (117).
- Faithfulness of replacement models beyond first-order Jacobians (071).
- Causal identification and stability of circuit claims (043, 076, 077, 078).
- Feature interactions and composition (051; PolySAE abstract).
- Coverage of newer architectures: for Gemma 4 E2B only unverified community SAEs were found, and no transcoder or crosscoder was found in my searches.

## 5. Evidence available

- `research/evidence/LIT-01_sources.csv` — 128 rows, fields as specified. Each row has `read_depth`, the claims source and limitations. For abstract_only rows, `key_claims` is mechanically extracted from the arXiv abstract and prefixed "[abstract-level]".
- `research/logs/search_LIT-01.jsonl` — 111 lines.
- Unsaved scratch (in `/tmp/lit01`, not deliverables): Semantic Scholar citation lists for 9 anchor/direction pairs, plus store, metadata and DataCite check files.

## 6. Remaining limitations and saturation statement

- **Topical keyword search is approaching saturation, by this count only.** The last 6 arXiv topical queries returned 50 results, and 4 were new relevant items (~8%, below the 10% threshold). Counting was manual.
- **Citation chasing is not saturated.**
  - Semantic Scholar returned 100 rows per call (unsorted).
  - Forward list of 068: 71 of 100 titles matched the keyword filter, and I added 12.
  - Forward list of 069: 8 added.
  - About 40 unseen keyword-matching titles in the forward lists of 012 and 001 were not triaged (e.g. "Beyond SAEs: Consistent Concept Discovery with Sparse Coding", "Shaped SAEs", "Symmetry-Matching … Blind-Spot Reduction in SAEs").
  - The literature after mid-2026 is voluminous, so expect more.
- Row count (128) exceeds the packet's 40–90 target. About half are relevance 3 and abstract-only; relevance ≥4 is 60 rows.
- Sources are almost all arXiv or Anthropic/GDM web pages. Conference and journal versions were not systematically checked. Version control is limited to arXiv `vN` and update dates.
- Not covered:
  - Critiques and replications of the Anthropic Circuit Tracing paper, beyond 073–076.
  - Full-text verification of all numbers.
  - Gemma 4 E2B architecture documentation. The 35 layers / d_model 1536 figures come only from a community repo snippet (126), not an official config.
  - Per-layer-embedding effects on SAEs. One pre-reset query on this returned nothing relevant; that is not a proof of absence.
- Replication status: 073 is the one actual open-model replication attempt of a Biology-paper claim that I found. 050 contests 049.

## 7. Claims needing separate verification

- All numbers in `key_claims` for "partial" rows are small-model extractions of the page; verify against the paper before using them in `SCIENTIFIC_KNOWLEDGE`.
- Any claim from an abstract_only or metadata_only row (including the Gemma Scope 2 contents, 121, whose PDF was unreadable).
- The "title changed in later arXiv version" note for 047: only the v2 title was seen.
- The Gemma 4 E2B community SAEs (124–127): hook sites, training data and quality metrics were not verified. 124 reports no L0/FVU/CE metrics.
- 122 and 115 describe a library that may not support Gemma 4. I did not check this.
- The 2026 arXiv postings of older work (e.g. 022, dated 2026-05-28 while the original is 2024) need a check of which version is cited.
- OpenAlex gave a wrong title for DOI `10.48550/arxiv.2310.01405` (data-quality issue); that item was excluded.

## 8. Justified next actions (Gemma 4 E2B, 2 vCPU / 8 GiB, CPU-only)

Preregister success criteria before running. **First measure** the memory footprint and tokens/s of `google/gemma-4-E2B-it` on this box. Quantised weights or an offloaded embedding table may be needed; I did not check this.

1. **Validate the community Gemma-4-E2B SAEs (124) before any interpretation.** Run them via SAELens on a few 100k tokens at short context. Measure FVU, delta-CE and L0 per layer (6, 17, 28), and compare with the random-decoder baseline from 046. Cost is dominated by forward passes. It tests whether the SAEs are usable.
2. **Seed-stability and random-baseline test on a small SAE.** Cache layer-17 activations once. Train 3–5 small SAEs (e.g. ~8k latents) with different seeds on a few million tokens, which is feasible on CPU, though I did not benchmark it. Report the shared-latent fraction at cosine ≥0.7 (as in 052/053) and a frozen-decoder baseline. This tests whether the seed and baseline results transfer to Gemma 4.
3. **Neuron-basis circuit baseline (RelP/attribution patching, 072).** Run subject–verb agreement and an IOI-like task on Gemma 4 E2B MLP activations. It needs no trained dictionary, so it fits the budget. Report faithfulness and completeness against circuit size, plus a random-neuron control. Do the same with an SAE on layer 17 if item 1 passes.
4. **Neurons-in-disguise measure (071).** For the SAEs that pass item 1, compute the share of latents aligned with one MLP neuron (gated MLP; follow 071's cosine thresholds). No training.
5. **Toolchain reproduction on a smaller model.** Gemma 3 1B or 270M with Gemma Scope 2 transcoders (121) is CPU-feasible. Reproduce a simple attribution graph with circuit-tracer (122), then the 073 position/strength sweep. This validates the toolchain before Gemma 4, which has no confirmed transcoders.
6. **Entity-recognition directions (063)** on Gemma 4 E2B with a validated SAE. Test steering against 10 random latents.

Defer: training transcoders or CLTs on Gemma 4 (cost and no validation path), and any claim about planning, hallucination or faithfulness before items 1–3 pass.
