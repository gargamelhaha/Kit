# Literature map (cycle 1) — synthesis of scouts LIT-01 … LIT-05

Owner: Synthesis & Meta-Analysis role (Subagent C). Inputs: `agents/reports/LIT-0{1..5}.md`, `SOURCE_REGISTRY.csv` (633 rows), `evidence/source_dedup_map.csv`, `agents/tasks/*.json`.
Status: **structured map, not a systematic review.** Every claim below is a scout's or an author's report; nothing here is verified knowledge (see `SCIENTIFIC_KNOWLEDGE.md` for that).

**ID convention.** All ids are the *kept* ids of `SOURCE_REGISTRY.csv`. Scout reports sometimes cite a duplicate id; map it with `evidence/source_dedup_map.csv` (e.g. IOI = LIT02-003 in LIT-02 → **LIT01-087**; Gemma 4 tech report = LIT03-001 → **LIT02-122**; Gemma Scope = LIT02-123/LIT03-021 → **LIT01-100**; Self-Refine = LIT05-106 → **LIT04-104**).
**Label convention.** "(abstract-level)" = the only evidence seen by any scout is the abstract/snippet. "(partial)" = a scout read selected sections (often through a summarising fetch tool, not verbatim).

---

## (a) Scope, protocol and numbers per area

### Corpus totals (computed from `SOURCE_REGISTRY.csv`, 2026-10-25)
| quantity | value |
|---|---|
| rows submitted by scouts (pre-dedup) | 711 = 128 + 138 + 105 + 186 + 154 |
| duplicates mapped | 78 |
| unique sources in registry | **633** (LIT01 128, LIT02 123, LIT03 55, LIT04 185, LIT05 142) |
| read_depth | **abstract_only 508**, partial 64, metadata_only 59, full_text 2 |
| flagged critique/replication | 184 "yes" + 10 "partial" (of 633) |
| logged search/verification calls | 744 log lines across 5 logs (111 + 229 + 141 + 101 + 162) |
| rows with neither arXiv id nor DOI (web, HF, GitHub, blogs) | 62 |

**Critical caveat on "full_text 2":** both rows are Hugging Face README/config reads of SAE repos (LIT03-026, LIT03-027), not papers. **No scout read any paper in full** (LIT-01 0, LIT-02 0, LIT-04 0, LIT-05 0 full-text; LIT-03 read Gemma 4 report §2/Table 1/refs and Gemma Scope 2 intro/§2/width table). 80% of the registry (508/633) is abstract-level.

### Per-area protocol and saturation (as reported by each scout)
| area (scout) | rows (reg.) | queries / calls actually logged | read depth as reported (full/partial/abstract/meta) | saturation status as reported | saturated? |
|---|---|---|---|---|---|
| **A. Dictionary learning, SAEs, transcoders, crosscoders, attribution graphs** (LIT-01) | 128 (128); 60 rel ≥4; 30 critique | 111 log lines: arXiv 25 live + 23 reconstructed, Semantic Scholar 11 live (805 rows), OpenAlex 12 reconstructed (143), DataCite 3 calls (131 DOI lookups), WebSearch 6, WebFetch 9 live + 22 reconstructed; backward+forward for 4 anchors (LIT01-068, -069, -100, -012), forward only for LIT01-001 | 0 / 28 / 93 / 7 | Topical: last 6 arXiv queries returned 50 results, 4 new relevant (~8% < 10%, **manual count**). Citation chasing: **not saturated** (~40 keyword-matching titles from forward lists of LIT01-012 and LIT01-001 never triaged) | topical ≈ yes (weak); citations **no** |
| **B. Causal methods, probing, phenomena** (LIT-02) | 138 (123); 53 rel ≥4; 47 critique | 229 log lines; 47 logged arXiv searches → 1,048 records (129 arXiv ids verified); 95/99 corpus papers' reference lists parsed; 20 code-URL checks; OpenAlex ≈65–90 pre-restore calls (counts lost) | 0 / 6 / 124 / 8 | Last 10 queries: 149 records, 6 first-time final-list papers (4.0%). **Forward citations failed** (OpenAlex budget, S2 429, OpenCitations empty); 23 rows are "seed_verified" (recalled from model memory, then verified) | topical ≈ yes; citations **no** |
| **C. Gemma 4 / Gemma family + tooling** (LIT-03) | 105 (55) | 141 log lines (first ≈53 lost in reset; one summary line); OpenAlex 80 searches + forward on 6 anchors; S2 citations for 3 anchors (Gemma 3 report stopped, rate-limited); HF API, GitHub API, source greps at pinned transformers commit | 2 / 16 / 66 / 21 | Last 10 queries: 161 new records, 2 entered CSV (1.2%); last 21: 211 new, 3 relevant (1.4%) — below 10% | **yes** for its query set (single-agent relevance judgement; forward tools under-index 2026) |
| **D. Automated science, metascience, multi-agent, memory, self-improvement** (LIT-04) | 186 (185); 69 critique | 101 log lines; 57 arXiv queries → 268 results; reference lists of 9 anchors parsed; OpenCitations 5 anchors, OpenAlex 3 anchors; S2 429 every time (unused) | 0 / 9 / 164 / 13 | Last 10 queries: 25 results, 7 new selected (**28.0%**); last 20: 71 results, 20 new (**28.2%**). Stopping criterion **not met**; stopped for time/rate-limit | **NO** |
| **E. Novel / self-learning / efficient architectures** (LIT-05) | 154 (142); 35 critique | 162 call records; 78 successful arXiv searches (10 failed with 429), OpenAlex 18, DataCite 45, S2 2 (both 429), WebSearch 2, WebFetch 2, GitHub API 1 batch (29 repos) | 0 / 6 / 123 / 25 | "Not saturated"; no saturation curve computed; late batches still surfaced new relevant records; backward citations **not done**; critique search "weakest-covered" step | **NO** |

Search-protocol deviations common to all scouts (from the reports, not from me): arXiv-centric discovery; single-screener LLM relevance judgement; relevance scores are the scout's own (registry `relevance` column has no value ≥4 for LIT03–LIT05 rows; ≥4 counts above exist for LIT01/LIT02 only); venue = "arXiv preprint" unless a `journal_ref` existed; 2–3 sandbox resets per scout wiped scratch data and early log lines (LIT-01: some entries "reconstructed" with approximate timestamps; LIT-02 log regenerated by re-running queries; LIT-04 ≈45 early lines not logged). **The search protocol is therefore only partially reproducible from the logs.**

Year profile (registry): ≤2023: 117 rows; 2024–2026: 516 rows (LIT01 119 of 128 from ≥2024; LIT04 139/185; LIT05 124/142). The 2026 literature is large, mostly single-group preprints verified only to exist via the arXiv API.

---

## (b) Map of research directions (key source_ids)

Strength tags: **[S]** several groups, **[C]** contested, **[1]** single source/lab, **[G4]** touches Gemma 4. Details and independence notes are in `meta_analysis/findings_map.md`.

### A. Representations and dictionary learning
1. **Superposition theory** — LIT01-001 (partial) and a 2024–2026 theory cluster LIT01-002…008, LIT01-010 (abstract-level).
2. **SAE methods** — LIT01-011, -012, -013, -014, -022; Gemma Scope LIT01-100 (partial); Gemma Scope 2 LIT01-121 / LIT03-022 (partial; duplicate row, see (c)).
3. **SAE evaluation and critique [S]** — SAEBench LIT01-028; board-game ground truth LIT01-029; benchmark audit LIT01-032; sanity checks vs random LIT01-046; random-transformer metrics LIT01-047; autointerp/identification LIT01-033, -035; seed dependence LIT01-052, -053; canonical units LIT01-051; absorption LIT01-054; hedging LIT01-055; reasoning/generalisation LIT01-065, -066; causal necessity of single-token features LIT01-040; unreliable SAE interventions LIT03-089.
4. **SAE utility vs baselines [C]** — negative: LIT01-048, LIT01-049, LIT01-120; rebuttal: LIT01-050; "discover, not act": LIT01-128.
5. **Transcoders / CLTs / attribution graphs** — LIT01-068, -069, -070, -115, -116, -117, -122; open-model replications LIT01-073, -074, -075; CLT-Forge LIT01-079; neuron-basis competitors LIT01-071, -072, -076; circuit-claim stability LIT01-077, -078, -043. Neuron-basis vs transcoder **[C]**.
6. **Crosscoders / model diffing [C]** — LIT01-114, -118, -093…-098; simple-baseline competitiveness LIT01-094.

### B. Causal methods and phenomena
7. **Activation/path/attribution patching** — LIT02-001, -005 (partial), -006, -008, -132; LIT01-087 (IOI), -089 (ACDC), -090 (attribution patching), -091 (best practices); 2026 critiques LIT02-013, -014, -030 (single preprints).
8. **Ablation choice, causal scrubbing, faithfulness** — LIT02-022 (partial), -029, -133, -134, LIT01-085, -107; circuit stability across scale LIT02-026, LIT02-011.
9. **Causal abstraction / subspace patching [C]** — LIT02-016, -017 vs -018, -024, -025.
10. **Steering / RepE [S]** — LIT02-034, -035, -036, -037; reliability LIT02-038, -039, -040, -041, -042.
11. **Lenses and probes** — logit lens LIT02-131; tuned lens LIT02-047; shortcut/linear LIT02-050; Patchscopes LIT02-051; readout-vs-computation LIT02-055, -056; probing LIT02-115, -116, -117.
12. **Induction / ICL** — LIT02-057, -058, -060, -061, -062, LIT02-130; Gemma-2 ICL circuit LIT03-098.
13. **Factual recall, multi-hop [C]** — LIT02-002 vs LIT02-020/-021; LIT02-065, -066, -067; multi-hop LIT02-069, -070, -071, -073.
14. **Planning [C]** — LIT02-074, -076, -077; poetry planning LIT01-116 vs LIT01-073, -074, -075.
15. **Multilingual / latent language [C]** — LIT02-079, -080, -081, -082 vs LIT02-084; LIT01-067.
16. **Refusal, hallucination [C]** — LIT02-088 vs LIT02-089, -090, -091, LIT03-088; entity-recognition LIT01-063; LIT02-095.
17. **CoT faithfulness and introspection [C]** — LIT02-096, -097, -098, -099 vs LIT02-102; metrics LIT02-103; introspection LIT02-109, -111, -114 vs LIT02-110, -112, -113.

### C. Gemma 4 E2B specifics (tooling and architecture)
18. **Architecture and hook sites [G4]** — LIT02-122 (tech report; Sec. 2 read by LIT-03), LIT03-004 (`modeling_gemma4.py`@50e35cc, partial), LIT03-005, LIT03-002/-003 (model cards), LIT03-006/-007 (Gemma 3n lineage). Architecture lineage: MatFormer LIT03-013, AltUp LIT03-014, LAuReL LIT03-015, cross-layer KV (CLA LIT03-020, YOCO LIT03-019) — abstract-level.
19. **Tools [G4]** — TransformerLens bridge LIT03-049/-050 (no processed mode), SAELens LIT03-055, nnsight LIT03-053 (BOS issue), circuit-tracer LIT03-056 (no Gemma 4 evidence), pyvene LIT03-057 (no evidence), Neuronpedia LIT03-058.
20. **Pretrained dictionaries [G4]** — `decoderesearch/gemma-4-saes` LIT03-026 (E2B base L6/L17/L28, Matryoshka BatchTopK, no quality metrics read; registry duplicate LIT01-124); Solshine L17 SAE LIT03-027 (self-reported EV 0.83); biosec SAE LIT03-028; gated/unusable LIT03-030. Gemma Scope 1/2 are not applicable (LIT03-023, LIT03-024).
21. **Published Gemma-4 interpretability** — LIT02-128 (repetition-loop neuron; names E2B), LIT02-127 (E2B-IT refusal vs chat template; uses community SAE), LIT02-129 (31B head ablation), LIT03-100 (emotional instability) — all abstract-level, all single-group.

### D. Research-process science (automated science, metascience, multi-agent)
22. **AI-scientist systems** — LIT04-021 (partial), -028, -026, -032, -025, -031, -017 (AlphaEvolve); critiques LIT04-001, -002, -004, -006 (all partial), LIT04-016, -013, -015.
23. **Interpretability-research agents** — LIT04-099, -100, -101, -102 (all abstract-level; none on Gemma 4).
24. **Hypothesis/idea generation** — LIT04-090, LIT04-096 (human studies, abstract-level).
25. **Benchmarks and evaluation validity** — LIT04-056…066, -072, -073, -075, -119.
26. **Multi-agent orchestration and failure [C]** — MAST LIT04-122 (partial); LIT04-141, -142, -143, -134; blogs LIT04-185 vs LIT04-186 (both partial).
27. **Memory, RAG, KG and memory safety** — LIT04-039…055 (53/54 poisoning/contagion).
28. **Judges, calibration, uncertainty** — LIT04-105…109, LIT04-077…087.
29. **Metascience / meta-analysis standards** — LIT04-172 (PRISMA), -173, -174, -175, -176, -114…-119, -117 (leakage).
30. **Self-improving agents** — LIT04-145…156 (DGM LIT04-154 partial), LIT04-158, -161, -162.

### E. Novel / efficient architectures and learning (LIT-05; see LIT-05 §4.2 for the comparison table)
31. **Alternative sequence mixers** — LIT05-003, -004, -005…-007, -022, -023, -026, -031, -032, -034; recall limits LIT05-008, -013, -015, -018; hybrids LIT05-009, -025 (partial); contested hybrid-at-scale LIT05-026 vs LIT05-152 (partial).
32. **Recurrent depth, latent reasoning, tiny reasoners, diffusion LMs [C]** — LIT05-040…042; HRM LIT05-045 vs LIT05-150 (partial), -047, -048 (partial); TRM LIT05-046 (no independent replication located); LIT05-049; diffusion LIT05-058, -062, -063; KAN LIT05-066 vs -067.
33. **Efficient training/inference** — BitNet LIT05-069, -070 vs -071, -075, -077; Muon LIT05-080 vs LIT05-081 (partial), -082; KV compression LIT05-088, -089, -092, -093; scaling/data efficiency LIT05-097, -098, -099, -100, -101.
34. **Self-learning, reflection, self-improvement [C]** — LIT04-104 (Self-Refine) vs LIT04-107, LIT05-109; STaR LIT05-110; self-reward LIT05-111; Mind the Gap LIT05-112; sharpening theory LIT05-113 (metadata); SEAL LIT05-114; Absolute Zero LIT05-115 vs RLVR critiques LIT05-125, -126, -127; model collapse LIT05-102 (title wrong, see (c)) vs LIT05-103.
35. **Continual learning / plasticity (bridge to "learn from scratch in any environment")** — LIT05-138, -139 (2026, abstract-level), LIT05-140, -141; BabyLM LIT05-101.

---

## (c) Coverage gaps and unavailable sources

### Gaps in topic coverage
1. **No mechanistic-interpretability study of Gemma 4 E2B** was located (LIT-02 §4.1; LIT-03 §4.5). Only LIT02-128 and LIT02-127 name E2B in an intervention/audit; no Gemma 4 circuit-tracing, attribution-graph, logit-lens or activation-patching paper was found (absence of arXiv hits ≠ absence).
2. **Architecture-specific interpretability absent:** no source found on per-layer embeddings (LIT-01: one pre-reset query, nothing relevant, not logged), cross-layer KV sharing (only architecture papers LIT03-019/-020, abstract-level), logit softcapping, or layer-scalar effects on SAEs/probes. All protocol transfer to E2B is unvalidated.
3. **Dictionaries:** no Gemma 3n SAEs, no Gemma 4 transcoders or crosscoders on HF (LIT-03 searches); quality metrics (FVU, ΔCE, L0) of every Gemma 4 E2B SAE unread (LIT03-026); base→IT transfer untested; hook-site/BOS conventions unverified. circuit-tracer and pyvene: "no evidence" of Gemma 4 support (not proof of incompatibility); nothing executed by LIT-03.
4. **Chat-template / instruction-tuning sensitivity** of mechanisms is barely studied (LIT-02 gap 5; one data point: LIT02-127).
5. **Critiques of circuit tracing:** replications beyond LIT01-073…-076 not covered (LIT-01 §6).
6. **Mechanistic (not behavioural) tests** of CoT faithfulness and introspection in small models (LIT-02 gap 6).
7. **AI-scientist reliability:** no controlled study isolating one design choice (verifier placement, memory, judge diversity); no independent replication of AI Scientist v2, co-scientist, Kosmos or DGM headline results; claim-level calibration for research agents barely studied; agent-memory evidence is mostly self-reported system papers (LIT-04 gaps 1–2, 5–6).
8. **No quantitative meta-analysis is possible** from the collected material: no source reports pooled effect sizes with heterogeneity for agent or interpretability studies; metrics are incompatible across sources (LIT-04 gap 4; LIT-02 §7).
9. **Novel-architecture scope is narrower than the mission statement.** A title-level grep of all 633 titles finds 0 hits for forward-forward, predictive coding, Hebbian, neuroevolution/architecture search, curiosity/intrinsic motivation, meta-learning, neuromorphic/spiking, local learning rules, active inference, program synthesis, neuro-symbolic; 3 for "continual", 1 for "energy-based", 1 for "world model". LIT-05 reports it ran one probe on architecture-search/automated discovery with no relevant hit, and that test-time-compute scaling and agent self-improvement received only a few probes. Whether these areas are empty or just unsearched is **unknown**. Interpretability of non-Transformer architectures is nascent (LIT05-018, -047, -048); SAE/transcoder compatibility with recurrent states is unaddressed in sources read.
10. **No independent ≥7B replication** of any hybrid-vs-attention or SSM-vs-Transformer claim (LIT-05 §4.4).

### Unavailable / unreadable sources and failed tools
- Citation tooling: OpenAlex free daily budget exhausted (LIT-01, LIT-02, LIT-04); OpenAlex has `referenced_works_count = 0` for arXiv-DOI records (LIT-03, LIT-04, LIT-05) so backward search via OpenAlex failed; Semantic Scholar HTTP 429 (LIT-02, LIT-04 every attempt, LIT-05 both attempts, LIT-03 for Gemma 3 report); OpenCitations sparse for arXiv DOIs (LIT-04: counts 43/10/2/3/0 vs OpenAlex totals 116/18/5 for overlapping anchors — a 2–3× spread). Forward-citation recall for ML preprints is therefore low everywhere; LIT-02's "forward-in-corpus" counts (IOI 34, Geva 29, tuned lens 18, …) are sampling-biased substitutes, not citation counts.
- arXiv API HTTP 429 from the shared IP in every scout; LIT-05 lost 10 queries.
- Unreadable or blocked: Gemma Scope 2 PDF (LIT-01: unreadable; LIT-03 did parse it), OpenReview "Use SAEs to Discover Unknown Concepts…" (blocked; LIT01-128 snippet only), `arxiv.org/html/2209.10652` 404, "Progress on Attention" and "Towards Monosemanticity" (snippets only), Medium mirror of GDM post (403; LessWrong used), gated `elrashid/sae-gemma-4-e2b-it-topk-l17` (LIT03-030).
- Not searched: Google Scholar, ACL Anthology, OpenReview reviews, DBLP, SSRN, PubMed/Europe PMC, conference proceedings, non-English literature, patents (LIT-04 §2/§6; LIT-05 §6). Peer-review status unknown for almost all rows.
- Reading-tool limitation: "partial" in several scouts means a small-model summary via WebFetch, not verbatim reading (LIT-01, LIT-05 §2, LIT-04 §5). Numbers from those rows (e.g. ARC Prize 32% vs 41%, MiniMax claims, Anthropic +90.2%) must be re-checked against primary pages.

### Registry integrity issues found while synthesising (not yet fixed; registry is owned by the coordinator/curator)
1. **LIT05-102** carries a wrong title ("Dynamic / ME-JEPA v2.0.0-rc1: Audited World-Model Runtime…") on arXiv 2305.17493 (Shumailov et al., model collapse). The row's authors, arXiv id and `key_claims` describe the Shumailov et al. model-collapse paper, so the title (apparently from the DataCite lookup) is a mismatch; true title to be re-verified. LIT-05's contested-claim item "102 vs 103" depends on it. Other DataCite-derived LIT05 titles should be spot-checked.
2. **LIT02-122 (Gemma 4 Technical Report)** is `abstract_only` in the registry, but the duplicate LIT03-001 was `partial` (LIT-03 read §2, Table 1 and references). The dedup kept the lower read depth — the only such case among the 78 mappings. Treat LIT02-122 as `partial`.
3. **Unmapped duplicates:** LIT01-022 and LIT03-043 (same title, "Scaling Monosemanticity…"; LIT01-022 dated 2026 vs 2024 original, version question open per LIT-01 §7); LIT01-121 and LIT03-022 (Gemma Scope 2 paper).
4. **Reconstructed/regenerated logs** (LIT-01, LIT-02) and the missing LIT-05 task packet at run time (scout defined sub-areas A–D itself; packet regenerated later) weaken provenance.
5. LIT-03 states 105 rows with 24 critique-flagged; registry shows only 8 critique-"yes" for LIT03 after dedup, because most critique rows were mapped into LIT01/LIT02 (consistent, not an error).

---

## (d) Reading debt — 15 most decision-relevant abstract-only sources to read in full next

Selection rule: source is `abstract_only` in the registry, and a pending or planned experiment, a replication target, or a methodological choice in `meta_analysis/replication_plan.md` depends on its details (protocol, hyper-parameters, exact metric). Order = priority.

| # | source_id | title (short) | why read in full |
|---|---|---|---|
| 1 | **LIT02-002** | Locating and Editing Factual Associations in GPT (ROME) | E004's replication target; scouts have only the abstract. Need exact causal-tracing protocol (Gaussian-noise corruption, window sizes, AIE metric) to decide which parts E004 re-implements and which it replaces with counterfactual patching. |
| 2 | **LIT01-091** | Towards Best Practices of Activation Patching (Zhang & Nanda) | Defines the metric and corruption choices E004 relies on (logit-difference, symmetric token replacement); the "intervention direction/metric" caveat is the main reason patching results disagree. Also needed to set up the AtP-vs-exact validation. |
| 3 | **LIT02-065** | Dissecting Recall of Factual Associations (Geva et al.) | Source of the attention-knockout protocol and the "enrichment → propagation → extraction" staging; H-003 and the K/V(13,14) bottleneck test are phrased against it. |
| 4 | **LIT02-057** | In-context Learning and Induction Heads (Olsson et al.) | E003 replication target; need induction-score definitions, prefix-matching score and the "phase change" claims actually tested, to keep E003's predictions faithful. |
| 5 | **LIT02-047** | Tuned Lens | Decides whether E002's early-layer logit-lens results are interpretable; need translator training recipe, loss, data size, and the paper's own notes on models whose unembedding/final norm differ (softcap, tied embeddings). |
| 6 | **LIT02-088** | Refusal Is Mediated by a Single Direction (Arditi et al.) | Directional-ablation/activation-addition protocol and the exact refusal score; the refusal experiment (R5 in the plan) and the contested multi-direction claims (LIT02-089, -090, LIT03-088) all pivot on its definitions. |
| 7 | **LIT01-012** | Scaling and Evaluating Sparse Autoencoders | Defines the TopK SAE and the normalised-MSE / downstream-loss metrics used to judge any SAE; needed before computing FVU/ΔCE on the community Gemma 4 SAEs. |
| 8 | **LIT02-128** | When Can One Neuron Fix Repetition Loops in LLMs? (names E2B) | The only causal-intervention paper found that names Gemma 4 E2B; reproduction requires exact prompts, seeds, attribution method and a verified code link (LIT-03 claim 9: code link unverified). |
| 9 | **LIT02-127** | BioRefusalAudit (E2B-IT refusal 65/75 vs 0/75 with/without chat template; uses community SAE) | Single-source Gemma 4 result that, if true, makes template choice a first-order confound for all E2B-IT experiments; also the only user of the E2B-IT community SAE. Read for prompt sets, scoring and SAE usage. |
| 10 | **LIT03-098** | Contextualize-then-Aggregate: Circuits for ICL in Gemma-2 2B | Candidate reproduction on the control model (Gemma 2 2B) for validating the lab's patching code before trusting E2B results. |
| 11 | **LIT02-017** | Is This the Subspace You Are Looking For? (Makelov et al.) — read with reply LIT02-018 | Determines how to interpret residual/subspace patching on E2B (dormant-pathway illusion) and whether the PLE re-injection is an instance of it. Both sides are abstract-level for scouts. |
| 12 | **LIT01-087** | Interpretability in the Wild (IOI) | Template/data/metric details for the IOI-style head-patching experiment; also the standard "does the tool work" sanity check. |
| 13 | **LIT05-013** | Zoology: Measuring and Improving Recall (MQAR) | Defines MQAR, the task for the cheapest small-architecture experiment (recall vs state size); need data generator specifics and baselines. |
| 14 | **LIT05-046** | Less is More: Recursive Reasoning with Tiny Networks (TRM) | Needed to implement the HRM-vs-refinement experiment (arms b/d) faithfully; scouts found no independent replication, and the repo is archived. |
| 15 | **LIT04-143** | Representational Collapse in Multi-Agent LLM Committees | The only quantitative evidence (n=100 GSM8K questions, one model family) behind the lab rule "same-weights roles are not independent verifiers"; decides how much weight the independent-audit step can carry. |

Runners-up (also abstract-only): LIT01-040 (are single-token SAE features causally necessary), LIT03-089 (SAE interventions unreliable), LIT02-008 / LIT01-090 (AtP* and attribution patching vs ACDC), LIT02-013 (when attribution patching lies), LIT05-080 (Muon), LIT05-022 (Gated DeltaNet), LIT05-112 (Mind the Gap), LIT04-016 (reward hacking in research agents).

Already `partial` and still needing full reading: LIT01-046, LIT01-048, LIT01-049, LIT01-032, LIT02-005, LIT02-022, LIT02-038/-039, LIT05-081, LIT05-150, LIT05-152 (LIT-03 §8 item 4 also asks for AxBench, SAE sanity checks, random-init, SAEBench and patching best practices in full).
