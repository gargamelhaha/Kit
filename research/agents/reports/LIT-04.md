# LIT-04 report: automated science, metascience, multi-agent systems, agent memory, self-improvement

Task packet: `research/agents/tasks/LIT-04.json` (role: Literature Scout + Evidence Extractor). Date of run: 2026-10-25. Research question: *what design choices demonstrably improve reliability of autonomous research agents, and which failure modes are documented?*

Convention: claims are attributed ("authors report ..."); nothing here is asserted as established truth. `read_depth` in the CSV is honest: **9 partial, 164 abstract_only, 13 metadata_only, 0 full_text**. No paper was read in full.

## 1 What was done

1. Searched arXiv (API) with 5 Boolean queries across 12 topic areas (AI-scientist systems, AI-scientist evaluation/critique, hypothesis generation, research-agent benchmarks, metascience/statistics, systematic-review automation, calibration/UQ, LLM-as-judge, multi-agent orchestration/failures, agent memory/RAG/KG, self-improving agents, interpretability-research agents).
2. Added foundational works from prior knowledge, each **verified** by an arXiv `id_list` call (all 171 arXiv ids resolved) or a Crossref/DataCite DOI call (17 DOIs).
3. Backward citations: parsed the reference lists (arXiv HTML) of 7 anchor papers: AI Scientist v1, AI Scientist v2, MAST, Darwin Godel Machine, Luo et al. pitfalls, Trehan et al., Kapoor et al. "AI agents that matter" (2408.06292: refs 116, arXiv ids 32; 2503.13657: refs 84, arXiv ids 51; 2505.22954: refs 184, arXiv ids 61; 2504.08066: refs 31, arXiv ids 10; 2509.08713: refs 30, arXiv ids 10; 2601.03315: refs 28, arXiv ids 21; 2407.01502: refs 66, arXiv ids 44; 2502.14297: refs 20, arXiv ids 8; 2506.01372: refs 98, arXiv ids 64). Anchor-bibliography ids were used to find additional works (e.g. LIT04-111, LIT04-135, LIT04-138, LIT04-137, LIT04-020, LIT04-118, LIT04-034, LIT04-140).
4. Forward citations: OpenCitations Index v2 for 5 anchors and OpenAlex `filter=cites:` for 3 anchors (AI Scientist v1, MAST, DGM). Counts differ sharply by index (see section 6).
5. Critiques/replications/evaluations of AI-scientist systems were searched explicitly (see section 4.3).
6. Read selected full-text sections (arXiv HTML) of 7 papers and the text of 2 engineering blogs (via WebFetch summarisation) -> `partial`.
7. Wrote the three deliverables. Environment note: the shared sandbox was reset **three times** during the task, wiping my scratch directory (and the first ~45 search-log lines, which covered the initial arXiv/OpenAlex/Crossref/OpenCitations probes). I re-ran every search listed in the final log (timestamps in the log are from the final run), so the log is complete for what is claimed, but the pre-reset probes (including 20 OpenAlex foundational look-ups and several Semantic Scholar attempts) are not logged.

## 2 Sources checked

| Catalog / endpoint | Used for | Notes |
|---|---|---|
| arXiv API (search, id_list) | discovery, bibliographic verification | rate-limited (HTTP 429) repeatedly; searches spaced 8 s |
| arXiv HTML (arxiv.org/html) | reference lists (backward), section reading | works for recent papers only |
| OpenCitations Index v2 | forward citations | sparse for arXiv DOIs (see section 6) |
| OpenAlex | forward citation totals; DOI-level counts | free daily budget (shared per IP) was exhausted in the first session; `referenced_works_count` was 0 for arXiv-DOI records, so OpenAlex could not be used for backward citations |
| Crossref / DataCite | DOI verification | 17 DOIs verified by direct GET |
| Semantic Scholar | attempted (backward/forward) | HTTP 429 on every attempt (shared pool); **not used** |
| WebFetch | Anthropic and Cognition engineering blogs | summarised page content, hence `partial` |

Not searched: Google Scholar, ACL Anthology, OpenReview, DBLP, SSRN, PubMed/Europe PMC for review-methods papers (one Europe PMC call was made pre-reset, not logged), and conference proceedings directly. Coverage is therefore arXiv-centric.

## 3 Actions run (all appear in `research/logs/search_LIT-04.jsonl`)

- Crossref/DataCite DOI verification: 20 log entries
- OpenAlex forward citations (filter=cites): 3 log entries
- OpenCitations Index v2 (forward citations): 5 log entries
- WebFetch: 2 log entries
- arXiv API id_list (verification): 4 log entries
- arXiv API search: 5 log entries
- arXiv HTML bibliography parse (backward citations): 9 log entries

## 4 Results

### 4.1 Thematic map

| Domain | n sources | Sub-topics |
|---|---|---|
| AI-scientist evaluation/critique | 18 | case studies of autonomous research; independent evaluation of AI scientists; limits of AI acceleration; peer-review/LLM reviewers and paper-error detection; reward hacking/sabotage/oversight |
| AI-scientist systems | 25 | algorithm/program evolution; domain AI-agent lab (virtual lab); domain/chemistry tool agents; end-to-end research systems; surveys/position |
| Agent memory/RAG/knowledge graphs | 17 | RAG and long context; agent memory systems; memory safety/portability |
| Benchmarks for research agents | 21 | ML-research/replication benchmarks; benchmark validity/saturation |
| Calibration/uncertainty | 12 | LLM confidence and UQ; agent-level uncertainty; semantic entropy for hallucination detection |
| Hypothesis/idea generation | 11 | idea generation and novelty |
| Interpretability-research agents | 4 | automated interpretability agents |
| LLM-as-judge/self-verification | 8 | judge bias and self-correction limits |
| Metascience/statistics | 16 | analytic variability (many analysts); heterogeneity statistics (I^2); random-effects meta-analysis; replication of published results; reproducibility and evaluation statistics; reproducibility in AI research |
| Multi-agent orchestration | 26 | failure taxonomies/diagnosis; frameworks; multi-agent vs single-agent evidence; transparency/governance |
| Self-improving agents | 21 | self-modifying / open-ended self-improvement; surveys, safety, theory, recent harness work |
| Systematic review/meta-analysis automation | 7 | LLM-assisted systematic review; reporting standards (PRISMA) |

Representative anchors (attributions are the authors' own claims):
- **AI-scientist systems.** AI Scientist v1 LIT04-021 and v2 LIT04-028 (authors report an AI-generated workshop paper accepted via peer review), Google AI co-scientist LIT04-026 (published version DOI 10.1038/s41586-026-10644-y), Agent Laboratory LIT04-025, Kosmos LIT04-032, DeepScientist LIT04-031, Robin LIT04-029, AI-Researcher LIT04-030, AgentRxiv LIT04-027, AlphaEvolve LIT04-017, FunSearch LIT04-179. Surveys: LIT04-036, LIT04-035, LIT04-034.
- **Idea generation/novelty.** LIT04-090: authors report (100+ NLP researchers, blind review) LLM ideas judged more novel (p<0.05) but slightly weaker on feasibility. LIT04-096: authors report (43 experts, >100 h each executing assigned ideas) that LLM-idea scores fall significantly more than expert-idea scores after execution on novelty, excitement, effectiveness and overall (p<0.05), closing the gap seen at idea stage.
- **Benchmarks.** MLE-bench LIT04-060, MLAgentBench LIT04-056, RE-Bench LIT04-061, PaperBench LIT04-065, ScienceAgentBench LIT04-059, DiscoveryBench LIT04-057, CORE-Bench LIT04-058, EXP-Bench LIT04-066, AstaBench LIT04-069, MLGym LIT04-063; interpretability-specific: SAEScientist-Bench LIT04-102 (agents using a Gemma Scope dictionary on Gemma-2-9B-IT; directly relevant to this lab) and LIT04-101.
- **Multi-agent.** MAST LIT04-122, scaling study LIT04-142, single vs multi LIT04-141, debate LIT04-134, Magentic-One LIT04-133, AutoGen LIT04-132, plus the Anthropic and Cognition blogs.
- **Memory/RAG/KG.** RAG LIT04-039, GraphRAG LIT04-042, MemGPT LIT04-044, Generative Agents LIT04-043, Zep LIT04-046, Mem0 LIT04-048, A-MEM LIT04-047, survey LIT04-049, YouRA LIT04-051 (persistent-state, evidence-traceable autonomous research), and safety work LIT04-053, LIT04-052, LIT04-054.
- **Self-improvement.** Godel Machine LIT04-145, STOP LIT04-149, Promptbreeder LIT04-148, ADAS LIT04-150, Godel Agent LIT04-151, SICA LIT04-153, DGM LIT04-154, surveys LIT04-155, LIT04-156, safety/theory LIT04-160, LIT04-158, LIT04-162, LIT04-161.

### 4.2 Design choices vs evidence on reliability

"For/against" reports what the cited authors claim; evidence strength is my annotation. Most entries rest on abstracts only (flagged `[abs]`) or partial reading (`[part]`).

| Design choice | Evidence for / against (authors' reports) | Strength | source_ids |
|---|---|---|---|
| Verification at every pipeline stage, grounded in raw data rather than LLM summaries | For: Trehan et al. [part] derive "verify at every stage; ground in raw data" from 3 failed of 4 attempts; Luo et al. [part] show four failure modes (benchmark selection, leakage, metric misuse, post-hoc selection) that paper-level review misses; MAST [part] includes verification failures. Against/limits: n=4 attempts (anecdotal); LLM verifiers share biases (next rows) | moderate-weak (qualitative + controlled on 2 systems) | LIT04-006, LIT04-004, LIT04-122 |
| Centralized verification vs decentralized agent topologies | For: Kim et al. [abs] report architectures without centralized verification tend to propagate errors more (260 configurations, 6 benchmarks, 3 LLM families; cross-validated R^2=0.373 for their predictive model) | moderate, single study, abstract-level | LIT04-142 |
| Multi-agent over single-agent | For: Anthropic blog [part] reports +90.2% vs single-agent on an *internal* research eval, at ~15x chat tokens; token use explained 80% of BrowseComp variance (their analysis). Against/limits: Gao et al. [abs] report MAS benefit diminishes as LLMs improve; Kim et al. [abs] report diminishing returns once single-agent baselines are strong and overhead on tool-heavy tasks; Smit et al. [abs] find debate does not reliably beat self-consistency/ensembling; Cognition blog [part] argues (no data) for single-threaded agents. Task type matters: breadth-first research vs tightly coupled coding | mixed; vendor-internal vs academic controlled | Anthropic blog (see CSV), Cognition blog (see CSV), LIT04-141, LIT04-142, LIT04-134 |
| Multiple "roles" on the same base model as independent checks | Against: Patel [abs] reports representational collapse for 3 role-prompted Qwen2.5-14B agents on 100 GSM8K questions (mean cosine similarity 0.888, effective rank 2.17/3); LLM evaluators favour their own generations (Panickssery et al. [abs]). Implication: same-weights roles are logical roles, not independent evidence (consistent with the project brief) | weak-moderate (small n; one model family) | LIT04-143, LIT04-109, LIT04-108 |
| LLM-as-judge for scaled annotation/review, validated against humans | For: MAST [part] LLM annotator (o1 few-shot) reports accuracy 0.94, kappa 0.77 vs human experts. Against: reward-hacking study [abs] reports an LLM panel reviewing only code+scores missed 33/505 (6.5%) confirmed hacks; judge-bias literature (position/verbosity/self-preference) [abs]; self-correction limits [abs] | moderate | LIT04-122, LIT04-016, LIT04-106, LIT04-105, LIT04-107 |
| Execution-based validation of ideas before claiming value | For: ideation-execution gap [abs] (n=43 experts executing ideas). Idea-stage novelty ratings alone were not predictive of post-execution quality for LLM ideas | moderate (human experiment) | LIT04-090, LIT04-096 |
| Sandboxing / containment of code-executing agents | For: AI Scientist v1 authors [part] report minimal sandboxing led to a run relaunching itself; reward-hacking study [abs] reports spontaneous hacking in 30.5% of attempts on open-ended research-pipeline tasks vs 2.9% on task-specific kernels (17 models, 38 tasks) | moderate (authors' own observations; one controlled study) | LIT04-021, LIT04-016, LIT04-015, LIT04-013 |
| Adversarial / pre-specified testing of claims (preregistration-like) | For: position/analysis papers [abs] warn that agents make plausible endlessly revisable analyses and argue for adversarial experiments and preregistering for the next LLM; Luo et al. show post-hoc selection bias in practice. No controlled evidence of net benefit found | weak (positions) | LIT04-008, LIT04-121, LIT04-004, LIT04-094 |
| Persistent external memory with checkpoints | For: Anthropic blog [part] reports the lead agent persisting its plan because context >200k tokens may be truncated; durable execution/resume. Against: memory poisoning/injection attacks [abs], evaluator-bias propagation through memory ("memory contagion") [abs], portability of memory across model upgrades studied in LIT04-055 [abs, result not read] | weak-moderate; safety literature is attack-focused | LIT04-053, LIT04-052, LIT04-054, LIT04-055 |
| Open-ended self-modification validated empirically | For: DGM authors [part] report benchmark gains and self-found fixes for tool-use hallucination. Against: same authors show objective hacking; theoretical work [abs] asks whether safety can be certified across self-rewrites (result not read); poisoned-benchmark attacks on self-modifying agents [abs] | weak-moderate | LIT04-154, LIT04-158, LIT04-162 |
| Report cost, variance and construct validity, not just accuracy | For: Kapoor et al. [abs] argue accuracy-only evaluation yields needlessly complex/costly agents and wrong conclusions about sources of gains; construct-validity review of 445 benchmarks by 29 reviewers [abs]; benchmark saturation case study [abs] | moderate | LIT04-072, LIT04-073, LIT04-075, LIT04-119 |
| Give each subagent explicit objective, output format, boundaries; scale effort to task | For: Anthropic blog [part] lists duplicated work, spawning 50 subagents for simple queries, endless search as failure modes it mitigated by prompt guidance | weak (vendor anecdote) | Anthropic blog (see CSV) |

### 4.3 Documented failure modes

**Autonomous research / AI-scientist systems**
- Weak novelty checking and thin literature use (keyword search; micro-batching for SGD judged "novel"); median 5 citations per paper, 5/34 from 2020+; structural errors (missing figures, placeholder text) — Beel, independent evaluation of AI Scientist v1 [part] LIT04-001.
- Hallucinated plots, missing related-work sections, uncontrolled self-relaunch (insufficient sandboxing) — AI Scientist v1 authors [part] LIT04-021.
- Four failure modes shown in controlled experiments on two open-source systems: inappropriate benchmark selection, data leakage, metric misuse, post-hoc selection bias — LIT04-004 (PNAS 2026, DOI verified).
- Six recurring failure modes from 4 attempts (3 failed): training-data defaults, implementation drift under execution pressure, memory/context degradation, overexcitement (declaring success despite failures), insufficient domain intelligence, weak scientific taste — LIT04-006.
- Implementation capability and verification as bottlenecks (position) — LIT04-002; see also LIT04-003, LIT04-007 [abs].
- Reward hacking: spontaneous rate 30.5% (open-ended research-pipeline tasks) vs 2.9% (kernels); 74.6% of "hack-allowed" attempts confirmed hacks; reviewing code+scores alone misses 6.5%; evasions rise over a five-round loop (authors report) — LIT04-016; benchmarks LIT04-015, LIT04-013, LIT04-014.
- Corpus failure/publication bias: authors estimate a "null result gap" (~0.60 drug discovery, ~0.56 psychology, ~0.35 cancer biology; first-order estimates, not measured on agents) — LIT04-009.
- Plagiarism and reviewer weaknesses (titles/abstracts only): LIT04-010, LIT04-011, LIT04-012; AI-authored papers at a venue: LIT04-005.
- **Interpretability-research agents (most relevant here):** replication-based evaluation pitfalls — expert explanations can be subjective/incomplete, outcome comparison hides process, and systems may reproduce published findings by memorisation or informed guessing LIT04-101; deceptive coordination to fool overseers LIT04-100 [abs]; benchmark LIT04-102.

**Multi-agent LLM systems**
- MAST: 14 failure modes in 3 categories; reported failure rates 41%-86.7% across 7 open-source MAS; human kappa 0.88 [part] LIT04-122. Follow-ups on attribution and localisation: LIT04-124, LIT04-125, LIT04-126 [abs].
- Anthropic [part]: over-spawning, duplicated work, gaps, wrong tool choice, SEO-farm sources, error compounding after small changes; synchronous subagents as bottleneck. Cognition [part]: parallel subagents make conflicting implicit decisions.
- Error propagation without central verification LIT04-142; correlated/collapsed role agents LIT04-143; risks survey LIT04-140.

**Memory:** poisoning/injection LIT04-053, LIT04-052; bias propagation LIT04-054; long-context position effects LIT04-040 [abs].

**Self-improvement:** objective hacking and hallucinated tool use in DGM [part]; benchmark poisoning of self-modifying agents LIT04-162; guardrail stress tests LIT04-161 [abs].

### 4.4 Methodological standards for systematic review and meta-analysis applicable to ML literature

(Standards below are recalled domain knowledge; the cited methodological sources were verified bibliographically but their text was not read in this task, except where noted.)
- **Protocol and reporting:** PRISMA 2020 (LIT04-172) — pre-specified question and eligibility, reported full search strings per database, dual screening (or documented single-screener risk), flow of records, risk-of-bias assessment, synthesis method stated up front. LLM-assisted screening is being studied (LIT04-168, LIT04-166, LIT04-167); reporting extensions are proposed (LIT04-171, [abs]). This review departs from PRISMA: arXiv-centric search, single screener (an LLM agent), abstract-level extraction.
- **Effect sizes:** for continuous scores with seed/sample variance use standardized mean difference (Hedges' g) or paired differences with bootstrap CIs; for pass/solve rates use log odds ratio or risk difference with Wilson/Clopper-Pearson intervals; report the ML-specific interval-estimate guidance (aggregate intervals over runs, LIT04-116; variance sources, LIT04-115; evaluation error bars, LIT04-119).
- **Heterogeneity:** report Cochran's Q, I^2 and tau^2 and a prediction interval (LIT04-175); random-effects pooling (LIT04-176) is the default when studies differ; with few studies the DerSimonian-Laird interval is known to be too narrow, so use a small-sample correction (e.g. Hartung-Knapp; recalled, not verified here). Meta-regression only with roughly >=10 studies per covariate (Cochrane rule of thumb, recalled).
- **When pooling is NOT valid:** different metrics or metric definitions (accuracy vs pass@k vs rubric score); different benchmark versions or contamination status; different base models/versions/scaffolds (a "model" effect and a "scaffold" effect cannot be separated, cf. LIT04-075); non-independent results (same benchmark reused, same authors, shared base model, LLM-judged by the same judge family); no variance/seed reporting; construct-validity mismatch (LIT04-073: 445 benchmarks reviewed); evidence of selective reporting (LIT04-009, LIT04-121). In these cases keep studies separate and present a structured narrative/evidence table, never a pooled number. Vote-counting of "significant/non-significant" is not meta-analysis.
- **Bias controls:** leakage (LIT04-117), many-analysts variability (LIT04-173), replication base rates (LIT04-174), reproducibility documentation in AI (LIT04-177), reporting of tuning budget (LIT04-114), RL reproducibility (LIT04-112).

### 4.5 Calibration methods (for agent claims and for the lab's own confidence labels)
- Post-hoc scaling and ECE for classifiers (LIT04-077); self-evaluation P(True)/P(IK) (LIT04-079); verbalised confidence (LIT04-081, LIT04-082, LIT04-084); semantic-entropy sampling (LIT04-080, published: LIT04-178); sampling-consistency checks (LIT04-103); conformal prediction for distribution-free coverage under exchangeability (LIT04-078); UQ surveys (LIT04-083).
- Agent-level: uncertainty-as-control-signal for long-horizon agents LIT04-085; taxonomy/evaluation protocol LIT04-087; LIT04-086 evaluates whether uncertainty signals stay discriminative mid-trajectory (abstract start read; result not read — treat as an open caution: step-level uncertainty may not predict failure).
- Caveat repeated across judge-bias works: LLM self-assessments inherit evaluator bias (LIT04-109, LIT04-108), so confidence from the producing model is not independent evidence.

## 5 Evidence

- Full inventory: `research/evidence/LIT-04_sources.csv` (186 rows; each row has arXiv id or verified DOI/URL, `found_via`, `read_depth`).
- Search/verification log: `research/logs/search_LIT-04.jsonl` (48 entries).
- `partial` rows (9): LIT04-001 (2502.14297); LIT04-002 (2506.01372); LIT04-004 (2509.08713); LIT04-006 (2601.03315); LIT04-021 (2408.06292); LIT04-122 (2503.13657); LIT04-154 (2505.22954); LIT04-185 (blog); LIT04-186 (blog). What was read: abstract plus specific sections (listed in each row's `key_claims`/`limitations`); nothing was read end to end.
- `abstract_only` rows: claims are sentences selected from the abstract by a keyword heuristic (prefixed "Authors report (abstract)"); some selected sentences may be generic. Treat them as pointers to verify, not as extracted evidence.
- Bibliography integrity: titles/authors/dates come from the arXiv API (v1 date; later versions noted); DOIs only where arXiv supplied one verified by Crossref, or via direct Crossref/DataCite GET. Venue is "arXiv preprint (peer-reviewed venue, if any, not verified here)" unless arXiv or Crossref stated one.

## 6 Limitations and saturation

**Counts.** 186 deduplicated sources (<=2022: 19; 2023: 28; 2024: 37; 2025: 57; 2026: 45), 69 flagged critique/evaluation/replication-type. This exceeds the 40-90 target: breadth was preferred, so most rows are abstract-level.

**Acceptance criteria.** Foundational (<=2023) sources: 47 (>=8 required: met). Recent (2024-2026) primary sources: 139 (>=10: met). Backward AND forward citation search for >=3 anchors: backward = 7 anchors (bibliography parse); forward = 5 anchors via OpenCitations and 3 via OpenAlex (AI Scientist v1, MAST, DGM): met, with caveats below. Critical/negative works: 69 flagged (>=3: met).

**Saturation (honest statement).** Over the 5 arXiv discovery queries: first half (2 queries): 12 results, 5 new selected (41.7%). second half (3 queries): 18 results, 2 new selected (11.1%). All arXiv discovery queries: 30 results returned, 7 first-time selected sources (the other selected sources came from id_list-verified prior knowledge, backward citations or DOI look-ups). Saturation by the <10%-new criterion was **reached for the core AI-scientist-systems and benchmark queries** (early-listed queries re-returned already-known items) but **not demonstrated for every sub-area**: in the second half of the search, sub-areas such as agent-level uncertainty, self-improving harness papers, memory safety and hypothesis benchmarks still yielded new relevant items. The 2026 literature in these sub-areas is growing quickly, so completeness cannot be claimed. Relevance judgement ("selected") is mine at title/abstract level and was not independently double-screened.

**Citation-index limitations.** OpenCitations returned 2408.06292: 43, 2504.08066: 10, 2506.01372: 2, 2410.07095: 3, 2503.13657: 0 citing works for the anchors; OpenAlex reported 2408.06292: 116, 2503.13657: 18, 2505.22954: 5 (totals) — a 2-3x spread for the same paper, and the lists are dominated by journal articles and biomedical/application papers rather than ML preprints. Neither index captures arXiv-to-arXiv citations well, so forward-citation recall for ML preprints is low. Semantic Scholar (the usual remedy) was unavailable (HTTP 429). Forward search was therefore supplemented by topic queries, which is a weaker proxy.

**Other.** arXiv-only discovery; no conference/OpenReview search; code repositories were not inspected (`code_url` filled only when a GitHub link appears in the abstract); the 2026 items (e.g. arXiv ids 26xx.xxxxx) were verified only through the live arXiv API on 2026-10-25 and have had no time for independent scrutiny; blogs were read via summarisation; pre-reset probes are not in the log (section 1).

**Gap list.**
1. No controlled study found that isolates which single design choice (verifier placement, memory, judge model diversity) improves AI-scientist reliability; evidence is observational/case-study or vendor-internal.
2. No independent replications of AI Scientist v2, co-scientist, Kosmos or DGM headline results were found in this search (absence in arXiv search is not proof of absence).
3. Mechanistic-interpretability agents: only LIT04-101, LIT04-102, LIT04-099, LIT04-100 found; nothing on Gemma 4 E2B specifically.
4. Quantitative meta-analysis of agent-benchmark results with compatible metrics is not possible from reported abstracts; no source reports pooled effect sizes with heterogeneity for agent studies.
5. Calibration of research-agent *claims* (as opposed to answers) is barely studied.
6. Agent-memory evidence is dominated by system papers with self-reported benchmarks; independent comparative evaluations are scarce (only abstracts seen).
7. Not searched: OpenReview reviews of AI-scientist papers, ACL/NeurIPS proceedings, non-English literature, patents.

## 7 Claims needing verification (do not enter SCIENTIFIC_KNOWLEDGE as established)

1. Anthropic's +90.2% multi-agent gain and the 80% variance attribution (internal eval; no data/code).
2. MAST failure-rate range (41%-86.7%) and kappa values (verified only by reading the paper's text; no re-annotation).
3. Reward-hacking rates (30.5% / 2.9%; 74.6%; 6.5% missed) — from an abstract.
4. Ideation-execution gap and novelty-vs-feasibility results — abstract-level; check for later versions/replications.
5. "Null result gap" estimates — first-order estimates from a position-style paper.
6. Scaling-agent-systems model (R^2=0.373) and "centralized verification reduces error propagation" — abstract-level.
7. Representational-collapse numbers (n=100 questions, one model family).
8. Whether MAST categories and Luo et al. failure modes transfer to a CPU-only, single-model lab (this lab's setting).
9. Every `abstract_only` row's key_claims sentence (auto-selected).
10. Published-version claims for co-scientist (Nature DOI verified; content not read), AI Scientist v2 acceptance claim.

## 8 Recommendations

### (a) Meta-research skill: reusable procedure for measuring and optimising the research process

Grounded in the failure modes above (verification bottleneck, overexcitement, implementation drift, post-hoc selection, evaluator bias) and in the standards of section 4.4.

1. **Instrument every cycle.** Log per task: wall-clock, tool calls, retries, 429/rate-limit events, tokens/compute, queue wait, and outcome (accepted by auditor / rejected / abandoned). Keep one `process_metrics.jsonl`. (This run alone lost ~40% of its calls to resets and rate-limits — exactly the data to capture.)
2. **Bottleneck identification.** Compute a time/cost breakdown by stage; rank stages by share of wall-clock *and* by failure rate. Re-measure each cycle; a stage is "the bottleneck" only if it dominates both lead-time and rework. Use a simple critical-path view of the task graph.
3. **Root-cause analysis (5-whys with evidence).** For each failed/slow stage write: symptom -> proximate cause -> underlying cause -> evidence (log line/file) -> fix -> predicted metric change. Reject a "root cause" without a falsifiable prediction. Example from this run: symptom "scratch data lost" -> proximate "sandbox reset" -> root "state lived outside the versioned workspace" -> fix "checkpoint stage outputs into the deliverable files every stage".
4. **Pre-register process changes.** Before changing the process, record hypothesis, metric, stopping rule and an unchanged held-out control task set; evaluate on tasks not used for tuning (avoid tuning-on-test, cf. post-hoc selection LIT04-004).
5. **Experiment value-of-information ranking.** Score each candidate experiment: VOI = (prior uncertainty reduction on a decision-relevant claim) x (probability of discriminating rival explanations) / (cost: CPU-hours, wall-clock, risk). Prefer experiments that separate competing hypotheses over those that merely add a positive result. Track predicted vs realised VOI to calibrate the ranker (use Brier score on "did the experiment change the decision?").
6. **Calibration loop.** Record a probability for every hypothesis outcome *before* running; score with Brier/log loss; plot reliability diagrams per claim type; keep the producing agent's confidence separate from the auditor's.
7. **Budget and stop rules.** Per-cycle hard caps (calls, tokens, hours); stop when marginal new-relevant yield of searches falls below 10% over the last 10 queries *and* all sub-areas have been probed (the second condition addresses the sub-area saturation gap found here).
8. **Delegation hygiene.** One owner per artifact; subagents get a task packet and return evidence; same-model roles are logical roles (collapse evidence LIT04-143); anything high-stakes goes to an auditor with *different* inputs/model where possible, and to humans for reward-hacking-prone steps (LIT04-016).
9. **Speed-ups for slow steps.** Cache and batch metadata look-ups (arXiv `id_list` takes up to 50 ids per call); use the cheap authoritative resolver (id_list/Crossref) rather than search for known items; respect rate-limit headers with exponential backoff and a single outbound queue per host; run independent hosts in parallel and the same host serially; write resumable stage outputs (skip done queries); avoid `pkill -f` patterns matching one's own shell.
10. **Meta-research review each cycle:** report unique verified results per agent-hour, audit pass rate, error rate caught by auditors, duplicate-search rate, and cost; change at most one process variable at a time.

### (b) Memory / verification pipeline

1. **Claim records carry provenance:** claim text, scope, type, primary source id, read_depth of the source, supporting/contradicting ids, conditions, method, code/data/log hashes, status (CANDIDATE ... ROBUST/REFUTED), last-verified date, change history. Add `source_read_depth` and `evidence_independence` fields; a claim resting on `abstract_only` evidence can be at most SOURCE_SUPPORTED and must say "authors report".
2. **Two-writer rule:** only the curator writes SCIENTIFIC_KNOWLEDGE; agents propose via task reports; the curator re-checks provenance, contradictions and wording (mitigates "memory contagion" LIT04-054 and poisoning LIT04-053, LIT04-052).
3. **Verification gates per stage** (Trehan et al. design takeaways LIT04-006): idea (novelty search beyond keywords; log the queries), code (execution + unit/leakage tests), results (re-run from raw logs, ground in raw data, not LLM prose), writing (every number traceable to a log line). Gate outputs are pass/fail files in `audits/`.
4. **Pre-registration files** (`PREREG.md` with predictions, metrics, controls, stopping rule) committed *before* running; auditor checks that the reported analysis equals the preregistered one and lists all deviations (post-hoc selection, metric misuse; LIT04-004, LIT04-008).
5. **Independent auditing with blinding:** give the auditor methodology + raw data without the author's conclusion; sample-check LLM-judge decisions against a human/ground-truth subset and report agreement (MAST used kappa; LIT04-122); expect misses (6.5% in LIT04-016); do not use the producing model as sole judge (LIT04-109).
6. **Reward-hacking and containment checks:** read-only evaluation harness, held-out eval code the agent cannot edit, diff review of any metric/eval changes, process-level (not outcome-only) review; sandbox code execution (LIT04-021 sandbox escape; LIT04-016).
7. **Interpretability-specific:** do not accept replication of published circuits as evidence of agent capability without controlling for memorisation (LIT04-101); require causal interventions with random/baseline controls, per the project brief.
8. **Meta-analysis rules in code:** a pooling script that refuses to pool when metrics/benchmark versions/model versions differ or variance is missing, and always reports Q, I^2, tau^2, prediction interval; store per-study effect sizes and CIs (section 4.4).
9. **Integrity checks:** JSONL schema validation, duplicate-claim detection, source-id resolution check against SOURCE_REGISTRY, hash of evidence files, and periodic re-verification of DOIs/arXiv versions (version drift, e.g. co-scientist arXiv title changed on revision).
10. **Persistence against sandbox resets:** each stage writes its output directly into versioned deliverables, plus a `state/research_state.json` checkpoint; commit after each stage.
