# Autonomous research lab — Gemma 4 E2B internals & research-process science

**Programs.** (1) Mechanistic study of Gemma 4 E2B (`google/gemma-4-E2B-it`; base `google/gemma-4-E2B`) with verified,
provenance-tracked findings. (2) Improving the research process itself (see `.claude/skills/meta-research/SKILL.md`).
(3) Survey of new architectures for autonomous / efficient learning (LIT-05; small-scale CPU tests planned).

## Current configuration (cycle 1, 2026-10-25)
- Host: Modal sandbox, 2 vCPU (AVX-512F, no AVX512_BF16), 8 GiB RAM box (cannot be raised), no GPU, network available.
- Python 3.12, torch 2.14.1+cpu, transformers 5.20.2 (`research/logs/install_*.log`).
- Weights: `.models/` (gitignored) — revisions and sha256 in `models/MODELS.md`. Gemma 2 2B control model **not available**
  (gated; no HF token configured), so Gemma Scope replication is blocked.

## How to run
```bash
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install transformers==5.20.2 numpy scipy pandas matplotlib statsmodels huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download google/gemma-4-E2B-it --revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 --local-dir .models/google/gemma-4-E2B-it
research/tools/memwatch.sh &                                   # protects the sandbox
cd research/experiments && python run_many.py E002_logit_lens:it E004_factual_patching:it   # one model process, auto-push per experiment
python analyze_cycle1.py                                        # preregistered criteria → <exp>/analysis.json
python ../tools/kb.py validate && python ../tools/kb.py render  # knowledge-base integrity + regenerate SCIENTIFIC_KNOWLEDGE.md
```
Library: `experiments/lib/gemma_loader.py` (text-only loader, PLE table memory-mapped, `use_fp32_compute`, `encode` adds BOS,
in-process RSS guard), `experiments/lib/runner.py` (hookable forward: resid/attn/mlp/PLE/head hooks, `last_only_from=15`).

## Layout
| path | content |
|---|---|
| `LITERATURE_MAP.md`, `SOURCE_REGISTRY.csv` | literature map; 631 deduplicated sources with read depth (`evidence/source_dedup_map.csv`, `evidence/source_overrides.csv`) |
| `SCIENTIFIC_KNOWLEDGE.md`, `scientific_knowledge.jsonl` | curated verified claims (md is generated from jsonl) |
| `HYPOTHESIS_REGISTER.md`, `hypotheses.jsonl` | candidate hypotheses |
| `OPEN_QUESTIONS.md` | unresolved questions |
| `experiments/E0xx_*` | PREREG.md, run.py, raw results, analysis.json |
| `agents/tasks`, `agents/reports` | subagent task packets and reports |
| `audits/` | independent audits |
| `meta_analysis/` | methods table, findings map, replication plan, PROCESS_LEDGER, cycle reviews |
| `logs/` | search logs, process metrics, install/download logs |
| `state/research_state.json` | current cycle, open tasks, next step; `state/knowledge.sqlite` is a derived index (`kb.py build`) |
| `tools/` | `kb.py`, `procmetrics.py`, `memwatch.sh` |

`papers/` is intentionally absent: no PDFs were stored (licensing; all sources are linked).
