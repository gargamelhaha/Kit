# Agent instructions

This repository is an autonomous research lab. At the start of every session read:
1. `research/state/research_state.json` — current cycle, open tasks, next step.
2. `research/README.md` — configuration and how to run things.
3. `.claude/skills/meta-research/SKILL.md` — **mandatory** process discipline (durability, instrumentation,
   root-cause analysis, value-of-information experiment choice, meta-analysis rules). Follow it in every cycle.

Rules
- Commit and push after every completed step; the sandbox can be lost without warning.
- Verified knowledge only in `research/SCIENTIFIC_KNOWLEDGE.md` + `research/scientific_knowledge.jsonl` (curator role,
  after evidence checks); hypotheses in `research/HYPOTHESIS_REGISTER.md` + `research/hypotheses.jsonl`.
  Run `python research/tools/kb.py validate` after any registry change.
- Preregister experiments (`PREREG.md`) before running. Never delete data from failed experiments.
- Weights live in `.models/` (gitignored, see `research/models/MODELS.md` to re-download). Load them with
  `research/experiments/lib/gemma_loader.py`; use `use_fp32_compute` for batched/causal work and `encode()` (adds BOS).
- One model process at a time on the 8 GiB box (`research/experiments/queue.sh`); keep `research/tools/memwatch.sh` running.
