# NOVELTY-001 — Do the closest full-text sources already report K-0006 / K-0007?
Date 2026-10-25. Method: `tools/novelty_audit.py` — DeepSeek 4.1 Flash and MiniMax M3.1 Flash each read the full text (≤110k chars)
of 7 sources and judged overlap (same / analogous / related / none) with supporting verbatim quotes, machine-verified against
the text. Raw: `audits/NOVELTY-001_raw.json`. Sources: LIT03-019 YOCO, LIT06-005 systematic cross-layer KV sharing, LIT06-016
layer-condensed KV cache, LIT06-012 YOCO++, LIT02-065 Geva et al. 2023 (dissecting factual recall), LIT06-022 ROME in Mamba,
LIT06-023 Mamba knockout.

| result | K-0006 | K-0007 |
|---|---|---|
| "same" judgements | 0 / 14 | 0 / 14 |
| "analogous" | YOCO (architectural: one global KV reused by all upper layers; MM), ROME-in-Mamba (two-site pattern; MM) | Geva et al. (attribute extraction by upper-layer attention heads; MM), ROME-in-Mamba (MM) |
| quotes verified | 25 of 30 exact/near | — |

Coordinator reading: K-0007 is a model-specific instance of Geva et al.'s extraction stage ("extraction is performed by attention
heads", verified quote). What the checked literature does not contain: (i) a causal test of *which* of two shared K/V sources carries
subject information in a KV-sharing model, (ii) concentration of extraction into one layer and one head among 20 KV-shared
layers, (iii) the same localisation in base and IT checkpoints. Scope: 7 sources only — novelty is bounded, not established.
Agreement between auditors on overlap level: identical on 8/14 judgements; MiniMax rated "analogous" more often.
