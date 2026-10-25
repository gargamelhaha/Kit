# LIT-03 — Gemma family literature + tooling compatibility (cycle 1, 2026-10-25)

Role: Literature Scout + Tool Compatibility Analyst. All authors' claims are attributed ("authors report"); nothing below is an experimental result of this lab.

## 1 What was done
- Read the Gemma 4 E2B-it model card, config.json, tokenizer/generation config; downloaded and parsed the Gemma 4 technical report (arXiv:2607.02770v2) and the Gemma Scope 2 technical paper (PDF); read the relevant classes of transformers `modeling_gemma4.py` / `configuration_gemma4.py` (pinned to commit 50e35cc0ca18ec19b80653690de553e6decf74f5, `main` on 2026-10-25) and grepped `modeling_gemma3n.py`.
- Enumerated Gemma Scope / Scope 2 and community Gemma 4 / 3n SAE/transcoder repos through the HF API; inspected README/cfg of the Gemma 4 SAE repos.
- Checked TransformerLens, nnsight, SAELens, circuit-tracer, pyvene (releases, README/source greps, issue/PR searches).
- Literature protocol: foundational -> recent -> backward (S2 references; reference lists of Gemma 4 and Gemma Scope 2 PDFs) -> forward (OpenAlex `cites:` and S2 citations) -> critiques -> code -> dedup -> gaps. Deliverables: `research/evidence/LIT-03_sources.csv` (105 rows), `research/logs/search_LIT-03.jsonl` (141 lines), this report.

## 2 Sources checked
105 deduplicated rows: read_depth full_text 2 (HF README+cfg of decoderesearch SAE repos / Solshine scope repo counted as full README reads), partial 16, abstract_only 66, metadata_only 21. By domain: sae 24, gemma-arch 20, critique/evaluation 19, tooling 14, circuits 11, gemma-interp 6, transcoder 4, other 7. 24 rows are flagged critique/negative/replication. Foundational (<=2023): 13; recent 2024-2026 arXiv: 56. (Target was 40-90; above it because tooling/issue/repo rows were kept as evidence.) **Only the Gemma 4 report (Sec. 2, Table 1, refs) and Gemma Scope 2 paper (intro/Sec. 2/widths table) were read beyond abstracts; every other arXiv row is abstract-only.**

## 3 Actions run
- arXiv API: title/abstract searches and 3 `id_list` verification batches (arXiv returned HTTP 429 for ~10 min because the sandbox IP is shared with other lit agents; resumed later). OpenAlex: 80 search queries + anchor forward-citation queries (Gemma Scope x2 records, Transcoders, Gemma 3, Gemma 4, MatFormer x2). OpenAlex has **no referenced_works for arXiv-based records**, so backward search used Semantic Scholar references (Gemma Scope: 94; Transcoders: 56) and the PDF reference lists. S2 citations: Gemma Scope (100), Transcoders (100), Gemma 4 (100). S2 for Gemma 3 report was rate-limited and was stopped (not completed).
- HF API: `search=gemma-scope|gemma-scope-2|gemma4 sae|gemma-4-E2B sae|gemma-3n sae|gemma4 transcoder|gemma-4 crosscoder|gemma4 steering` + per-repo metadata. GitHub API/raw: releases, issue/PR search, README/source greps. WebSearch x2.
- **Incident:** the sandbox was reset mid-task (scratch and ~53 early search log lines lost). All key fetches were re-run post-reset; the log contains one `Pre-reset summary` line for the lost lines. GitHub issue/PR search results (TransformerLens/nnsight/circuit-tracer) were read pre-reset; their URLs were re-verified HTTP 200 post-reset but the issue bodies were not re-read.
- No models downloaded, no experiments run, no packages installed.

## 4 Results

### 4.1 Gemma 4 E2B text decoder — architecture spec
Refs: **C**=config.json of `google/gemma-4-E2B-it` (sha 3e22461f65e8…, created 2026-03-02, lastModified 2026-07-20); **M**=`modeling_gemma4.py`@50e35cc; **Cf**=`configuration_gemma4.py`@50e35cc; **T**=tech report 2607.02770v2 Sec. 2; **MC**=model card.
| Item | Value | Evidence |
|---|---|---|
| Class | `Gemma4ForConditionalGeneration` (text path `model.language_model`, `lm_head` tied to `model.language_model.embed_tokens`) | C; M:2526 |
| Size | 35 layers, hidden 1536, 8 Q heads, 1 KV head, head_dim 256 (512 on full-attention layers via `per_layer_config`), MLP intermediate 6144, vocab 262144, GELU-tanh, rms eps 1e-6 | C; Cf:210-222 |
| Layer pattern | full attention at layers 4,9,14,19,24,29,34; others sliding window 512 (4:1; last layer forced full) | C `layer_types`; T §2 ("4-to-1 for E2B"); Cf:191-201; MC |
| RoPE | sliding: default, theta 1e4; full: "proportional" p-RoPE, partial_rotary_factor 0.25, theta 1e6 | C `rope_parameters`; T §2 |
| Attention | q_norm (scaled), k_norm (scaled), v_norm (no scale) per head; `scaling=1.0` (no 1/sqrt(d)); no attention-logit softcap in text attention (the `softcap` code at M:284-349 is the audio tower) | M:1151-1255 |
| Block (sandwich norms) | `h = x + post_attention_layernorm(attn(input_layernorm(x)))`; `h = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))`; then PLE block; then `h *= layer_scalar` | M:1384-1429 |
| RMSNorm | `x*rsqrt(mean(x^2)+eps)*weight` — **no (1+weight)** offset (unlike Gemma 1-3) | M:205-223 |
| `layer_scalar` | per-layer learned scalar buffer multiplying the **whole residual stream** at layer end (after PLE) | M:1356, 1429 |
| Embedding | `embed_tokens` output scaled by sqrt(1536) | M:1565 |
| PLE | `embed_tokens_per_layer`: Embedding(262144, 35*256) scaled by sqrt(256); `per_layer_model_projection` Linear(1536->35*256)*1536^-0.5, reshape [B,T,35,256], RMSNorm(256); combined `(proj + token_embedding)*2^-0.5`. In layer i: `gate=act(Linear_{1536->256}(h))`, `*per_layer_input[:,:,i]`, `Linear_{256->1536}`, `post_per_layer_input_norm`, added to residual. Param count consistent with T Table 1 (400M + 2,340M embedder params). | M:1574-1594, 1700-1779, 1420-1428; T Table 1; MC |
| KV sharing | `num_kv_shared_layers=20` -> first shared layer 15. Layers 15-34 have no k_proj/v_proj/k_norm/v_norm; they read `shared_kv_states[layer_type]`, written by the **last non-shared layer of that type: sliding layers (15-18,20-23,25-28,30-33) use layer 13's K/V; full layers (19,24,29,34) use layer 14's K/V** (mapping derived by applying M's rule to C `layer_types`). Stored K is post-k_norm+RoPE, V is post-v_norm. Queries and o_proj stay per-layer. | M:1169-1176, 1222-1244; T §2 ("20/35"); Cf:104 |
| Double-wide MLP | `use_double_wide_mlp` doubles `intermediate_size` (6144->12288) **only for KV-shared layers 15-34** | M:1060-1076 |
| Softcapping | final logits `30*tanh(logits/30)` after `lm_head` | M:1854-1857, 2618-2621; C |
| Final norm | `self.norm` (RMSNorm) after the last layer | M:1692 |
| AltUp / LAuReL / MatFormer / activation sparsity | **absent** from Gemma 4 code (zero symbol hits). Present in Gemma 3n code (`Gemma3nTextAltUp`, `Gemma3nTextLaurelBlock`, `activation_sparsity_pattern`); Gemma 3n docs cite PLE + MatFormer. | grep M; modeling_gemma3n.py:932,988,960,1289-1290; ai.google.dev/gemma/docs/gemma-3n |
| K=V | `attention_k_eq_v=false` for E2B (model card text on "unified keys/values" applies to larger models; T: values=keys "except in E2B and E4B") | C; T §2; M:1164 |
| Footprint | `model.safetensors` 10,246,621,918 bytes (bf16, incl. vision/audio towers and the 2.35B-param PLE table) vs 8 GiB sandbox RAM | HF API |
| Tokenizer | tokenizer_config `add_bos_token: null`; nnsight issue 782 reports Gemma 4 tokenizers do not prepend `<bos>` (id 2) on plain text (verified by reporter on 12B-it, not on E2B) | HF raw; issue 782 |

### 4.2 Where hooks must be placed (residual stream)
- **resid_post[i]**: output of `model.language_model.layers[i]` (a `Gemma4TextDecoderLayer`; HF itself records `hidden_states` on this class, M:1552-1556). This already includes the PLE contribution and the `layer_scalar` multiplication. This is the site (`hook_name="model.language_model.layers.17"`) of the decoderesearch SAEs.
- `output_hidden_states`: entry 0 is the input of layer 0 (scaled embeddings); the **last** entry is overwritten with the post-final-norm `last_hidden_state` (`tie_last_hidden_states`, output_capturing.py:113-122, 234-246). Do not read the last hidden state as pre-norm resid_post[34].
- **resid_mid[i]**: input of `pre_feedforward_layernorm` (pre-hook). **attn_out**: output of `post_attention_layernorm` (post-norm, what is added to the stream); raw `o_proj` output is pre-norm. **mlp_out**: output of `post_feedforward_layernorm`. **ple_out**: output of `post_per_layer_input_norm`.
- Consequences specific to E2B: (a) PLE re-injects token identity into every layer additively, so ablating/patching the residual at layer L does not remove token-ID information from later layers; patch `per_layer_inputs` too (HF accepts `per_layer_inputs` only together with `inputs_embeds`, M:1632-1633, and PLE lookup needs `input_ids`, otherwise it reverse-matches the embedding table); (b) K/V patching must target layers 13 (sliding) and 14 (full) — layers 15-34 cannot have independent K/V; (c) layer_scalar makes residual norms layer-dependent, so activation scale for SAEs/probes must be checked per layer; (d) logit lens must apply `self.norm`, the tied unembedding, then the softcap; (e) sliding-window 512 is equivalent to causal attention for prompts <512 tokens (to be verified).

### 4.3 Tool compatibility matrix
"No evidence" = nothing found in README/source/issues on 2026-10-25, **not** proof of incompatibility. Nothing was executed.
| Tool | Supports gemma4? | Evidence | Version |
|---|---|---|---|
| transformers (HF) | Yes (native `gemma4` model type) | M@50e35cc (URL in CSV LIT03-004); C `transformers_version 5.5.0.dev0` | main@50e35cc; checkpoint exported with 5.5.0.dev0 |
| TransformerLens (TransformerBridge) | Yes for E2B/E4B/31B/26B-A4B/12B via `Gemma4ArchitectureAdapter`; only "phases 1,2,4" (HF parity, hooks, text quality) — no LN-folding/processed mode because PLE, layer_scalar, MoE are not fold-safe; MLP hook aliases fixed in v3.7.2; dtype-conversion gap for PLE weights (issue 1935, closed) | https://github.com/TransformerLensOrg/TransformerLens/pull/1385 ; .../issues/1646 ; .../issues/1935 ; `transformer_lens/model_bridge/supported_architectures/gemma4.py` | v3.5.0 (2026-06-29) first with Gemma 4; v3.7.2 (2026-08-15); latest v4.3.0 (2026-10-23); `HookedTransformer` removed in 4.0.0 |
| nnsight | Works on Gemma 4 12B-it via NDIF (issue); **BOS pitfall** (open issue 782); no E2B-local test found | https://github.com/ndif-team/nnsight/issues/782 | v0.8.1 (2026-10-20) |
| SAELens | Loads Gemma 4 SAEs by repo id (`SAE.from_pretrained("decoderesearch/gemma-4-saes", id)`, trained with SAELens 6.39.0); `pretrained_saes.yaml` has 0 gemma-4 entries (168 gemma-scope-2 entries) | https://huggingface.co/decoderesearch/gemma-4-saes ; SAELens README; yaml grep | v6.54.12 (2026-10-20) |
| circuit-tracer | No evidence (README 0 mentions of Gemma 4; transcoder list = Gemma-2 2B, Gemma-3 via nnsight backend, Llama, Qwen3, GPT-OSS; no Gemma 4 issue/PR); no Gemma 4/3n transcoders on HF | https://github.com/decoderesearch/circuit-tracer | v0.5.3 (2026-10-14) |
| pyvene | No evidence (model card maps only `gemma`, `gemma2`; 0 matches for gemma3/gemma4); generic module-path interventions may still work | stanfordnlp/pyvene `intervenable_modelcard.py` grep | v0.1.8 (2025-05-26) |
| Neuronpedia | Hosts Gemma Scope 2 (Gemma 3); Gemma 4 hosting not checked | https://www.neuronpedia.org/gemma-scope-2 | v2.8.1 |
| Gemma Scope 1 / 2 weights | **Not applicable** to Gemma 4 / 3n (trained for Gemma 2 / Gemma 3) | HF README of Gemma Scope 2; HF API search | — |

### 4.4 Pretrained SAEs / transcoders and their targets
| Repo | Target model | Layers / site | Details | Status |
|---|---|---|---|---|
| `decoderesearch/gemma-4-saes` | **google/gemma-4-E2B (base)** | L6, L17, L28 / layer output (`model.language_model.layers.N`) | Matryoshka BatchTopK k=100, width 65,536 (prefixes 2k/16k), Pile-Uncopyrighted, 1e9 tokens, ctx 1024, prepend_bos true, ~806 MB each; same repo has E4B (L7,21,35) and 31B (L30) | README + cfg read; **no quality metrics read** |
| `Solshine/gemma-4-e2b-scope-v1-L17-batchtopk-k64-seed17` | E2B base | L17 resid | BatchTopK, 24,576 wide, k=64, trained under NF4 4-bit; 16M of 100M tokens; self-reported EV 0.83 | self-reported |
| `Solshine/gemma4-e2b-biosec-sae-v1` | E2B-IT | L17 resid post-MLP | TopK 6,144 wide, k=32, contrastive bio objective, HL3 license | domain-specific |
| `Solshine/gemma4-e2b-bio-sae-v2-auxk`, `-v3-auxk` | E2B (card absent) | unknown | metadata only | unread |
| `elrashid/sae-gemma-4-e2b-it-topk-l17` | E2B-IT | L17 | access restricted (gated manual) | unusable |
| Gemma 4 E4B/31B/audio community SAEs (michaelwaves L14, caiovicentino1 L21, lamm-mit L20, atawfeek, ddasd 31B, praphul555 audio, MettaMazza) | E4B/31B/other | various | metadata only | not target |
| Google Gemma Scope 2 (`google/gemma-scope-2-*`) | Gemma 3 270M/1B/4B/12B/27B, PT+IT | SAEs at attention-out, MLP-out, post-MLP residual on all layers; skip-transcoders; partial crosscoders (4 layers at fixed depth percentiles); CLTs for small models | JumpReLU; widths 16k-1M (paper table) | official, Gemma 3 only |
| Google Gemma Scope 1 (`google/gemma-scope-{2b,9b,27b}-*`) | Gemma 2 2B/9B/27B | res/att/mlp (+2b transcoders) | arXiv 2408.05147 | official, Gemma 2 only |
| `mntss/gemma-scope-transcoders`; mwhanna Gemma-3 collection | Gemma 2 2B; Gemma 3 | per-layer transcoders for circuit-tracer | | community re-packaging |
| Anything for Gemma 3n | — | HF search "gemma-3n sae" -> 0 results | | none found |
| Gemma 4 transcoders / crosscoders | — | HF searches -> 0 results | | none found |
Note: layer 17 (the community SAEs' main layer) lies in the **KV-shared region** (layers >=15); layer 6 and 28 straddle it.

### 4.5 Published interpretability findings on Gemma models (authors' claims; abstract-level reading unless stated)
- Gemma Scope (2408.05147): open JumpReLU SAEs across layers/sites of Gemma 2 2B/9B. Gemma Scope 2 paper (read partially): SAEs, skip-transcoders, crosscoders, CLTs for Gemma 3; IT SAEs fine-tuned on chat data.
- 2504.00132: circuits for in-context learning in Gemma-2 2B ("contextualize-then-aggregate").
- 2609.04808: SAE translation-initiation features in Gemma 2/3: >20 recurring features, but only one per model (Gemma 2 L10/5717; Gemma 3 L20/2456) causally consistent over 23 language settings — authors say recurrence overstates cross-lingual transfer (replication-style critique).
- 2606.13705 (the only paper found that names Gemma 4 E2B in a causal intervention): repetition loops on enumeration prompts; per-layer ablation + neuron attribution; in E2B one sign-inverted neuron suffices on the selection prompts; across four Gemma 4 models held-out loops 46/384 -> 12/384; authors note rankings vary across examples.
- 2605.30162: Gemma 4 E2B-IT refused 65/75 prompts with chat-template formatting and 0/75 without; both Gemma models collapsed to 0% under an 80-token cap; uses the community E2B-IT SAE for an internal divergence score.
- Critiques/negatives relevant to any SAE-on-Gemma-4 plan: AxBench 2501.17148, SAE sparse probing 2502.16681, GDM negative-results post (snippet only), SAE sanity checks 2602.14111, random-init 2501.17727, SAE intervention recovery 2606.18322, refusal not a single direction 2602.02132 / 2610.04245, causality of SAE features 2607.20596, circuit consistency 2605.08348 / 2607.18921.
- No published Gemma 4 circuit-tracing, attribution-graph, logit-lens or activation-patching study was found (see gaps).

## 5 Evidence
Files: `research/evidence/LIT-03_sources.csv`, `research/logs/search_LIT-03.jsonl`. Raw sources: HF API/README/config of `google/gemma-4-E2B-it`; arXiv 2607.02770v2 PDF; transformers `modeling_gemma4.py` lines cited in 4.1 (commit-pinned URL in CSV row LIT03-004); decoderesearch SAE `cfg.json`/`runner_cfg.json`; GitHub issues/PRs listed in 4.3. Every URL in the CSV returned HTTP 2xx at check time, except the Medium mirror of the GDM post (403 to curl; the LessWrong URL is used).

## 6 Limitations & saturation
- Saturation: last 10 logged queries returned 161 new records of which 2 (1.2%) entered the CSV; last 21 queries: 211 new, 3 relevant (1.4%) — below the 10% criterion. Caveat: relevance judged on titles/abstracts by one agent; forward citation tools (OpenAlex/S2) are noisy and under-index 2026 papers.
- Backward AND forward search done for Gemma Scope, Transcoders, Gemma 4 (forward via S2+OpenAlex; backward via S2 or PDF refs); Gemma 3 report forward via OpenAlex only; MatFormer forward via OpenAlex only.
- Read depth is shallow: 66 rows abstract-only; Anthropic transformer-circuits pages are metadata-only (title verified; authors/dates not extracted). Gemma Scope 2 has no arXiv version found. Dates for two community HF collections are placeholders (flagged in notes).
- OpenAlex lacks reference lists for arXiv records; S2 rate-limited (429); arXiv 429 from shared IP. First-session log lines lost (see 3).
- Not checked: Gemma 3n SAE/interp literature beyond HF search; Gemma 4 vision/audio towers; the 12B/26B/31B variants; Gemma 4 MTP drafter; TL/nnsight behavior on E2B by execution.

## 7 Claims needing verification (empirically on the loaded model)
1. KV mapping: sliding layers 15-34 use layer 13's K/V and full layers use layer 14's (hook `shared_kv_states` via `return_shared_kv_states`, compare tensors).
2. `layer_scalar` values and per-layer residual norms; whether decoderesearch SAEs (trained on base) reconstruct E2B-IT activations acceptably (FVU, delta-LM-loss) at L6/L17/L28 — base->IT transfer untested.
3. Hook-site identity: layer-output hook == `hidden_states[i+1]` (and last entry is post-norm).
4. Tokenizer: does E2B's tokenizer add `<bos>` for plain text? SAEs expect prepend_bos=true; chat template behavior (2605.30162 reports 65/75 vs 0/75 refusals with/without template).
5. Logit-lens validity with tied embeddings + final norm + softcap 30; whether a tuned lens is needed for early layers.
6. PLE ablation: effect of zeroing/mean-ablating `per_layer_inputs` vs residual-only patching; whether token-ID information is recoverable after residual ablation.
7. Sliding window 512 ≡ causal attention for short prompts; attention mask construction matches.
8. TransformerBridge parity vs HF on CPU (E2B) and the PLE dtype issue (1935) on installed version; memory feasibility: 10.2 GB bf16 file on 8 GiB RAM (mmap/offload of PLE table; text-only load).
9. 2606.13705's single-neuron repetition fix reproduces on E2B (needs exact prompts/seeds, code link not verified).
10. Whether any Gemma 4 SAE metrics claimed by community repos (EV 0.83, etc.) hold.

## 8 Next actions
1. Install-time check (no weights): pin transformers version matching `5.5.0.dev0`-compatible release; test `AutoConfig` + meta-device instantiation of `Gemma4ForCausalLM` to confirm module names/paths and parameter counts (cheap, no download).
2. After the model loads: run claims 1, 3, 4, 5 as a smoke-test suite (all CPU-cheap), then claim 2 on decoderesearch L17 SAE.
3. Reproduction targets with published Gemma results: Gemma-2-2B ICL circuit (2504.00132) and Gemma Scope SAEs on Gemma 2 2B (tool validation), then Gemma 4 E2B repetition-neuron study (2606.13705).
4. Full-text read of AxBench, SAE sanity checks, random-init, SAEBench, activation-patching best practices before designing causal tests; read Gemma Scope 2 evaluation sections.
5. If circuit-level analysis on E2B is wanted: no transcoders exist — budget for training per-layer transcoders (CLT-Forge 2603.21014 as candidate) or restrict to SAE + patching.
