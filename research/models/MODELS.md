# Model checkpoints

| role | HF repo | revision (commit sha) | license / gating | files | model.safetensors sha256 (LFS oid) |
|---|---|---|---|---|---|
| primary (instruction-tuned) | `google/gemma-4-E2B-it` | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | ungated; Apache-2.0 (per base card) | model.safetensors 10 246 621 918 B, tokenizer.json 32 169 626 B | `2db5482b20d746879bb3ef79b5203e9075a2e2b98f54ec7c2f281c1477ddc550` |
| secondary (pretrained base) | `google/gemma-4-E2B` | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | ungated; Apache-2.0 | model.safetensors 10 246 621 918 B, tokenizer.json 32 170 070 B | `76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57` |
| control (Gemma Scope compatible) | `google/gemma-2-2b` | `c5ebcd40d208330abc697524c919956e692655cf` | **gated (manual approval)** — not downloaded: no HF token is configured in this project | — | — |

Local sha256 of the downloaded files is logged in `research/logs/download.log` and must equal the LFS oid above.

Re-download: `hf download <repo> --revision <sha> --local-dir .models/<repo>` (set `HF_HUB_ENABLE_HF_TRANSFER=1`).

## Architecture facts used by the tooling (config + transformers 5.20.2 source)
- `Gemma4ForConditionalGeneration` = text decoder + audio tower (0.61 GB) + vision tower (0.34 GB). Only
  `model.language_model.*` is loaded (`experiments/lib/gemma_loader.py`).
- Text decoder: 35 layers, d_model 1536, 8 query heads / 1 KV head, head_dim 256 (512 for full-attention layers),
  sliding window 512 with every 5th layer (4, 9, …, 34) full attention, q/k RMSNorm with attention scale 1.0, v RMSNorm
  without scale, final logit softcap 30, tied embeddings, vocab 262 144.
- KV sharing: layers 15–34 have no k/v projections and reuse K/V stored by layer 13 (sliding) / 14 (full).
  The checkpoint still contains 60 k/v tensors for these layers that the model never reads (stats in E001).
- MLP width 6144 in layers 0–14 and 12 288 ("double wide") in KV-shared layers 15–34.
- Per-layer embeddings (PLE): a 262 144 × (35·256) token lookup table (4.7 GB) plus a projection of the token embedding,
  injected at each layer via gate (1536→256) · PLE → projection (256→1536) → RMSNorm, added to the residual.
- Every layer ends with `hidden_states *= layer_scalar` (a learned scalar per layer; scales the whole residual stream).

## Host constraints (measured in E000)
2 vCPU (AVX-512F, **no AVX512_BF16**), 8 GiB RAM, no GPU. bf16 GEMM is emulated (slow for multi-token batches), so all
causal experiments use `use_fp32_compute` (bf16 storage, fp32 matmuls).
