"""Inference fast paths for HF Gemma 4 E2B (engineering line). Each is a reversible patch with an exactness/quality test.

enable_prefill_skip(model): during any multi-token forward (prefill, chunked prefill, multi-turn continuation) the KV-shared
  layers 15-34 run only at the final position. Exact for next-token logits because those layers write no K/V (K-0001/K-0008);
  NOT valid when logits/hidden states of non-final positions are needed (perplexity/scoring, output_hidden_states).
set_small_batch_bf16(model, max_rows): with use_fp32_compute active, linear layers whose input has <= max_rows rows run as
  native bf16 matmuls (memory-bound decode steps) instead of upcasting weights to fp32. Changes numerics -> quality-tested.
"""
import torch
import torch.nn.functional as F


def enable_prefill_skip(model):
    tm = model.model
    first = model.config.num_hidden_layers - model.config.num_kv_shared_layers
    handles = []

    def pre(mod, args, kwargs):
        h, pli = args[0], args[1]
        if pli is None or h.shape[1] == pli.shape[1] and (mod.layer_idx != first or h.shape[1] == 1):
            return None
        if mod.layer_idx == first:
            h = h[:, -1:]
        pli = pli[:, -1:]
        kw = dict(kwargs)
        kw["position_embeddings"] = tuple(x[:, -1:] for x in kw["position_embeddings"])
        m = kw.get("attention_mask")
        if isinstance(m, torch.Tensor):
            kw["attention_mask"] = m[..., -1:, :]
        if isinstance(kw.get("position_ids"), torch.Tensor):
            kw["position_ids"] = kw["position_ids"][..., -1:]
        return (h, pli) + tuple(args[2:]), kw

    for layer in tm.layers[first:]:
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
    model._prefill_skip_handles = handles
    return handles


def disable_prefill_skip(model):
    for h in getattr(model, "_prefill_skip_handles", []):
        h.remove()
    model._prefill_skip_handles = []


def set_small_batch_bf16(model, max_rows=0):
    """max_rows=0 disables. Requires use_fp32_compute(model) to have been applied first."""
    from torch import nn
    model._bf16_rows = max_rows
    fw = model._fp32_weight
    for mod in model.modules():
        if isinstance(mod, nn.Linear) and mod.weight.shape[0] <= 65536 and not getattr(mod, "_adaptive", False):
            def fwd(x, m=mod):
                if model._bf16_rows and x.numel() // x.shape[-1] <= model._bf16_rows:
                    return F.linear(x.to(torch.bfloat16), m.weight).float()
                return F.linear(x.float(), fw(m.weight))
            mod.forward = fwd; mod._adaptive = True
