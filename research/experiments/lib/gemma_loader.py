"""Memory-lean loader for the Gemma 4 E2B text decoder on CPU.

Loads only `model.language_model.*` weights into a Gemma4ForCausalLM. All weights are zero-copy,
read-only memory maps of the safetensors file (clean file-backed pages the kernel can reclaim), so
anonymous memory stays ~1-2 GB even though the text decoder is 9.3 GB in bf16 (4.7 GB of it is the
per-layer-embedding lookup table). Cycle-1 lesson: copying weights to RAM put peak RSS at 8.6 GB on an
8 GiB box and the sandbox was lost.
"""
import json, os, struct, time, warnings
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".models"))
MODELS = {"it": "google/gemma-4-E2B-it", "base": "google/gemma-4-E2B"}


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def _tensor(path, info, base, mmap=True):
    assert info["dtype"] == "BF16", info["dtype"]
    s, _ = info["data_offsets"]
    arr = np.memmap(path, dtype=np.uint16, mode="r", offset=base + s, shape=tuple(info["shape"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-writable numpy array warning
        t = torch.from_numpy(arr).view(torch.bfloat16)
    return t if mmap else t.clone()


def load(variant="it", mmap=True):
    t0 = time.time()
    path = os.path.join(ROOT, MODELS[variant])
    cfg = AutoConfig.from_pretrained(path).text_config
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(path)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg)
    st = os.path.join(path, "model.safetensors")
    h, base = _header(st)
    sd = {k: _tensor(st, h["model.language_model." + k[len("model."):]], base, mmap)
          for k in model.state_dict() if k != "lm_head.weight"}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    model.tie_weights()
    _rebuild_meta_buffers(model, cfg)
    leftover = [n for n, b in list(model.named_parameters()) + list(model.named_buffers()) if b.device.type == "meta"]
    model.eval()
    return model, tok, {"variant": variant, "repo": MODELS[variant], "missing": [m for m in missing if m != "lm_head.weight"],
                        "unexpected": unexpected, "meta_leftover": leftover, "load_s": round(time.time() - t0, 1)}


def _rebuild_meta_buffers(model, cfg):
    from transformers.models.gemma4 import modeling_gemma4 as mg
    for name, mod in model.named_modules():
        for bname, buf in list(mod._buffers.items()):
            if buf is None or buf.device.type != "meta":
                continue
            if isinstance(mod, mg.Gemma4TextRotaryEmbedding):
                mod._buffers[bname] = getattr(mg.Gemma4TextRotaryEmbedding(cfg, device="cpu"), bname)
            elif bname == "embed_scale":
                mod._buffers[bname] = torch.tensor(mod.scalar_embed_scale, dtype=torch.float32)
            else:
                raise RuntimeError(f"unhandled meta buffer {name}.{bname}")


def use_fp32_compute(model):
    """Keep bf16 weights but run every matmul in fp32.

    Root cause of slow batched inference on this host: the CPU lacks AVX512_BF16, so bf16 GEMMs are
    emulated (measured 8 vs 104 GFLOP/s at 32 tokens). Upcasting each layer's weights per call is
    cheaper and gives fp32 activations (better numerics for patching/ablation deltas).
    """
    import torch.nn.functional as F
    from torch import nn
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod.forward = (lambda m: (lambda x: F.linear(x.float(), m.weight.float())))(mod)
        elif isinstance(mod, nn.Embedding):
            mod.register_forward_hook(lambda m, i, o: o.float())
    for p in model.parameters():
        if p.dim() == 1:
            p.data = p.data.float()
    for mod in model.modules():
        for bname, buf in mod._buffers.items():
            if buf is not None and buf.dtype == torch.bfloat16 and buf.numel() < 1e6:
                mod._buffers[bname] = buf.float()
    model.config.dtype = torch.float32
    return model


def encode(tok, texts, add_bos=True):
    """Tokenize with an explicit BOS: this tokenizer does not add one by default."""
    if isinstance(texts, str):
        texts = [texts]
    return [([tok.bos_token_id] if add_bos else []) + tok(t, add_special_tokens=False).input_ids for t in texts]
