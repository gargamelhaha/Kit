"""Memory-lean loader for the Gemma 4 E2B text decoder on CPU.

Loads only `model.language_model.*` weights into a Gemma4ForCausalLM. Decoder weights and the token
embedding (4.6 GB bf16) are copied to RAM; the 4.7 GB per-layer-embedding (PLE) table stays a read-only
memory map, since a forward pass only reads the rows of the tokens present.
Cycle-1 lessons (PROCESS_LEDGER): memory-mapping *all* weights made forward passes I/O-bound on this
sandbox's filesystem (~10% CPU); the 8.6 GB peak that preceded the sandbox loss came from transient fp32
copies (e.g. the 1.6 GB unembedding upcast), which `use_fp32_compute` now avoids by chunking.
"""
import json, os, struct, time, warnings
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".models"))
MODELS = {"it": "google/gemma-4-E2B-it", "base": "google/gemma-4-E2B"}
MMAP_KEYS = {"model.embed_tokens_per_layer.weight"}


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


class PreadEmbedding(torch.nn.Module):
    """PLE token table read row-by-row with os.pread instead of a memory map (PL-13).
    In the restored sandbox (overlay FS under gVisor) touching one row of an mmapped file made the *whole* 4.7 GB
    mapping resident (RSS 0.07 -> 4.77 GB after a single-row read), which tripped the RSS guard on the first forward.
    pread only ever materialises the rows of the tokens present (17.9 KB per token)."""
    def __init__(self, path, offset, shape, scale):
        super().__init__()
        self.path, self.offset, self.shape, self.scalar_embed_scale = path, offset, tuple(shape), scale
        self.fd = os.open(path, os.O_RDONLY)
        self.row_bytes = self.shape[1] * 2
        self.embed_scale = torch.tensor(scale)
        self.weight = torch.empty(0, self.shape[1], dtype=torch.bfloat16)  # placeholder: dtype/device queries only

    def forward(self, ids):
        uniq, inv = torch.unique(ids, return_inverse=True)
        buf = bytearray(len(uniq) * self.row_bytes)
        for j, t in enumerate(uniq.tolist()):
            buf[j * self.row_bytes:(j + 1) * self.row_bytes] = os.pread(self.fd, self.row_bytes, self.offset + t * self.row_bytes)
        rows = torch.frombuffer(buf, dtype=torch.bfloat16).view(len(uniq), self.shape[1])
        return rows[inv] * self.embed_scale.to(torch.bfloat16)


def load(variant="it"):
    t0 = time.time()
    path = os.path.join(ROOT, MODELS[variant])
    cfg = AutoConfig.from_pretrained(path).text_config
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(path)
    with torch.device("meta"):
        model = Gemma4ForCausalLM(cfg)
    st = os.path.join(path, "model.safetensors")
    h, base = _header(st)
    sd = {k: _tensor(st, h["model.language_model." + k[len("model."):]], base, mmap=False)
          for k in model.state_dict() if k != "lm_head.weight" and k not in MMAP_KEYS}
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    ple_info = h["model.language_model.embed_tokens_per_layer.weight"]
    old = model.model.embed_tokens_per_layer
    model.model.embed_tokens_per_layer = PreadEmbedding(st, base + ple_info["data_offsets"][0], ple_info["shape"],
                                                        float(old.scalar_embed_scale))
    missing = [m for m in missing if m not in MMAP_KEYS]
    _drop_page_cache(st)
    model.tie_weights()
    _rebuild_meta_buffers(model, cfg)
    leftover = [n for n, b in list(model.named_parameters()) + list(model.named_buffers()) if b.device.type == "meta"]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    start_rss_guard()
    return model, tok, {"variant": variant, "repo": MODELS[variant], "missing": [m for m in missing if m != "lm_head.weight"],
                        "unexpected": unexpected, "meta_leftover": leftover, "load_s": round(time.time() - t0, 1)}


def start_rss_guard(limit_gb=float(os.environ.get("RSS_GUARD_GB", 6.8)), period=0.2):
    """Exit this process before it can push the 8 GiB sandbox over its box limit (the sandbox does not
    enforce RLIMIT_DATA, E005). Logs the event to research/logs/memwatch.log."""
    import threading
    if getattr(start_rss_guard, "started", False):
        return
    start_rss_guard.started = True
    page = os.sysconf("SC_PAGE_SIZE")
    log = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "memwatch.log")
    def loop():
        peak = 0.0
        while True:
            rss = int(open("/proc/self/statm").read().split()[1]) * page / 1e9
            peak = max(peak, rss)
            start_rss_guard.peak = peak
            if rss > limit_gb:
                with open(log, "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} rss_guard: RSS {rss:.2f} GB > {limit_gb} GB, exiting pid {os.getpid()}\n")
                os._exit(3)
            time.sleep(period)
    threading.Thread(target=loop, daemon=True).start()


def _drop_page_cache(path):
    """Ask the kernel to drop cached pages of the checkpoint after the RAM copy (best effort; the sandbox
    may account page cache against its memory box). PLE rows are re-read on demand."""
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass


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
    if getattr(model, "_fp32_compute", False):
        return model
    model._fp32_compute = True
    # Persistent fp32 buffers, one per weight shape: no per-call allocation churn (PL-6: repeated
    # 75-200 MB allocate/free cycles are the leading suspect for whole-sandbox kills under gVisor).
    bufs = {}
    def fp32_weight(w, rows=None):
        key = (rows or w.shape[0], w.shape[1])
        if key not in bufs:
            bufs[key] = torch.empty(key, dtype=torch.float32)
        b = bufs[key][: w.shape[0]]
        b.copy_(w)
        return b
    def chunked(m, x, rows=32768):  # 262k-row unembedding through one reused 200 MB buffer
        x = x.float()
        return torch.cat([F.linear(x, fp32_weight(m.weight[r:r + rows], rows)) for r in range(0, m.weight.shape[0], rows)], dim=-1)
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            if mod.weight.shape[0] > 65536:
                mod.forward = (lambda m: (lambda x: chunked(m, x)))(mod)
            else:
                mod.forward = (lambda m: (lambda x: F.linear(x.float(), fp32_weight(m.weight))))(mod)
        elif isinstance(mod, nn.Embedding):
            mod.register_forward_hook(lambda m, i, o: o.float())
    model._fp32_weight = fp32_weight
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
