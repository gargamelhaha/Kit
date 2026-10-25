"""Hookable forward pass for the Gemma 4 E2B text decoder (equal-length batches, no padding).

Hook points, fn(tensor, layer) -> tensor | None (None keeps the value):
  resid_pre   residual stream entering layer L            [b, t, d]
  attn_out    attention contribution (post-attn norm)      [b, t, d]
  mlp_out     MLP contribution (post-ffw norm)             [b, t, d]
  ple_out     per-layer-embedding contribution             [b, t, d]
  head_in     o_proj input split per head                  [b, t, h, dh]
  resid_post  residual leaving layer L (after layer_scalar)
  attn_probs  read-only attention probabilities [b, h, q, k] (requires attn="eager")
  kv_store    (key, value) [b, 1, t, dh] that layer 13 (sliding) / 14 (full) stores for the KV-shared layers 15-34;
              runs after that layer's own attention, so it changes only what the shared layers read
`last_only_from=K` runs layers >= K on the final position only. Exact for final-position outputs when
K >= first KV-shared layer (15): those layers read K/V produced by layers 13/14 only (verified in E001).
"""
import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask


class Runner:
    def __init__(self, model):
        self.model, self.tm, self.cfg = model, model.model, model.config
        self.L = self.cfg.num_hidden_layers
        self.first_shared = self.L - self.cfg.num_kv_shared_layers
        self.cap = self.cfg.final_logit_softcapping
        self._active = {}
        for i, layer in enumerate(self.tm.layers):
            layer.post_attention_layernorm.register_forward_hook(self._mk("attn_out", i))
            layer.post_feedforward_layernorm.register_forward_hook(self._mk("mlp_out", i))
            layer.post_per_layer_input_norm.register_forward_hook(self._mk("ple_out", i))
            layer.self_attn.o_proj.register_forward_pre_hook(self._mk_head(i))
            layer.self_attn.register_forward_hook(self._mk_probs(i))

    def _mk(self, point, i):
        def hook(mod, inp, out):
            fn = self._active.get((point, i))
            if fn is None:
                return None
            r = fn(out, i)
            return out if r is None else r
        return hook

    def _mk_probs(self, i):
        def hook(mod, inp, out):
            fn = self._active.get(("attn_probs", i))
            if fn is not None:
                fn(out[1], i)
        return hook

    def _mk_head(self, i):
        def hook(mod, args):
            fn = self._active.get(("head_in", i))
            if fn is None:
                return None
            x = args[0]
            b, t, hd = x.shape
            r = fn(x.view(b, t, self.cfg.num_attention_heads, -1), i)
            return None if r is None else (r.reshape(b, t, hd),)
        return hook

    @torch.no_grad()
    def prefix(self, ids, hooks=None, attn="sdpa"):
        """Run layers 0..first_shared-1 on all positions once; return the state the KV-shared suffix needs.
        Every intervention at layers >= 15 or on the stored K/V only changes the suffix, so it can be evaluated with
        suffix() for many conditions without recomputing layers 0-14 (PL-12)."""
        self.cfg._attn_implementation = attn
        self._active = dict(hooks or {})
        tm = self.tm
        emb = tm.embed_tokens(ids)
        ple = tm.project_per_layer_inputs(emb, tm.get_per_layer_inputs(ids, emb))
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        mk = dict(config=self.cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
        pe = {lt: tm.rotary_emb(emb, pos, lt) for lt in tm.unique_layer_types}
        h, shared = emb, {}
        for i in range(self.first_shared):
            lt = self.cfg.layer_types[i]
            fn = self._active.get(("resid_pre", i))
            if fn is not None:
                r = fn(h, i); h = h if r is None else r
            h = tm.layers[i](h, ple[:, :, i, :], shared_kv_states=shared, position_embeddings=pe[lt],
                             attention_mask=masks[lt], position_ids=pos)
            fn = self._active.get(("resid_post", i))
            if fn is not None:
                r = fn(h, i); h = h if r is None else r
        self._active = {}
        return {"h": h[:, -1:].clone(), "shared": {k: tuple(t.clone() for t in v) for k, v in shared.items()},
                "ple": ple[:, -1:].clone(), "pe": {k: tuple(x[:, -1:] for x in v) for k, v in pe.items()},
                "masks": {k: (None if v is None else v[..., -1:, :]) for k, v in masks.items()}, "pos": pos, "T": ids.shape[1]}

    @torch.no_grad()
    def suffix(self, st, n, hooks=None, attn="sdpa", keep_resid=False):
        """Run layers first_shared..L-1 at the final position for n copies of prefix state `st` (batch rows = conditions).
        kv_store hooks for layers 13/14 are applied to the (expanded) stored K/V before the suffix runs."""
        self.cfg._attn_implementation = attn
        self._active = dict(hooks or {})
        shared = {}
        for i in range(self.first_shared):
            if self.tm.layers[i].self_attn.store_full_length_kv:
                lt = self.cfg.layer_types[i]
                kv = tuple(t.expand(n, *t.shape[1:]).clone() for t in st["shared"][lt])
                fn = self._active.get(("kv_store", i))
                r = fn(kv, i) if fn is not None else None
                shared[lt] = kv if r is None else r
        h, resid = st["h"].expand(n, -1, -1).clone(), []
        for i in range(self.first_shared, self.L):
            lt = self.cfg.layer_types[i]
            fn = self._active.get(("resid_pre", i))
            if fn is not None:
                r = fn(h, i); h = h if r is None else r
            if keep_resid:
                resid.append(h[:, 0].clone())
            h = self.tm.layers[i](h, st["ple"][:, :, i, :].expand(n, -1, -1), shared_kv_states=shared,
                                  position_embeddings=st["pe"][lt], attention_mask=st["masks"][lt], position_ids=st["pos"])
            fn = self._active.get(("resid_post", i))
            if fn is not None:
                r = fn(h, i); h = h if r is None else r
        self._active = {}
        out = {"logits": self.unembed(h[:, -1]), "final_h": h}
        if keep_resid:
            out["resid"] = resid + [h[:, 0].clone()]
        return out

    @torch.no_grad()
    def unembed(self, h):
        x, W = self.tm.norm(h).float(), self.model.lm_head.weight
        fw = getattr(self.model, "_fp32_weight", None)  # reuse use_fp32_compute's persistent buffer
        logits = torch.cat([torch.nn.functional.linear(x, fw(W[r:r + 32768], 32768) if fw else W[r:r + 32768].float())
                            for r in range(0, W.shape[0], 32768)], dim=-1)
        return torch.tanh(logits / self.cap) * self.cap if self.cap else logits

    @torch.no_grad()
    def forward(self, ids, hooks=None, last_only_from=None, keep_resid=False, attn="sdpa", logits="last", return_shared=False, allow_long=False):
        if last_only_from is not None and last_only_from < self.first_shared:
            raise ValueError(f"last_only_from must be >= {self.first_shared}: earlier layers still write K/V for other positions")
        self.cfg._attn_implementation = attn
        self._active = dict(hooks or {})
        tm = self.tm
        emb = tm.embed_tokens(ids)
        ple = tm.project_per_layer_inputs(emb, tm.get_per_layer_inputs(ids, emb))
        pos = torch.arange(ids.shape[1]).unsqueeze(0)
        mk = dict(config=self.cfg, inputs_embeds=emb, attention_mask=None, past_key_values=None, position_ids=pos)
        masks = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
        pe = {lt: tm.rotary_emb(emb, pos, lt) for lt in tm.unique_layer_types}
        h, shared, resid, sliced = emb, {}, [], False
        for i, layer in enumerate(tm.layers):
            lt = self.cfg.layer_types[i]
            m, p, pli = masks[lt], pe[lt], ple[:, :, i, :]
            if last_only_from is not None and i >= last_only_from:
                if ids.shape[1] > self.cfg.sliding_window and not allow_long:
                    raise NotImplementedError("last_only mode is only validated for inputs <= sliding window")
                if not sliced:
                    h, sliced = h[:, -1:], True
                pli, p = pli[:, -1:], tuple(x[:, -1:] for x in p)
                m = None if m is None else m[..., -1:, :]
            fn = self._active.get(("resid_pre", i))
            if fn is not None:
                r = fn(h, i); h = h if r is None else r
            if keep_resid:
                resid.append(h.detach().clone())
            h = layer(h, pli, shared_kv_states=shared, position_embeddings=p, attention_mask=m, position_ids=pos)
            fn = self._active.get(("kv_store", i))
            if fn is not None and layer.self_attn.store_full_length_kv:
                r = fn(shared[lt], i)
                if r is not None:
                    shared[lt] = r
            fn = self._active.get(("resid_post", i))
            if fn is not None:
                r = fn(h, i); h = h if r is None else r
        if keep_resid:
            resid.append(h.detach().clone())
        self._active = {}
        out = {"resid": resid, "final_h": h}
        if return_shared:
            out["shared"] = {k: tuple(t.clone() for t in v) for k, v in shared.items()}
        if logits == "last":
            out["logits"] = self.unembed(h[:, -1])
        elif logits == "all":
            out["logits"] = self.unembed(h)
        return out
