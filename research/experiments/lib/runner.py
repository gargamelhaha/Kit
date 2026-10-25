"""Hookable forward pass for the Gemma 4 E2B text decoder (equal-length batches, no padding).

Hook points, fn(tensor, layer) -> tensor | None (None keeps the value):
  resid_pre   residual stream entering layer L            [b, t, d]
  attn_out    attention contribution (post-attn norm)      [b, t, d]
  mlp_out     MLP contribution (post-ffw norm)             [b, t, d]
  ple_out     per-layer-embedding contribution             [b, t, d]
  head_in     o_proj input split per head                  [b, t, h, dh]
  resid_post  residual leaving layer L (after layer_scalar)
  attn_probs  read-only attention probabilities [b, h, q, k] (requires attn="eager")
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

    def unembed(self, h):
        x, W = self.tm.norm(h).float(), self.model.lm_head.weight
        logits = torch.cat([torch.nn.functional.linear(x, W[r:r + 32768].float()) for r in range(0, W.shape[0], 32768)], dim=-1)
        return torch.tanh(logits / self.cap) * self.cap if self.cap else logits

    @torch.no_grad()
    def forward(self, ids, hooks=None, last_only_from=None, keep_resid=False, attn="sdpa", logits="last"):
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
                if ids.shape[1] > self.cfg.sliding_window:
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
            fn = self._active.get(("resid_post", i))
            if fn is not None:
                r = fn(h, i); h = h if r is None else r
        if keep_resid:
            resid.append(h.detach().clone())
        self._active = {}
        out = {"resid": resid, "final_h": h}
        if logits == "last":
            out["logits"] = self.unembed(h[:, -1])
        elif logits == "all":
            out["logits"] = self.unembed(h)
        return out
