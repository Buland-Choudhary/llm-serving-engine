"""Decoder-only transformer (Llama / Qwen3 families), written from scratch.

The forward pass works on a flat batch of new tokens from many sequences (see
``llmserve.attention``). Projections are fused (q|k|v and gate|up) so each layer runs 4 matmuls
instead of 7.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llmserve.attention import AttentionMetadata, paged_attention
from llmserve.config import ModelConfig
from llmserve.models.layers import RMSNorm, RotaryEmbedding, apply_rope


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv * self.head_dim
        self.qkv_proj = nn.Linear(
            cfg.hidden_size, self.q_size + 2 * self.kv_size, bias=cfg.attention_bias
        )
        self.o_proj = nn.Linear(self.q_size, cfg.hidden_size, bias=cfg.attention_bias)
        self.qk_norm = cfg.qk_norm
        if cfg.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.scale = self.head_dim**-0.5

    def qkv(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        T = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(T, self.n_heads, self.head_dim)
        k = k.view(T, self.n_kv, self.head_dim)
        v = v.view(T, self.n_kv, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        rope_cs: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        q, k, v = self.qkv(x, *rope_cs)
        o = paged_attention(q, k, v, k_cache, v_cache, meta, self.scale)
        return self.o_proj(o.reshape(x.shape[0], self.q_size))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.inter = cfg.intermediate_size
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).split(self.inter, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = MLP(cfg)

    # The layer is split at attention: the two halves are pure tensor code that
    # ``torch.compile`` can fuse; attention (paged, data-dependent) runs in between.
    def pre_attn(self, x, cos, sin):
        return self.self_attn.qkv(self.input_layernorm(x), cos, sin)

    def post_attn(self, x, o):
        x = x + self.self_attn.o_proj(o)
        return x + self.mlp(self.post_attention_layernorm(x))

    def forward(self, x, rope_cs, k_cache, v_cache, meta):
        q, k, v = self._pre(x, *rope_cs)
        o = paged_attention(q, k, v, k_cache, v_cache, meta, self.self_attn.scale)
        return self._post(x, o.reshape(x.shape[0], self.self_attn.q_size))

    _pre = pre_attn
    _post = post_attn

    def compile(self, **kw) -> None:
        import torch

        self._pre = torch.compile(self.pre_attn, dynamic=True, **kw)
        self._post = torch.compile(self.post_attn, dynamic=True, **kw)


class CausalLM(nn.Module):
    """Embeddings -> N decoder layers -> final norm. ``compute_logits`` applies the LM head."""

    def __init__(self, cfg: ModelConfig, max_positions: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.rope = RotaryEmbedding(
            cfg.head_dim,
            cfg.rope_theta,
            max_positions or min(cfg.max_position_embeddings, 32768),
            cfg.rope_scaling,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        """input_ids, positions: [T]. Returns final hidden states [T, hidden]."""
        if self.fused_ops:
            return self._forward_fused(input_ids, positions, kv_caches, meta)
        x = self.embed_tokens(input_ids)
        rope_cs = self.rope.tables(positions, x.dtype)
        for layer, (kc, vc) in zip(self.layers, kv_caches, strict=True):
            x = layer(x, rope_cs, kc, vc, meta)
        return self.norm(x)

    # Set by the engine (EngineConfig.fused_ops) when the C++ fused kernels are available.
    fused_ops = False

    def _forward_fused(self, input_ids, positions, kv_caches, meta) -> torch.Tensor:
        """Same math as ``forward`` with the elementwise ops fused (``kernels/csrc/fused_ops.cpp``).

        The residual stream ``x`` is updated in place and each residual add is fused into the
        following RMSNorm (the next layer's input norm, or the final norm).
        """
        from llmserve.kernels import load_fused_ops

        fo = load_fused_ops()
        eps = self.cfg.rms_norm_eps
        x = self.embed_tokens(input_ids).contiguous()
        T = x.shape[0]
        cos, sin = self.rope.tables(positions, x.dtype)
        cos, sin = cos.reshape(T, -1).contiguous(), sin.reshape(T, -1).contiguous()
        h = fo.rmsnorm(x, self.layers[0].input_layernorm.weight, eps)
        n = len(self.layers)
        for i, (layer, (kc, vc)) in enumerate(zip(self.layers, kv_caches, strict=True)):
            a = layer.self_attn
            qkv = a.qkv_proj(h).contiguous()
            fo.rope_qk_(
                qkv,
                cos,
                sin,
                a.n_heads,
                a.n_kv,
                a.head_dim,
                a.q_norm.weight if a.qk_norm else None,
                a.k_norm.weight if a.qk_norm else None,
                eps,
            )
            q, k, v = qkv.split([a.q_size, a.kv_size, a.kv_size], dim=-1)
            o = paged_attention(
                q.view(T, a.n_heads, a.head_dim),
                k.view(T, a.n_kv, a.head_dim),
                v.view(T, a.n_kv, a.head_dim),
                kc,
                vc,
                meta,
                a.scale,
            )
            attn = a.o_proj(o.reshape(T, a.q_size)).contiguous()
            h = fo.add_rmsnorm_(x, attn, layer.post_attention_layernorm.weight, eps)
            mlp = layer.mlp
            down = mlp.down_proj(fo.silu_mul(mlp.gate_up_proj(h).contiguous())).contiguous()
            nxt = self.layers[i + 1].input_layernorm if i + 1 < n else self.norm
            h = fo.add_rmsnorm_(x, down, nxt.weight, eps)
        return h

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()
