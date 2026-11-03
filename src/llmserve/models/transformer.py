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

    def forward(
        self,
        x: torch.Tensor,
        rope_cs: tuple[torch.Tensor, torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        T = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(T, self.n_heads, self.head_dim)
        k = k.view(T, self.n_kv, self.head_dim)
        v = v.view(T, self.n_kv, self.head_dim)
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, *rope_cs)
        o = paged_attention(q, k, v, k_cache, v_cache, meta, self.scale)
        return self.o_proj(o.reshape(T, self.q_size))


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

    def forward(self, x, rope_cs, k_cache, v_cache, meta):
        x = x + self.self_attn(self.input_layernorm(x), rope_cs, k_cache, v_cache, meta)
        return x + self.mlp(self.post_attention_layernorm(x))


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
        x = self.embed_tokens(input_ids)
        rope_cs = self.rope.tables(positions, x.dtype)
        for layer, (kc, vc) in zip(self.layers, kv_caches, strict=True):
            x = layer(x, rope_cs, kc, vc, meta)
        return self.norm(x)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden).float()
