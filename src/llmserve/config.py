"""Model and engine configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_ARCHS = {"llama", "qwen3"}


@dataclass
class ModelConfig:
    """The subset of a Hugging Face ``config.json`` the engine needs."""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    qk_norm: bool = False  # Qwen3 normalizes q and k per head before RoPE
    bos_token_id: int | None = None
    eos_token_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelConfig:
        mt = d["model_type"]
        if mt not in SUPPORTED_ARCHS:
            raise ValueError(f"unsupported model_type {mt!r}; supported: {sorted(SUPPORTED_ARCHS)}")
        n_heads = d["num_attention_heads"]
        eos = d.get("eos_token_id")
        eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        # transformers >= 5 nests RoPE settings in "rope_parameters"; older files use
        # top-level "rope_theta" + "rope_scaling".
        rope_params = d.get("rope_parameters") or {}
        theta = rope_params.get("rope_theta", d.get("rope_theta", 10000.0))
        scaling = d.get("rope_scaling")
        if scaling is None and rope_params.get("rope_type", "default") != "default":
            scaling = rope_params
        return cls(
            model_type=mt,
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=d.get("num_key_value_heads", n_heads),
            head_dim=d.get("head_dim") or d["hidden_size"] // n_heads,
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=theta,
            max_position_embeddings=d.get("max_position_embeddings", 4096),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            rope_scaling=scaling,
            attention_bias=d.get("attention_bias", False),
            qk_norm=(mt == "qwen3"),
            bos_token_id=d.get("bos_token_id"),
            eos_token_ids=eos_ids,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> ModelConfig:
        cfg = cls.from_dict(json.loads((Path(path) / "config.json").read_text()))
        # generation_config.json often lists extra stop tokens (e.g. chat end-of-turn).
        gen = Path(path) / "generation_config.json"
        if gen.exists():
            g = json.loads(gen.read_text()).get("eos_token_id")
            extra = [] if g is None else ([g] if isinstance(g, int) else list(g))
            cfg.eos_token_ids = list(dict.fromkeys(cfg.eos_token_ids + extra))
        return cfg

    def kv_bytes_per_token(self, dtype_bytes: int = 4) -> int:
        """Bytes of K+V cache one token occupies across all layers."""
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * dtype_bytes
