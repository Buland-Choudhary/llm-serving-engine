"""Load Hugging Face safetensors checkpoints into the from-scratch model."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors import safe_open

from llmserve.config import ModelConfig
from llmserve.models.hub import resolve_model_path
from llmserve.models.layers import LINEAR_BACKENDS, onednn_available, swap_linears
from llmserve.models.transformer import CausalLM

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# HF parameter -> (our fused parameter, shard index). Shards are concatenated along dim 0.
_FUSED = {
    "self_attn.q_proj": ("self_attn.qkv_proj", 0),
    "self_attn.k_proj": ("self_attn.qkv_proj", 1),
    "self_attn.v_proj": ("self_attn.qkv_proj", 2),
    "mlp.gate_proj": ("mlp.gate_up_proj", 0),
    "mlp.up_proj": ("mlp.gate_up_proj", 1),
}


def _iter_checkpoint(path: Path):
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors files in {path}")
    for f in files:
        with safe_open(str(f), framework="pt") as fh:
            for name in fh.keys():  # noqa: SIM118 (safe_open has no __iter__)
                yield name, fh.get_tensor(name)


def load_state_dict_into(model: CausalLM, tensors, dtype: torch.dtype) -> None:
    """Copy HF-named tensors into ``model``, fusing q/k/v and gate/up projections."""
    params = dict(model.named_parameters())
    pending: dict[str, dict[int, torch.Tensor]] = {}
    loaded: set[str] = set()

    def put(name: str, t: torch.Tensor) -> None:
        p = params[name]
        if p.shape != t.shape:
            raise ValueError(f"{name}: checkpoint {tuple(t.shape)} vs model {tuple(p.shape)}")
        with torch.no_grad():
            p.copy_(t.to(dtype))
        loaded.add(name)

    for name, t in tensors:
        if name == "lm_head.weight" and model.cfg.tie_word_embeddings:
            continue
        if name.startswith("model."):
            m = _LAYER.match(name)
            if m:
                idx, rest = m.groups()
                mod, _, kind = rest.rpartition(".")  # kind: weight / bias
                if mod in _FUSED:
                    fused, shard = _FUSED[mod]
                    key = f"layers.{idx}.{fused}.{kind}"
                    pending.setdefault(key, {})[shard] = t
                    n_shards = 3 if "qkv" in fused else 2
                    if len(pending[key]) == n_shards:
                        put(key, torch.cat([pending[key][i] for i in range(n_shards)], dim=0))
                        del pending[key]
                    continue
                put(f"layers.{idx}.{rest}", t)
            else:
                put(name.removeprefix("model."), t)
        elif name == "lm_head.weight":
            put(name, t)
        elif "rotary_emb" in name:
            continue
        else:
            raise KeyError(f"unexpected checkpoint tensor {name}")

    if pending:
        raise ValueError(f"incomplete fused weights: {sorted(pending)}")
    expected = set(params)
    if model.cfg.tie_word_embeddings:
        expected.discard("lm_head.weight")
    missing = expected - loaded
    if missing:
        raise ValueError(f"weights missing from checkpoint: {sorted(missing)[:10]}")


def load_model(
    name_or_path: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    max_positions: int | None = None,
    linear_backend: str = "auto",
    quantization: str | None = None,
    quantize_lm_head: bool = False,
    group_size: int = 64,
    calib_token_ids: list[list[int]] | None = None,
) -> CausalLM:
    """Load a checkpoint. ``quantization`` (int8 | int4 | int4-awq) replaces the linear layers
    with weight-only quantized ones; int4-awq calibrates on ``calib_token_ids`` (default: a
    slice of WikiText-2 *train*)."""
    path = resolve_model_path(str(name_or_path))
    cfg = ModelConfig.from_pretrained(path)
    with torch.device("meta"):
        model = CausalLM(cfg, max_positions=max_positions)
    model = model.to_empty(device=device)
    # Buffers (RoPE tables) are not in the checkpoint: rebuild them on the real device.
    model.rope.__init__(
        cfg.head_dim,
        cfg.rope_theta,
        max_positions or min(cfg.max_position_embeddings, 32768),
        cfg.rope_scaling,
    )
    model.rope.to(device)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    load_state_dict_into(model, _iter_checkpoint(path), dtype)
    model.to(dtype)
    model.rope.cos = model.rope.cos.float()
    model.rope.sin = model.rope.sin.float()
    model.eval()
    model.requires_grad_(False)
    if quantization:
        from llmserve.quant import collect_linear_inputs, quantize_model

        acts = None
        if quantization == "int4-awq":
            from llmserve.eval.perplexity import forward_logits

            if calib_token_ids is None:
                calib_token_ids = _default_calibration(path)
            acts = collect_linear_inputs(
                model, lambda: [forward_logits(model, c) for c in calib_token_ids]
            )
        quantize_model(model, quantization, group_size, quantize_lm_head, acts)
        linear_backend = "torch"  # remaining fp32 linears (e.g. lm_head) stay as nn.Linear
    if linear_backend == "auto":
        on_cpu = torch.device(device).type == "cpu"
        linear_backend = (
            "onednn" if on_cpu and dtype == torch.float32 and onednn_available() else "torch"
        )
    if linear_backend not in LINEAR_BACKENDS:
        raise ValueError(
            f"unknown linear backend {linear_backend!r}; use {sorted(LINEAR_BACKENDS)}"
        )
    factory = LINEAR_BACKENDS[linear_backend]
    if factory is not None:
        swap_linears(model, factory)
    model.linear_backend = linear_backend
    return model


def _default_calibration(path: Path, n_seqs: int = 8, seq_len: int = 512) -> list[list[int]]:
    from transformers import AutoTokenizer

    from llmserve.eval.perplexity import wikitext_text

    tok = AutoTokenizer.from_pretrained(str(path))
    ids = tok(wikitext_text("train")[:400_000], add_special_tokens=False).input_ids
    return [ids[i * seq_len : (i + 1) * seq_len] for i in range(n_seqs)]
