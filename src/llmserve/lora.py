"""Multi-LoRA serving: many adapters over one base model, mixed in the same batch.

Each request may name a LoRA adapter (PEFT format). Base weights are shared; every adapted linear
layer adds ``scale * (x @ A^T) @ B^T`` for the tokens of requests using that adapter (the S-LoRA /
Punica idea: batch requests across adapters instead of one model copy per adapter). Tokens are
grouped by adapter once per step; each layer applies one small matmul pair per adapter present.

Adapters target HF module names (``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``gate_proj``,
``up_proj``, ``down_proj``). Because our model fuses q|k|v and gate|up, an adapter for ``k_proj``
updates the k slice of the fused output.

Prefix caching stays correct because a sequence's block hashes are seeded with its adapter name
(the same prompt under two adapters has different KV).
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

_PEFT_KEY = re.compile(r"layers\.(\d+)\.(self_attn|mlp)\.(\w+_proj)\.lora_([AB])\.weight$")


class _Ctx(threading.local):
    groups: list[tuple[int, torch.Tensor]] = []  # (adapter slot, token indices) for this step


_ctx = _Ctx()


def set_batch_adapters(token_slots: list[int] | None) -> None:
    """Called by the model runner before each forward: slot per flat token (0 = no adapter)."""
    if not token_slots or not any(token_slots):
        _ctx.groups = []
        return
    t = torch.tensor(token_slots)
    _ctx.groups = [(int(s), (t == s).nonzero().flatten()) for s in t.unique().tolist() if s]


@dataclass
class _Adapter:
    A: torch.Tensor  # [r, in]
    B: torch.Tensor  # [out_slice, r]
    scale: float


class LoRALinear(nn.Module):
    """Wraps any linear backend; ``slices`` names the output column ranges (for fused layers)."""

    def __init__(self, base: nn.Module, slices: dict[str, tuple[int, int]]) -> None:
        super().__init__()
        self.base = base
        self.slices = slices
        self.adapters: dict[int, dict[str, _Adapter]] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        for slot, idx in _ctx.groups:
            ad = self.adapters.get(slot)
            if not ad:
                continue
            xs = x.index_select(0, idx)
            for name, a in ad.items():
                s, e = self.slices[name]
                delta = (xs @ a.A.t()) @ a.B.t() * a.scale
                y[idx, s:e] += delta
        return y


@dataclass
class LoRAManager:
    model: nn.Module
    slots: dict[str, int] = field(default_factory=dict)
    max_rank: int = 0

    def _wrap(self) -> None:
        for layer in self.model.layers:
            at = layer.self_attn
            if not isinstance(at.qkv_proj, LoRALinear):
                q, kv = at.q_size, at.kv_size
                at.qkv_proj = LoRALinear(
                    at.qkv_proj,
                    {"q_proj": (0, q), "k_proj": (q, q + kv), "v_proj": (q + kv, q + 2 * kv)},
                )
                at.o_proj = LoRALinear(at.o_proj, {"o_proj": (0, self.model.cfg.hidden_size)})
            mlp = layer.mlp
            if not isinstance(mlp.gate_up_proj, LoRALinear):
                i = mlp.inter
                mlp.gate_up_proj = LoRALinear(
                    mlp.gate_up_proj, {"gate_proj": (0, i), "up_proj": (i, 2 * i)}
                )
                mlp.down_proj = LoRALinear(
                    mlp.down_proj, {"down_proj": (0, self.model.cfg.hidden_size)}
                )

    def load(self, name: str, path: str | Path) -> int:
        """Load a PEFT adapter folder (adapter_config.json + adapter_model.safetensors)."""
        if name in self.slots:
            return self.slots[name]
        path = Path(path)
        cfg = json.loads((path / "adapter_config.json").read_text())
        r = cfg["r"]
        alpha = cfg.get("lora_alpha", r)
        scale = alpha / (r**0.5) if cfg.get("use_rslora") else alpha / r
        weights = load_file(str(path / "adapter_model.safetensors"))
        dtype = next(self.model.parameters()).dtype
        self._wrap()
        slot = len(self.slots) + 1
        found: dict[tuple[int, str, str], dict[str, torch.Tensor]] = {}
        for k, w in weights.items():
            m = _PEFT_KEY.search(k)
            if not m:
                if "lora_" in k:
                    raise ValueError(f"unsupported LoRA target in {k}")
                continue
            li, block, proj, ab = m.groups()
            found.setdefault((int(li), block, proj), {})[ab] = w.to(dtype)
        if not found:
            raise ValueError(f"no LoRA weights recognised in {path}")
        for (li, block, proj), ab in found.items():
            layer = self.model.layers[li]
            if block == "self_attn":
                mod = layer.self_attn.o_proj if proj == "o_proj" else layer.self_attn.qkv_proj
            else:
                mod = layer.mlp.down_proj if proj == "down_proj" else layer.mlp.gate_up_proj
            s, e = mod.slices[proj]
            if ab["B"].shape[0] != e - s or ab["A"].shape[0] != ab["B"].shape[1]:
                raise ValueError(f"shape mismatch for layer {li} {proj}")
            mod.adapters.setdefault(slot, {})[proj] = _Adapter(
                ab["A"].contiguous(), ab["B"].contiguous(), scale
            )
        self.slots[name] = slot
        self.max_rank = max(self.max_rank, r)
        return slot

    def slot(self, name: str | None) -> int:
        if name is None:
            return 0
        if name not in self.slots:
            raise ValueError(f"unknown LoRA adapter {name!r}; loaded: {sorted(self.slots)}")
        return self.slots[name]
