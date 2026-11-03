"""Basic transformer layers written from scratch."""

from __future__ import annotations

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _llama3_inv_freq(inv_freq: torch.Tensor, scaling: dict) -> torch.Tensor:
    """Llama-3.1 style frequency-dependent RoPE scaling."""
    factor = scaling["factor"]
    low = scaling.get("low_freq_factor", 1.0)
    high = scaling.get("high_freq_factor", 4.0)
    old_ctx = scaling["original_max_position_embeddings"]
    low_wavelen = old_ctx / low
    high_wavelen = old_ctx / high
    wavelen = 2 * math.pi / inv_freq
    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low) / (high - low)
    mid = (1 - smooth) * scaled / factor + smooth * scaled
    is_mid = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(is_mid, mid, scaled)


class RotaryEmbedding(nn.Module):
    """Non-interleaved ("rotate half") RoPE, as in HF Llama/Qwen.

    cos/sin tables are built once for ``max_positions`` and indexed by absolute position, so a
    batch can mix tokens from many sequences at different offsets.
    """

    def __init__(
        self, head_dim: int, theta: float, max_positions: int, scaling: dict | None = None
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
        rope_type = (scaling or {}).get("rope_type", (scaling or {}).get("type"))
        if rope_type == "llama3":
            inv_freq = _llama3_inv_freq(inv_freq, scaling)
        elif rope_type == "linear":
            inv_freq = inv_freq / scaling["factor"]
        elif rope_type not in (None, "default"):
            raise NotImplementedError(f"rope scaling {rope_type!r}")
        t = torch.arange(max_positions, dtype=torch.float64)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos().float(), persistent=False)
        self.register_buffer("sin", emb.sin().float(), persistent=False)

    def tables(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin rows for ``positions`` ([T] -> [T, 1, D]); computed once per forward pass."""
        return (
            self.cos[positions].unsqueeze(1).to(dtype),
            self.sin[positions].unsqueeze(1).to(dtype),
        )

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q: [T, Hq, D], k: [T, Hk, D], positions: [T]
        cos, sin = self.tables(positions, q.dtype)
        return apply_rope(q, k, cos, sin)


def apply_rope(q, k, cos, sin):
    return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class TransposedLinear(nn.Module):
    """``y = x @ W_t (+ b)`` with the weight stored as ``[in, out]``.

    A portable fallback: on MKL, a pre-transposed contiguous weight avoids the transposed-GEMM
    path that is slow for 4-64 rows (``results/microbench/linear_layout.md``). On CPU the oneDNN
    backend below is faster still. Created from an ``nn.Linear`` after loading.
    """

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight_t = nn.Parameter(linear.weight.detach().t().contiguous(), requires_grad=False)
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight_t
        return y if self.bias is None else y + self.bias


def swap_linears(module: nn.Module, factory) -> int:
    """Replace every ``nn.Linear`` child (recursively) with ``factory(linear)``."""
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, factory(child))
            n += 1
        else:
            n += swap_linears(child, factory)
    return n


class OneDNNLinear(nn.Module):
    """Linear layer on oneDNN with the weight pre-packed into the kernel's blocked layout.

    This is what ``torch.compile`` (inductor, freezing) does for fp32 CPU linears. Packing once at
    load time removes the per-call weight reorder and gives the best decode-regime (1-64 rows)
    throughput on the benchmark CPU (``results/microbench/linear_layout.md``).
    """

    def __init__(self, linear: nn.Linear, batch_hint: int = 64) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w = linear.weight.detach().contiguous()
        self.weight_packed = nn.Parameter(
            torch.ops.mkldnn._reorder_linear_weight(w, batch_hint), requires_grad=False
        )
        self.bias = (
            None
            if linear.bias is None
            else nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.mkldnn._linear_pointwise(x, self.weight_packed, self.bias, "none", [], "")


def onednn_available() -> bool:
    return torch.backends.mkldnn.is_available() and hasattr(torch.ops.mkldnn, "_linear_pointwise")


LINEAR_BACKENDS = {"torch": None, "transposed": TransposedLinear, "onednn": OneDNNLinear}
