"""Weight-only quantization: int8 per-channel and int4 group-wise (round-to-nearest).

Decode on CPU is memory-bandwidth bound: every step streams all weights from DRAM. Storing them
in 8 or 4 bits cuts that traffic 4x / ~7x versus fp32. Activations stay fp32; the matmuls use
PyTorch's fused CPU kernels that dequantize on the fly (``_weight_int8pack_mm``,
``_weight_int4pack_mm_for_cpu``).

Optional AWQ-style activation-aware scaling (``awq_scale``) protects the input channels with the
largest activations before int4 rounding (Lin et al. 2023, simplified: one grid search of the
scale exponent per layer on calibration activations).
"""

from __future__ import annotations

import torch
from torch import nn

from llmserve.kernels import load_int8_gemv
from llmserve.models.layers import swap_linears


class Int8Linear(nn.Module):
    """Int8 weights (one fp32 scale per output channel), two execution paths:

    * <= ``gemv_max_rows`` rows (decode): exact weight-only math with our C++ GEMV, which streams
      each int8 weight row once (~3x faster than fp32 for one row on the benchmark CPU).
    * more rows (batched decode, prefill): activations are quantized per token to int8 too
      (W8A8) and multiplied with an int8 x int8 -> int32 GEMM (``torch._int_mm``, VNNI/AMX).
      This is the fastest path from ~16 rows up and adds a small activation-rounding error.

    Measured in ``results/m8/kernels.md``. PyTorch's ``_weight_int8pack_mm`` was slower than fp32
    at every size there and is only kept as a fallback when the C++ kernel cannot be built.
    """

    gemv_max_rows = 4
    # "auto" (GEMV / W8A8 dispatch), or force one path: "gemv", "w8a8", "aten", "dequant"
    kernel = "auto"

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        w = linear.weight.detach().float()
        scale = (w.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
        q = torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8)
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.contiguous())
        bias = linear.bias
        self.bias = None if bias is None else nn.Parameter(bias.detach().clone(), False)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self._qweight_t = None  # [in, out] int8 for _int_mm, built lazily

    def _w8a8(self, x2: torch.Tensor) -> torch.Tensor:
        if self._qweight_t is None:
            self._qweight_t = self.qweight.t().contiguous()
        xs = (x2.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
        xq = torch.round(x2 / xs).clamp(-127, 127).to(torch.int8)
        return torch._int_mm(xq, self._qweight_t).float() * xs * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).contiguous()
        path = self.kernel
        if path == "auto":
            path = "gemv" if x2.shape[0] <= self.gemv_max_rows else "w8a8"
        k = load_int8_gemv() if path == "gemv" else None
        if path == "gemv" and (k is None or x2.shape[0] > 64):
            path = "aten"
        if path == "gemv":
            y = k.int8_linear(x2, self.qweight, self.scale)
        elif path == "w8a8":
            y = self._w8a8(x2)
        elif path == "dequant":
            y = x2 @ self.dequantize().t()
        else:
            y = torch.ops.aten._weight_int8pack_mm(x2, self.qweight, self.scale)
        y = y.reshape(*shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias

    def dequantize(self) -> torch.Tensor:
        return self.qweight.float() * self.scale[:, None]


def _int4_quantize(w: torch.Tensor, group_size: int):
    """Asymmetric 4-bit, one (scale, zero) per group of ``group_size`` input channels."""
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // group_size, group_size)
    mn, mx = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    scale = ((mx - mn) / 15.0).clamp_min(1e-8)
    q = torch.round((g - mn) / scale).clamp(0, 15).to(torch.int32).reshape(out_f, in_f)
    zero = mn + 8.0 * scale  # kernel convention: w = (q - 8) * scale + zero
    return q, scale.squeeze(-1), zero.squeeze(-1)


class Int4Linear(nn.Module):
    def __init__(
        self, linear: nn.Linear, group_size: int = 64, act_scale: torch.Tensor | None = None
    ) -> None:
        super().__init__()
        w = linear.weight.detach().float()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if self.in_features % group_size:
            raise ValueError(f"in_features {self.in_features} not divisible by group {group_size}")
        # AWQ-style: quantize W * s (per input channel) and divide activations by s at runtime.
        if act_scale is not None:
            self.register_buffer("inv_act_scale", (1.0 / act_scale).contiguous())
            w = w * act_scale[None, :]
        else:
            self.inv_act_scale = None
        q, scale, zero = _int4_quantize(w, group_size)
        self.group_size = group_size
        self.register_buffer("packed", torch.ops.aten._convert_weight_to_int4pack_for_cpu(q, 1))
        # Our own nibble packing (2 weights per byte) for the dequantize + GEMM path used on
        # prefill-sized inputs, where the fused kernel is slow (one GEMV per row).
        q8 = q.to(torch.uint8)
        self.register_buffer("nibbles", (q8[:, 0::2] | (q8[:, 1::2] << 4)).contiguous())
        self.register_buffer("g_scale", scale.contiguous())  # [out, n_groups]
        self.register_buffer("g_zero", zero.contiguous())
        # [n_groups, out, 2] as the kernel expects
        self.register_buffer("scales_zeros", torch.stack([scale.t(), zero.t()], -1).contiguous())
        self.bias = (
            None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), False)
        )

    max_fused_rows = 16  # above this: dequantize + GEMM
    kernel = "auto"  # "auto" (our C++ GEMV for small batches) or "aten" (fused PyTorch kernel)

    def dequantize(self) -> torch.Tensor:
        """fp32 weight [out, in] of the (scaled) quantized matrix."""
        lo = (self.nibbles & 0xF).float()
        hi = (self.nibbles >> 4).float()
        q = torch.stack([lo, hi], dim=-1).reshape(self.out_features, -1, self.group_size)
        w = (q - 8) * self.g_scale[..., None] + self.g_zero[..., None]
        return w.reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if self.inv_act_scale is not None:
            x2 = x2 * self.inv_act_scale
        k = load_int8_gemv() if self.kernel == "auto" else None
        if x2.shape[0] > self.max_fused_rows:
            y = x2 @ self.dequantize().t()
        elif k is not None:
            y = k.int4_linear(
                x2.contiguous(), self.nibbles, self.g_scale, self.g_zero, self.group_size
            )
        else:
            y = torch.ops.aten._weight_int4pack_mm_for_cpu(
                x2.contiguous(), self.packed, self.group_size, self.scales_zeros
            )
        y = y.reshape(*shape[:-1], self.out_features)
        return y if self.bias is None else y + self.bias


def _search_awq_scale(w: torch.Tensor, x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pick s = mean|x|^alpha (alpha on a grid) minimizing ||x W^T - (x/s) Q(W s)^T||."""
    act = x.abs().mean(0).clamp_min(1e-5)
    ref = x @ w.t()
    best, best_s = float("inf"), None
    for alpha in [i / 10 for i in range(11)]:
        s = act.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()  # keep scales centered around 1
        q, scale, zero = _int4_quantize(w * s[None, :], group_size)
        deq = (q.float().reshape(w.shape[0], -1, group_size) - 8) * scale[..., None] + zero[
            ..., None
        ]
        deq = deq.reshape(w.shape) / s[None, :]
        err = (x @ deq.t() - ref).pow(2).mean().item()
        if err < best:
            best, best_s = err, s
    return best_s


def quantize_model(
    model: nn.Module,
    method: str,
    group_size: int = 64,
    quantize_lm_head: bool = False,
    calib_acts: dict[str, torch.Tensor] | None = None,
) -> nn.Module:
    """Swap linear layers for quantized ones in place. ``method``: int8 | int4 | int4-awq.

    Works on a model loaded with ``linear_backend="torch"`` (plain ``nn.Linear`` layers).
    """
    if method not in ("int8", "int4", "int4-awq"):
        raise ValueError(f"unknown quantization {method!r}")
    if method == "int4-awq" and not calib_acts:
        raise ValueError("int4-awq needs calibration activations (see collect_linear_inputs)")
    lm_head = model.lm_head if hasattr(model, "lm_head") else None
    names = {id(m): n for n, m in model.named_modules()}

    def factory(lin: nn.Linear) -> nn.Module:
        if lin is lm_head and not quantize_lm_head:
            return lin
        if method == "int8":
            return Int8Linear(lin)
        s = None
        if method == "int4-awq":
            x = calib_acts.get(names[id(lin)])
            if x is not None:
                s = _search_awq_scale(lin.weight.detach().float(), x, group_size)
        return Int4Linear(lin, group_size, s)

    swap_linears(model, factory)
    model.quantization = method
    return model


@torch.inference_mode()
def collect_linear_inputs(
    model: nn.Module, run_fn, max_rows: int = 2048
) -> dict[str, torch.Tensor]:
    """Record (a sample of) the inputs every ``nn.Linear`` sees while ``run_fn()`` runs."""
    acts: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):

            def hook(m, inp, out, name=name):
                acts.setdefault(name, []).append(inp[0].reshape(-1, inp[0].shape[-1]).detach())

            hooks.append(mod.register_forward_hook(hook))
    try:
        run_fn()
    finally:
        for h in hooks:
            h.remove()
    out = {}
    for name, xs in acts.items():
        x = torch.cat(xs)
        if x.shape[0] > max_rows:
            x = x[torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(0))[:max_rows]]
        out[name] = x
    return out


def weight_bytes(model: nn.Module) -> int:
    """Bytes of parameters + buffers (tied tensors counted once), excluding RoPE tables."""
    seen, total = set(), 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        try:
            key = t.data_ptr()
        except RuntimeError:  # opaque tensors (oneDNN prepacked weights) have no storage
            key = id(t)
        if "rope" in name or key in seen:
            continue
        seen.add(key)
        total += t.numel() * t.element_size()
    return total
