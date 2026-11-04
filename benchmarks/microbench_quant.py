"""Quantized linear kernels vs fp32, by number of rows (cold weights, real Qwen3-0.6B shape)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn

from llmserve.envinfo import env_info
from llmserve.models.layers import OneDNNLinear
from llmserve.quant import Int4Linear, Int8Linear


class _Fused8(Int8Linear):
    max_fused_rows = 1 << 30  # always the fused kernel


class _W8A8(nn.Module):
    """Dynamic per-token int8 activations x per-channel int8 weights via torch._int_mm."""

    def __init__(self, lin: nn.Linear) -> None:
        super().__init__()
        w = lin.weight.detach()
        self.ws = (w.abs().amax(1) / 127).clamp_min(1e-8)
        self.wq_t = (
            torch.round(w / self.ws[:, None]).clamp(-127, 127).to(torch.int8).t().contiguous()
        )

    def forward(self, x):
        xs = (x.abs().amax(1, keepdim=True) / 127).clamp_min(1e-8)
        xq = torch.round(x / xs).clamp(-127, 127).to(torch.int8)
        return torch._int_mm(xq, self.wq_t).float() * xs * self.ws


def timed(mods, x, repeats=3):
    for m in mods:
        m(x)
    t = time.perf_counter()
    for _ in range(repeats):
        for m in mods:
            m(x)
    return (time.perf_counter() - t) / (repeats * len(mods)) * 1e3


def main():
    torch.manual_seed(0)
    K, N = 1024, 6144  # Qwen3-0.6B gate_up
    copies = 8  # 8 x 25 MB fp32 > L3: weights stream from DRAM
    base = [nn.Linear(K, N, bias=False) for _ in range(copies)]
    impls = {
        "fp32 oneDNN": [OneDNNLinear(b) for b in base],
        "int8 fused kernel": [_Fused8(b) for b in base],
        "int8 (dispatch)": [Int8Linear(b) for b in base],
        "int4 fused kernel": [Int4Linear(b, 64) for b in base],
        "W8A8 _int_mm": [_W8A8(b) for b in base],
    }
    rows = []
    with torch.inference_mode():
        for M in (1, 4, 16, 64, 256, 1024):
            x = torch.randn(M, K)
            r = {"M": M}
            for name, mods in impls.items():
                try:
                    r[name] = timed(mods, x)
                except RuntimeError as e:  # e.g. _int_mm shape limits
                    r[name] = None
                    r[name + " error"] = str(e)[:80]
            rows.append(r)
            print(r)
    env = env_info()
    out = Path("results/m8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernels.json").write_text(
        json.dumps({"env": env, "shape": [K, N], "rows": rows}, indent=1)
    )
    names = list(impls)
    lines = [
        "# Quantized linear kernels by number of rows (CPU)",
        "",
        f"{env['cpu']}, torch {env['torch']}, {env['torch_threads']} threads, "
        f"git {env['git_sha']}.",
        f"Layer shape {K} -> {N} (Qwen3-0.6B gate/up), ms per call, weights cycled from DRAM.",
        "",
        "| rows | " + " | ".join(names) + " |",
        "|---:|" + "---:|" * len(names),
    ]
    for r in rows:
        cells = " | ".join("n/a" if r[n] is None else f"{r[n]:.2f}" for n in names)
        lines.append(f"| {r['M']} | {cells} |")
    (out / "kernels.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
