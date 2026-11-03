"""Linear-layer backend microbenchmark with cold weights (real model sizes, DRAM-bound).

Each timing cycles through enough distinct weight matrices (>= 256 MB) that weights stream from
DRAM, as in a real forward pass, instead of sitting in the L3 cache.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn

from llmserve.envinfo import env_info
from llmserve.models.layers import LINEAR_BACKENDS


def timed(fns, x, repeats=3):
    for f in fns:
        f(x)
    t = time.perf_counter()
    for _ in range(repeats):
        for f in fns:
            f(x)
    return (time.perf_counter() - t) / (repeats * len(fns)) * 1e3


def main():
    torch.manual_seed(0)
    shapes = {  # (in, out) of real layers
        "smollm2-135m gate_up": (576, 3072),
        "smollm2-135m down": (1536, 576),
        "qwen3-0.6b qkv": (1024, 4096),
        "qwen3-0.6b gate_up": (1024, 6144),
        "qwen3-0.6b down": (3072, 1024),
    }
    rows = []
    for name, (K, N) in shapes.items():
        copies = max(4, (256 << 20) // (K * N * 4))
        base = [nn.Linear(K, N, bias=False) for _ in range(copies)]
        backends = {
            "F.linear [out,in]": base,
            "x @ W_t [in,out]": [LINEAR_BACKENDS["transposed"](m) for m in base],
            "oneDNN packed": [LINEAR_BACKENDS["onednn"](m) for m in base],
        }
        for M in (1, 4, 16, 64, 256):
            x = torch.randn(M, K)
            r = {"layer": name, "M": M}
            with torch.inference_mode():
                for bname, mods in backends.items():
                    r[bname] = timed(mods, x)
            rows.append(r)
            print(r)
    env = env_info()
    out = Path("results/microbench")
    out.mkdir(parents=True, exist_ok=True)
    (out / "linear_layout.json").write_text(json.dumps({"env": env, "rows": rows}, indent=1))
    names = ["F.linear [out,in]", "x @ W_t [in,out]", "oneDNN packed"]
    lines = [
        "# Linear-layer backends on CPU (cold weights)",
        "",
        f"{env['cpu']}, torch {env['torch']}, {env['torch_threads']} threads, fp32, "
        f"git {env['git_sha']}. Milliseconds per call, averaged while cycling through >= 256 MB",
        "of distinct weights so they stream from DRAM as in a real forward pass.",
        "",
        "| layer | rows (M) | " + " | ".join(names) + " | best vs F.linear |",
        "|---|---:|" + "---:|" * (len(names) + 1),
    ]
    for r in rows:
        best = min(r[n] for n in names)
        cells = " | ".join(f"{r[n]:.3f}" for n in names)
        lines.append(f"| {r['layer']} | {r['M']} | {cells} | {r[names[0]] / best:.2f}x |")
    (out / "linear_layout.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
