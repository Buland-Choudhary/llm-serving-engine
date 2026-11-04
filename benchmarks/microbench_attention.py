"""Paged-attention decode microbenchmark: v1 kernel vs v2 (fp32 and int8 KV).

One timed call = one decode step through every layer (a separate KV tensor per layer, so the
cache streams from memory as in the engine instead of staying in L2). Reports time per layer
call and the K+V bytes read per second.

    .venv/bin/python benchmarks/microbench_attention.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from llmserve.envinfo import env_info
from llmserve.kernels import load_paged_attention

SHAPES = {  # name: (q heads, kv heads, head dim, layers)
    "smollm2-135m": (9, 3, 64, 30),
    "qwen3-0.6b": (16, 8, 128, 28),
}
BS = 16


def quant(x):
    s = x.abs().amax(-1).clamp_min(1e-8) / 127
    return (x / s[..., None]).round().to(torch.int8), s.contiguous()


def run_case(m, shape, batch, ctx, repeats=5):
    hq, hk, d, layers = shape
    nblk = -(-ctx // BS)
    slots = batch * nblk * BS
    g = torch.Generator().manual_seed(0)
    # Enough per-layer copies to stream from DRAM, capped at ~512 MB of fp32 K+V.
    per_layer = slots * hk * d * 4 * 2
    layers = max(2, min(layers, (512 << 20) // per_layer))
    caches = []
    for _ in range(layers):
        k, v = torch.randn(slots, hk, d, generator=g), torch.randn(slots, hk, d, generator=g)
        kq, ks = quant(k)
        vq, vs = quant(v)
        caches.append((k, v, kq, vq, ks, vs))
    bt = torch.randperm(batch * nblk, generator=g).view(batch, nblk).contiguous()
    cl = torch.full((batch,), ctx, dtype=torch.int64)
    q = torch.randn(batch, hq, d, generator=g)
    sc = d**-0.5
    impls = {
        "v1 fp32": lambda c: m.paged_attention_decode_v1(q, c[0], c[1], bt, cl, BS, sc),
        "v2 fp32": lambda c: m.paged_attention_decode(q, c[0], c[1], bt, cl, BS, sc),
        "v2 int8": lambda c: m.paged_attention_decode_int8(
            q, c[2], c[3], c[4], c[5], bt, cl, BS, sc
        ),
    }
    fp32_bytes = batch * ctx * hk * d * 4 * 2
    int8_bytes = batch * ctx * hk * (d + 4) * 2
    row = {"batch": batch, "ctx": ctx}
    for name, fn in impls.items():
        for c in caches[:2]:
            fn(c)  # warm-up
        best = float("inf")
        for _ in range(repeats):
            t = time.perf_counter()
            for c in caches:
                fn(c)
            best = min(best, (time.perf_counter() - t) / layers)
        nbytes = int8_bytes if "int8" in name else fp32_bytes
        row[name] = {"us_per_layer": best * 1e6, "GB_per_s": nbytes / best / 1e9}
    return row


def main():
    torch.set_num_threads(4)
    m = load_paged_attention()
    out = {"env": env_info(), "block_size": BS, "simd": m.simd_available(), "results": {}}
    for name, shape in SHAPES.items():
        rows = []
        for batch in (1, 16):
            for ctx in (256, 1024, 4096):
                r = run_case(m, shape, batch, ctx)
                rows.append(r)
                print(
                    name,
                    batch,
                    ctx,
                    {k: round(v["us_per_layer"]) for k, v in r.items() if isinstance(v, dict)},
                )
        out["results"][name] = rows
    d = Path("results/microbench")
    d.mkdir(parents=True, exist_ok=True)
    (d / "attention.json").write_text(json.dumps(out, indent=1))
    lines = [
        "# Paged-attention decode kernel: v1 vs v2",
        "",
        f"CPU: {out['env'].get('cpu', '?')}, 4 threads, block size {BS},"
        f" AVX-512 path: {out['simd']}."
        " Time per layer call (us) and K+V bytes read per second.",
        "",
    ]
    for name, rows in out["results"].items():
        lines += [
            f"## {name}",
            "",
            "| batch | ctx | v1 fp32 | v2 fp32 | v2 int8 KV | v2/v1 speedup | v1 GB/s | v2 GB/s |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            a, b, c = r["v1 fp32"], r["v2 fp32"], r["v2 int8"]
            lines.append(
                f"| {r['batch']} | {r['ctx']} | {a['us_per_layer']:.0f} | {b['us_per_layer']:.0f}"
                f" | {c['us_per_layer']:.0f} | {a['us_per_layer'] / b['us_per_layer']:.2f}x"
                f" | {a['GB_per_s']:.1f} | {b['GB_per_s']:.1f} |"
            )
        lines.append("")
    (d / "attention.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
