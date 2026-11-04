"""Decode-step time with and without the fused C++ elementwise kernels.

B sequences with 300-token prompts decode together; the median of 60 steps (after 10 warm-up
steps) is recorded. ``fused_ops`` is the only difference between the two runs.

    .venv/bin/python benchmarks/microbench_step.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from llmserve import LLM, SamplingParams
from llmserve.envinfo import env_info


def step_ms(model: str, fused: bool, batches=(1, 4, 8)) -> dict:
    eng = LLM(model, num_threads=4, fused_ops=fused).engine
    assert eng.model.fused_ops is fused
    out = {}
    for B in batches:
        sp = SamplingParams(temperature=0, max_tokens=120, ignore_eos=True)
        for i in range(B):
            eng.add_request(list(range(100 + i, 400 + i)), sp)
        for _ in range(10):
            eng.step()
        ts = []
        for _ in range(60):
            t = time.perf_counter()
            eng.step()
            ts.append(time.perf_counter() - t)
        while eng.has_unfinished_requests():
            eng.step()
        out[B] = sorted(ts)[len(ts) // 2] * 1e3
    return out


def main():
    res = {"env": env_info(), "results": {}}
    lines = [
        "# Decode step time: fused C++ elementwise kernels on/off",
        "",
        f"CPU: {res['env']['cpu']}, 4 threads, fp32, 300-token prompts, median of 60 steps (ms).",
        "",
        "| model | batch | plain | fused | change |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("smollm2-135m", "qwen3-0.6b"):
        plain, fused = step_ms(model, False), step_ms(model, True)
        res["results"][model] = {"plain_ms": plain, "fused_ms": fused}
        for B in plain:
            lines.append(
                f"| {model} | {B} | {plain[B]:.1f} | {fused[B]:.1f} |"
                f" {100 * (fused[B] / plain[B] - 1):+.0f}% |"
            )
    d = Path("results/microbench")
    (d / "decode_step.json").write_text(json.dumps(res, indent=1))
    (d / "decode_step.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
