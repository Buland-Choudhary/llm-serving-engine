"""Greedy agreement between the fp32 and the int8 KV cache on real checkpoints.

Same ShareGPT prompts, greedy, fixed output length. Reports the share of requests whose whole
output is identical and the mean length of the identical prefix (after the first divergence the
contexts differ, so later token-by-token comparisons are not meaningful).

    .venv/bin/python benchmarks/kv_int8_agreement.py --model qwen3-0.6b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmserve import LLM, SamplingParams
from llmserve.bench import workloads
from llmserve.envinfo import env_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--num-requests", type=int, default=50)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--kv-k-smooth", type=float, default=None, help="int8 key smoothing alpha")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    outs = {}
    for dt in ("auto", "int8"):
        llm = LLM(
            args.model, kv_cache_dtype=dt, kv_cache_memory_gb=1.0, kv_k_smooth=args.kv_k_smooth
        )
        reqs = workloads.sharegpt(llm.engine.tokenizer, args.num_requests, 0, 512, 16)
        sp = SamplingParams(temperature=0, max_tokens=args.output_len, ignore_eos=True)
        res = llm.generate([r.prompt_token_ids for r in reqs], sp)
        outs[dt] = [r.outputs[0].token_ids for r in res]
        del llm
    prefix = []
    for a, b in zip(outs["auto"], outs["int8"], strict=True):
        n = 0
        while n < len(a) and a[n] == b[n]:
            n += 1
        prefix.append(n)
    L = args.output_len
    res = {
        "model": args.model,
        "kv_k_smooth": args.kv_k_smooth,
        "num_requests": args.num_requests,
        "output_len": L,
        "identical_outputs": sum(p == L for p in prefix) / len(prefix),
        "mean_identical_prefix": sum(prefix) / len(prefix),
        "first_token_agreement": sum(p > 0 for p in prefix) / len(prefix),
        "identical_prefix_lens": prefix,
        "env": env_info(),
    }
    print(json.dumps({k: v for k, v in res.items() if k not in ("env", "identical_prefix_lens")}))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
