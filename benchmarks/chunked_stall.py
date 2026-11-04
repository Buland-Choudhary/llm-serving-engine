"""How long do running streams freeze when a long prompt arrives?

8 requests decode steadily; after they are running, one 2048-token prompt arrives. Without
chunked prefill the next step processes the whole prompt and every stream stalls for its full
prefill time. With chunked prefill the prompt is split into budget-sized chunks interleaved with
decode steps, so the worst inter-token gap is bounded by one chunk.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from llmserve.engine import EngineConfig, LLMEngine
from llmserve.envinfo import env_info
from llmserve.sampling import SamplingParams


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--long-len", type=int, default=2048)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--out", default="results/m5/chunked_stall.json")
    args = ap.parse_args()
    torch.set_num_threads(4)
    res = {"env": env_info(), "args": vars(args), "runs": {}}
    for chunked in (False, True):
        res["runs"]["chunked" if chunked else "no_chunk"] = measure(
            args.model, chunked, args.budget, args.long_len
        )
        print(res["runs"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


def measure(model: str, chunked: bool, budget: int, long_len: int) -> dict:
    eng = LLMEngine(
        EngineConfig(
            model=model,
            enable_chunked_prefill=chunked,
            max_num_batched_tokens=budget,
            enable_prefix_caching=False,
            max_model_len=4096,
        )
    )
    rng = random.Random(0)
    v = eng.model_cfg.vocab_size
    sp = SamplingParams(temperature=0, max_tokens=150, ignore_eos=True)
    streams = {
        eng.add_request([rng.randrange(100, v - 100) for _ in range(32)], sp) for _ in range(8)
    }
    for _ in range(20):
        eng.step()
    t_arrive = time.monotonic()
    long_id = eng.add_request(
        [rng.randrange(100, v - 100) for _ in range(long_len)],
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
    )
    final = {}
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                final[o.request_id] = o
    long = final[long_id].metrics
    gaps = []
    for rid in streams:
        tt = final[rid].metrics.token_times
        gaps += [b - a for a, b in zip(tt, tt[1:], strict=False) if b > t_arrive]
    gaps.sort()
    return {
        "long_prompt_ttft_s": long.first_token_time - t_arrive,
        "stream_max_gap_s": gaps[-1],
        "stream_p99_gap_s": gaps[int(0.99 * (len(gaps) - 1))],
        "stream_p50_gap_s": gaps[len(gaps) // 2],
    }


if __name__ == "__main__":
    main()
