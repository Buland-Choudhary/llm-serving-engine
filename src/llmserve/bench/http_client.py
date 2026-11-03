"""HTTP load generator for any OpenAI-compatible server (this one, vLLM, llama.cpp, ...).

Sends ShareGPT or synthetic requests with Poisson arrivals to ``/v1/completions`` with streaming
and measures client-side TTFT, inter-token latency and end-to-end latency from the SSE stream.

    llmserve loadtest --base-url http://127.0.0.1:8000 --rate 1 --num-requests 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from llmserve.bench import workloads
from llmserve.bench.metrics import RequestRecord, summarize
from llmserve.envinfo import env_info


async def _one(client, url, model, req, arrival, t0, rec: RequestRecord):
    await asyncio.sleep(max(0.0, t0 + arrival - time.monotonic()))
    body = {
        "model": model,
        "prompt": req.prompt_token_ids,
        "max_tokens": req.output_len,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    async with client.stream("POST", url, json=body) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])
            if ch.get("choices") and ch["choices"][0].get("text"):
                rec.token_times.append(time.monotonic() - t0)
    rec.first_token = rec.token_times[0]
    rec.finish = rec.token_times[-1]


async def run(args) -> dict:
    from transformers import AutoTokenizer

    from llmserve.models.hub import resolve_model_path

    tok = AutoTokenizer.from_pretrained(str(resolve_model_path(args.tokenizer or args.model)))
    reqs = workloads.sharegpt(
        tok, args.num_requests, args.seed, args.max_prompt_len, args.max_output_len
    )
    arrivals = workloads.poisson_arrivals(len(reqs), args.rate, args.seed)
    records = [
        RequestRecord(a, r.prompt_len, r.output_len) for r, a in zip(reqs, arrivals, strict=True)
    ]
    url = args.base_url.rstrip("/") + "/v1/completions"
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=1000)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        t0 = time.monotonic()
        await asyncio.gather(
            *(
                _one(client, url, args.model, r, a, t0, rec)
                for r, a, rec in zip(reqs, arrivals, records, strict=True)
            )
        )
        duration = time.monotonic() - t0
    return {
        "env": env_info(),
        "args": {k: (str(v) if v == float("inf") else v) for k, v in vars(args).items()},
        "workload": workloads.summarize(reqs),
        "summary": summarize(records, duration),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="llmserve loadtest", description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--num-requests", type=int, default=100)
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-output-len", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = asyncio.run(run(args))
    s = res["summary"]
    print(
        f"{s['output_throughput']:.1f} out tok/s, TTFT p50 {s['ttft_ms']['p50']:.0f} ms "
        f"p99 {s['ttft_ms']['p99']:.0f} ms, ITL p50 {s['itl_ms']['p50']:.1f} ms"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
