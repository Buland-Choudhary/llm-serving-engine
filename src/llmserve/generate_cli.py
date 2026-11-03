"""``llmserve generate``: quick text generation from the command line."""

from __future__ import annotations

import argparse
import time

from llmserve.engine import LLM
from llmserve.sampling import SamplingParams


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="llmserve generate")
    ap.add_argument("prompt", nargs="+")
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--chat", action="store_true", help="apply the model's chat template")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)
    llm = LLM(args.model)
    sp = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
    )
    t = time.monotonic()
    if args.chat:
        outs = llm.chat([[{"role": "user", "content": p}] for p in args.prompt], sp)
    else:
        outs = llm.generate(args.prompt, sp)
    dt = time.monotonic() - t
    n = 0
    for p, o in zip(args.prompt, outs, strict=True):
        n += len(o.outputs[0].token_ids)
        print(f"### {p}\n{o.outputs[0].text}\n")
    print(f"[{n} tokens in {dt:.2f}s, {n / dt:.1f} tok/s]")


if __name__ == "__main__":
    main()
