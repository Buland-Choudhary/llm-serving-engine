"""``llmserve bench``: run one system on one workload and write a result JSON."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from llmserve.bench import workloads
from llmserve.bench.metrics import summarize
from llmserve.envinfo import env_info
from llmserve.models.hub import resolve_model_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llmserve bench", description=__doc__)
    ap.add_argument("--system", choices=["engine", "hf-seq", "hf-static"], required=True)
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument(
        "--workload",
        choices=["sharegpt", "synthetic", "shared-prefix", "rewrite"],
        default="sharegpt",
    )
    ap.add_argument("--num-requests", type=int, default=200)
    ap.add_argument("--rate", type=float, default=float("inf"), help="requests/s (inf = offline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--max-output-len", type=int, default=128)
    ap.add_argument("--prompt-len", type=int, default=256, help="synthetic")
    ap.add_argument("--output-len", type=int, default=128, help="synthetic / shared-prefix")
    ap.add_argument("--prefix-len", type=int, default=1024, help="shared-prefix")
    ap.add_argument("--question-len", type=int, default=32, help="shared-prefix")
    ap.add_argument("--threads", type=int, default=4)
    # engine
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=512)
    ap.add_argument("--no-chunked-prefill", action="store_true")
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--kv-cache-gb", type=float, default=2.0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--linear-backend", default="auto")
    ap.add_argument("--attention-backend", default="auto")
    ap.add_argument("--quantization", default=None, choices=["int8", "int4", "int4-awq"])
    ap.add_argument("--quantize-lm-head", action="store_true")
    ap.add_argument("--speculative", choices=["ngram", "draft"], default=None)
    ap.add_argument("--num-spec-tokens", type=int, default=4)
    ap.add_argument("--draft-model", default=None)
    # hf
    ap.add_argument("--batch-size", type=int, default=16, help="hf-static batch size")
    # output
    ap.add_argument("--slo-ttft-ms", type=float, default=None)
    ap.add_argument("--slo-tpot-ms", type=float, default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing result")
    return ap


def make_workload(args, tokenizer, vocab_size):
    if args.workload == "sharegpt":
        return workloads.sharegpt(
            tokenizer, args.num_requests, args.seed, args.max_prompt_len, args.max_output_len
        )
    if args.workload == "rewrite":
        return workloads.rewrite(tokenizer, args.num_requests, args.seed)
    if args.workload == "synthetic":
        return workloads.synthetic(
            args.num_requests, args.prompt_len, args.output_len, vocab_size, args.seed
        )
    return workloads.shared_prefix(
        tokenizer, args.num_requests, args.prefix_len, args.question_len, args.output_len, args.seed
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"skip: {out} exists")
        return
    torch.set_num_threads(args.threads)
    from transformers import AutoTokenizer

    path = resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(str(path))
    reqs = make_workload(args, tok, tok.vocab_size)
    arrivals = workloads.poisson_arrivals(len(reqs), args.rate, args.seed)
    warm = workloads.synthetic(4, 32, 8, tok.vocab_size, seed=99)

    extra = {}
    if args.system == "engine":
        from llmserve.bench.runners import run_engine
        from llmserve.engine import EngineConfig, LLMEngine

        cfg = EngineConfig(
            model=args.model,
            kv_cache_memory_gb=args.kv_cache_gb,
            block_size=args.block_size,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enable_chunked_prefill=not args.no_chunked_prefill,
            enable_prefix_caching=not args.no_prefix_caching,
            max_model_len=max(2048, args.max_prompt_len + args.max_output_len + args.prefix_len),
            linear_backend=args.linear_backend,
            attention_backend=args.attention_backend,
            quantization=args.quantization,
            quantize_lm_head=args.quantize_lm_head,
            speculative=args.speculative,
            num_speculative_tokens=args.num_spec_tokens,
            spec_draft_model=args.draft_model,
        )
        engine = LLMEngine(cfg)
        run_engine(engine, warm, [0.0] * len(warm))
        # Warm-up must not leave cached prefixes behind for the measured run.
        engine.block_manager.__init__(
            engine.block_manager.num_blocks, cfg.block_size, cfg.enable_prefix_caching
        )
        records, duration, extra = run_engine(engine, reqs, arrivals, args.max_seconds)
        extra["engine_stats"] = engine.stats()
        steps = extra.pop("steps")
        extra["num_steps"] = len(steps)
        extra["mean_batch_seqs"] = sum(s[0] for s in steps) / max(1, len(steps))
        system_cfg = asdict(cfg)
        system_cfg["linear_backend"] = engine.model.linear_backend
        system_cfg["attention_backend"] = engine.runner.attention_backend
    else:
        from transformers import AutoModelForCausalLM

        from llmserve.bench.runners import run_hf

        model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32).eval()
        bs = 1 if args.system == "hf-seq" else args.batch_size
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        run_hf(model, warm, [0.0] * len(warm), bs, pad)
        records, duration, extra = run_hf(model, reqs, arrivals, bs, pad, args.max_seconds)
        system_cfg = {"batch_size": bs, "attn_implementation": model.config._attn_implementation}

    slo = None
    if args.slo_ttft_ms is not None and args.slo_tpot_ms is not None:
        slo = {"ttft_ms": args.slo_ttft_ms, "tpot_ms": args.slo_tpot_ms}
    result = {
        "env": env_info(),
        "args": {k: (str(v) if v == float("inf") else v) for k, v in vars(args).items()},
        "system": args.system,
        "system_config": system_cfg,
        "workload": workloads.summarize(reqs),
        "summary": summarize(records, duration, slo),
        "extra": extra,
        "records": [
            {
                "arrival": r.arrival,
                "prompt_len": r.prompt_len,
                "output_len": r.output_len,
                "ttft": r.ttft,
                "e2e": r.e2e,
                "tpot": r.tpot,
            }
            for r in records
        ],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    s = result["summary"]
    print(
        f"{args.system} {args.model} {args.workload} rate={args.rate}: "
        f"{s['output_throughput']:.1f} out tok/s, {s['request_throughput']:.2f} req/s, "
        f"TTFT p50 {s['ttft_ms']['p50']:.0f} ms, "
        f"TPOT p50 {s['tpot_ms'].get('p50', float('nan')):.1f} ms -> {out}"
    )


if __name__ == "__main__":
    main()
