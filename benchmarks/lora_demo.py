"""Multi-LoRA demo + overhead measurement on Qwen3-0.6B with two public adapters.

Adapters (downloaded to models/lora/ by `python benchmarks/lora_demo.py --fetch`):
  * charent/self_cognition_Alice: changes the assistant's self-identity
  * phh/Qwen3-0.6B-TLDR-Lora: one-line summaries
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from llmserve.engine import EngineConfig, LLMEngine
from llmserve.envinfo import env_info
from llmserve.sampling import SamplingParams

ADAPTERS = {"alice": "charent/self_cognition_Alice", "tldr": "phh/Qwen3-0.6B-TLDR-Lora"}
ARTICLE = (
    "The city council voted on Tuesday to convert three downtown parking lots into public parks. "
    "Supporters said the change will reduce flooding and give residents more green space, while "
    "some business owners worried that customers would have nowhere to park. Construction is "
    "expected to begin next spring and take about eighteen months."
)


def fetch() -> None:
    from huggingface_hub import snapshot_download

    for name, repo in ADAPTERS.items():
        snapshot_download(
            repo, local_dir=f"models/lora/{repo.split('/')[1]}", allow_patterns=["adapter_*"]
        )
        print("fetched", name)


def chat(tok, text: str) -> list[int]:
    p = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return tok(p, add_special_tokens=False).input_ids


def run(eng: LLMEngine, reqs) -> tuple[dict, float, int]:
    ids = {}
    t = time.monotonic()
    for i, (prompt, lora, sp) in enumerate(reqs):
        ids[eng.add_request(prompt, sp, request_id=f"r{i}", lora=lora)] = i
    done, ntok = {}, 0
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                done[ids[o.request_id]] = o.outputs[0].text
                ntok += len(o.outputs[0].token_ids)
    return done, time.monotonic() - t, ntok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default="results/lora/qwen3-0.6b_multi_lora.json")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    torch.set_num_threads(4)
    eng = LLMEngine(EngineConfig(model="qwen3-0.6b", max_num_seqs=64))
    for name, repo in ADAPTERS.items():
        eng.load_lora(name, f"models/lora/{repo.split('/')[1]}")
    tok = eng.tokenizer
    greedy = SamplingParams(temperature=0, max_tokens=48)

    # 1) one mixed batch: base + both adapters at once
    demo = [
        (chat(tok, "Who are you? Answer in one sentence."), None, greedy),
        (chat(tok, "Who are you? Answer in one sentence."), "alice", greedy),
        (chat(tok, "Summarize in one sentence:\n\n" + ARTICLE), None, greedy),
        (chat(tok, "Summarize in one sentence:\n\n" + ARTICLE), "tldr", greedy),
    ]
    texts, _, _ = run(eng, demo)
    labels = ["base: who are you", "alice: who are you", "base: summarize", "tldr: summarize"]
    for lab, i in zip(labels, range(4), strict=True):
        print(f"--- {lab}\n{texts[i].strip()}\n")

    # 2) overhead: 48 requests x 64 tokens, all base vs mixed across adapters
    fixed = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True)
    prompts = [chat(tok, f"Write a short note about topic number {i}.") for i in range(48)]
    results = {}
    for mode, loras in {
        "base_only": [None],
        "one_adapter": ["alice"],
        "mixed_3way": [None, "alice", "tldr"],
    }.items():
        reqs = [(p, loras[i % len(loras)], fixed) for i, p in enumerate(prompts)]
        run(eng, reqs[:4])  # warm-up
        _, dt, n = run(eng, reqs)
        results[mode] = {"tokens": n, "seconds": dt, "tok_s": n / dt}
        print(mode, f"{n / dt:.1f} tok/s")
    out = {
        "env": env_info(),
        "demo": dict(zip(labels, [texts[i] for i in range(4)], strict=True)),
        "throughput": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
