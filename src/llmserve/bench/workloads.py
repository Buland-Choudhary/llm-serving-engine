"""Benchmark workloads.

* ``sharegpt``: real user prompts from the ShareGPT dataset, with the *reference reply's* token
  count as each request's output length (the standard setup in vLLM/SGLang serving benchmarks).
* ``synthetic``: random-token prompts with fixed or uniform lengths.
* ``shared-prefix``: one long system prompt shared by every request + a short unique question
  (the case prefix caching is built for).

Outputs are forced to their target length (``ignore_eos``) so every system does the same work.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

SHAREGPT_FILE = Path("data/raw/ShareGPT_V3_unfiltered_cleaned_split.json")


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    output_len: int

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


def sharegpt(
    tokenizer,
    num_requests: int,
    seed: int = 0,
    max_prompt_len: int = 512,
    max_output_len: int = 128,
    min_len: int = 4,
    path: Path = SHAREGPT_FILE,
) -> list[BenchRequest]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Download with:\n  huggingface-cli download --repo-type dataset "
            "anon8231489123/ShareGPT_Vicuna_unfiltered ShareGPT_V3_unfiltered_cleaned_split.json "
            "--local-dir data/raw"
        )
    data = json.loads(path.read_text())
    convs = [
        d["conversations"]
        for d in data
        if len(d.get("conversations", [])) >= 2
        and d["conversations"][0]["from"] == "human"
        and d["conversations"][1]["from"] == "gpt"
    ]
    rng = random.Random(seed)
    rng.shuffle(convs)
    out: list[BenchRequest] = []
    for c in convs:
        p = tokenizer(c[0]["value"], add_special_tokens=False).input_ids
        o = len(tokenizer(c[1]["value"], add_special_tokens=False).input_ids)
        if not (min_len <= len(p) <= max_prompt_len and min_len <= o):
            continue
        out.append(BenchRequest(p, min(o, max_output_len)))
        if len(out) == num_requests:
            break
    return out


def synthetic(
    num_requests: int,
    prompt_len: int | tuple[int, int],
    output_len: int | tuple[int, int],
    vocab_size: int,
    seed: int = 0,
) -> list[BenchRequest]:
    rng = random.Random(seed)

    def pick(x):
        return rng.randint(*x) if isinstance(x, tuple) else x

    return [
        BenchRequest(
            [rng.randrange(100, vocab_size - 100) for _ in range(pick(prompt_len))],
            pick(output_len),
        )
        for _ in range(num_requests)
    ]


def shared_prefix(
    tokenizer,
    num_requests: int,
    prefix_len: int = 1024,
    question_len: int = 32,
    output_len: int = 32,
    seed: int = 0,
) -> list[BenchRequest]:
    """Same long system prompt + a unique question per request (e.g. an assistant with a big
    instruction block or a document every user asks about)."""
    rng = random.Random(seed)
    vocab = tokenizer.vocab_size
    prefix = [rng.randrange(100, vocab - 100) for _ in range(prefix_len)]
    return [
        BenchRequest(
            prefix + [rng.randrange(100, vocab - 100) for _ in range(question_len)], output_len
        )
        for _ in range(num_requests)
    ]


def rewrite(
    tokenizer,
    num_requests: int,
    seed: int = 0,
    min_passage_tokens: int = 120,
    max_passage_tokens: int = 400,
    path: Path = SHAREGPT_FILE,
) -> list[BenchRequest]:
    """ "Fix the typos in this text" over real ShareGPT replies, in the model's chat format.

    The answer mostly copies the passage, the case n-gram (prompt lookup) speculation targets
    (also typical of code edits and RAG). Output length = passage length.
    """
    data = json.loads(path.read_text())
    texts = [
        d["conversations"][1]["value"]
        for d in data
        if len(d.get("conversations", [])) >= 2 and d["conversations"][1]["from"] == "gpt"
    ]
    rng = random.Random(seed)
    rng.shuffle(texts)
    out: list[BenchRequest] = []
    for t in texts:
        n = len(tokenizer(t, add_special_tokens=False).input_ids)
        if not (min_passage_tokens <= n <= max_passage_tokens) or "```" in t:
            continue
        msg = "Rewrite the following text exactly, only fixing spelling mistakes:\n\n" + t
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True
        )
        out.append(BenchRequest(tokenizer(prompt, add_special_tokens=False).input_ids, n))
        if len(out) == num_requests:
            break
    return out


def poisson_arrivals(n: int, rate: float, seed: int = 0) -> list[float]:
    """Arrival offsets in seconds for a Poisson process with ``rate`` requests/second."""
    if rate == float("inf"):
        return [0.0] * n
    rng = random.Random(seed + 1)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def summarize(reqs: list[BenchRequest]) -> dict:
    p = sorted(r.prompt_len for r in reqs)
    o = sorted(r.output_len for r in reqs)
    h = hashlib.sha256(json.dumps([asdict(r) for r in reqs]).encode()).hexdigest()[:12]
    return {
        "num_requests": len(reqs),
        "prompt_len_mean": sum(p) / len(p),
        "prompt_len_p50": p[len(p) // 2],
        "prompt_len_max": p[-1],
        "output_len_mean": sum(o) / len(o),
        "output_len_p50": o[len(o) // 2],
        "output_len_max": o[-1],
        "total_prompt_tokens": sum(p),
        "total_output_tokens": sum(o),
        "workload_hash": h,
    }
