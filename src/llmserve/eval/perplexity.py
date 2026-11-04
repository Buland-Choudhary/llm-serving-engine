"""WikiText-2 perplexity and greedy agreement, for checking quantization quality.

llmserve ppl --model qwen3-0.6b --quantization int4 --max-tokens 40000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from llmserve.attention import AttentionMetadata
from llmserve.kv_cache import KVCache

WIKITEXT = "Salesforce/wikitext"
WIKITEXT_FILES = {
    "test": "wikitext-2-raw-v1/test-00000-of-00001.parquet",
    "train": "wikitext-2-raw-v1/train-00000-of-00001.parquet",
}


def wikitext_text(split: str = "test") -> str:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        WIKITEXT, WIKITEXT_FILES[split], repo_type="dataset", local_dir="data/raw/wikitext"
    )
    import pyarrow.parquet as pq

    return "\n\n".join(pq.read_table(p).column("text").to_pylist())


@torch.inference_mode()
def forward_logits(model, ids: list[int], block_size: int = 16) -> torch.Tensor:
    """Logits for every position of one sequence (single prefill through the paged cache)."""
    n = len(ids)
    nb = -(-n // block_size)
    cache = KVCache(model.cfg, nb, block_size, next(model.parameters()).dtype)
    meta = AttentionMetadata.build([n], [n], [list(range(nb))], block_size)
    h = model(torch.tensor(ids), torch.arange(n), cache.layers, meta)
    return model.compute_logits(h)


@torch.inference_mode()
def perplexity(
    model, token_ids: list[int], window: int = 1024, max_tokens: int | None = None
) -> dict:
    """Non-overlapping windows; each window's first token is not scored."""
    ids = token_ids[: max_tokens or len(token_ids)]
    nll, count = 0.0, 0
    for s in range(0, len(ids) - 1, window):
        chunk = ids[s : s + window]
        if len(chunk) < 2:
            break
        logits = forward_logits(model, chunk)
        lp = torch.log_softmax(logits[:-1], -1)
        tgt = torch.tensor(chunk[1:])
        nll -= lp.gather(1, tgt[:, None]).sum().item()
        count += len(chunk) - 1
    return {"ppl": math.exp(nll / count), "nll": nll / count, "tokens": count, "window": window}


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoTokenizer

    from llmserve.envinfo import env_info
    from llmserve.models.hub import resolve_model_path
    from llmserve.models.loader import load_model
    from llmserve.quant import weight_bytes

    ap = argparse.ArgumentParser(prog="llmserve ppl")
    ap.add_argument("--model", default="smollm2-135m")
    ap.add_argument("--quantization", default=None, choices=["int8", "int4", "int4-awq"])
    ap.add_argument("--quantize-lm-head", action="store_true")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    path = resolve_model_path(args.model)
    tok = AutoTokenizer.from_pretrained(str(path))
    t0 = time.time()
    model = load_model(
        path,
        quantization=args.quantization,
        quantize_lm_head=args.quantize_lm_head,
        group_size=args.group_size,
    )
    load_s = time.time() - t0
    ids = tok(wikitext_text("test"), add_special_tokens=False).input_ids
    t0 = time.time()
    res = perplexity(model, ids, args.window, args.max_tokens)
    res.update(
        model=args.model,
        quantization=args.quantization or "fp32",
        quantize_lm_head=args.quantize_lm_head,
        group_size=args.group_size,
        weight_bytes=weight_bytes(model),
        load_s=load_s,
        eval_s=time.time() - t0,
        env=env_info(),
    )
    print(json.dumps({k: v for k, v in res.items() if k != "env"}))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
