"""Model registry and downloads from the Hugging Face Hub.

Only config, tokenizer and safetensors files are fetched. Weights land in ``models/<alias>/``
(git-ignored).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Short aliases used throughout the repo, configs and results files.
MODEL_REGISTRY: dict[str, str] = {
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}

ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def models_dir() -> Path:
    """Root folder for downloaded weights (override with ``LLMSERVE_MODELS``)."""
    return Path(os.environ.get("LLMSERVE_MODELS", Path.cwd() / "models"))


def resolve_model_path(name_or_path: str, download: bool = True) -> Path:
    """Map an alias, a hub repo id or a local folder to a local folder with weights."""
    p = Path(name_or_path)
    if p.is_dir():
        return p
    repo_id = MODEL_REGISTRY.get(name_or_path, name_or_path)
    alias = name_or_path if name_or_path in MODEL_REGISTRY else repo_id.replace("/", "--")
    local = models_dir() / alias
    if (local / "config.json").exists() and any(local.glob("*.safetensors")):
        return local
    if not download:
        raise FileNotFoundError(f"model {name_or_path!r} not found at {local}")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(local), allow_patterns=ALLOW_PATTERNS)
    return local


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download models by alias or hub repo id.")
    ap.add_argument("models", nargs="*", help="aliases or repo ids")
    ap.add_argument("--list", action="store_true", help="list known aliases")
    args = ap.parse_args(argv)
    if args.list or not args.models:
        for alias, repo in MODEL_REGISTRY.items():
            print(f"{alias:14s} {repo}")
        return
    for m in args.models:
        print(m, "->", resolve_model_path(m))


if __name__ == "__main__":
    main()
