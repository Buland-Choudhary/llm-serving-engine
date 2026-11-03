#!/usr/bin/env python3
"""Pre-commit guard: block secrets, weights and large files from entering git."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 2_000_000
BLOCKED_SUFFIXES = {".safetensors", ".bin", ".gguf", ".pt", ".pth", ".ckpt", ".so", ".onnx"}
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]


def staged_files() -> list[Path]:
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(p) for p in out.splitlines() if p]


def main() -> int:
    problems = []
    for p in staged_files():
        if not p.exists():
            continue
        if p.suffix in BLOCKED_SUFFIXES or p.name.startswith(".env"):
            problems.append(f"{p}: blocked file type")
            continue
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"{p}: larger than {MAX_BYTES} bytes")
            continue
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                problems.append(f"{p}: looks like it contains a secret ({pat.pattern[:12]}...)")
    for msg in problems:
        print("repo-safety:", msg, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
