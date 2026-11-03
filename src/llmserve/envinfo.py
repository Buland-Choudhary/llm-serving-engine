"""Environment fingerprint stored with every result file."""

from __future__ import annotations

import os
import platform
import subprocess
import time

import torch


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def env_info() -> dict:
    return {
        "git_sha": git_sha(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cpu": cpu_model(),
        "num_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": "cpu",
    }
