"""Upload the demo Space: `HF_TOKEN=... python space/deploy.py [repo_id]`."""

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

root = Path(__file__).resolve().parent.parent
api = HfApi()
repo_id = sys.argv[1] if len(sys.argv) > 1 else f"{api.whoami()['name']}/llmserve-demo"
api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
with tempfile.TemporaryDirectory() as d:
    out = Path(d)
    for f in ("Dockerfile", "README.md"):
        shutil.copy(root / "space" / f, out / f)
    for f in ("pyproject.toml", "uv.lock", "LICENSE"):
        shutil.copy(root / f, out / f)
    shutil.copytree(
        root / "src", out / "src", ignore=shutil.ignore_patterns("__pycache__", "*.so", "build")
    )
    api.upload_folder(
        folder_path=d, repo_id=repo_id, repo_type="space", commit_message="Deploy llmserve demo"
    )
print(f"https://huggingface.co/spaces/{repo_id}")
