#!/usr/bin/env bash
# Run several queue files back to back (each skips results that already exist).
cd "$(dirname "$0")/.."
for q in "$@"; do .venv/bin/python scripts/run_queue.py "$q"; done
