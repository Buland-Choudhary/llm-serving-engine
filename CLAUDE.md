# CLAUDE.md: rules for any Claude session working on this repo

## Start of every session
1. Run `uptime`. If the machine was reset, recreate the venv (see Commands) and check `runs/`.
2. Read `PLAN.md` (scope, milestones) and `PROGRESS.md` (where we are, what's next).
3. Pick the next unchecked item in the current milestone and work through it.

## Golden rules
- **Never invent numbers.** Every number in README, docs or results comes from a results file written
  by a real run (with git SHA, CPU, torch version, model, config). If a run didn't happen, write "TBD".
- **Correctness before speed.** A feature is done when a test proves it gives the same output as the
  reference path (HF transformers, or the engine with the feature turned off).
- **Public repo:** never commit secrets, tokens, `.env` files, model weights, raw datasets, or personal
  info. `scripts/check_repo_safety.py` runs as a pre-commit hook.
- **Honest benchmarks:** same prompts, output lengths and decoding for every system compared; always
  include the HF baseline; say "CPU" next to every number.
- Scope changes go into the Decision Log in `PLAN.md`.
- Work in tier order: finish Tier 1 and tag `v1.0` before Tier 2.

## Git
- Identity (set once per clone):
  `git config user.name "Buland Choudhary"`
  `git config user.email "101297124+Buland-Choudhary@users.noreply.github.com"`
- Solo repo: commit straight to `main`. Small commits, pushed right away (at least every ~30-60 min of
  work). Run `pytest` and `ruff` before pushing.
- Commit messages: imperative and specific, e.g. `Add block allocator with ref counts`.

## Environment notes
- Cloud machine: 4 vCPU Xeon (AVX-512, VNNI, no AMX/bf16), 15 GB RAM, no GPU. fp32 is the fast dtype.
- Hugging Face, PyPI and download.pytorch.org are reachable.
- The cloud machine can be reset while the session is idle; detached jobs die. Keep long jobs
  resumable and in chunks of <= 30 minutes; run one heavy CPU job at a time.
- Never use `pkill -f`/`pgrep -f` with text that also appears in your own command; kill by PID.

## Where things go
| Path | What | In git? |
|---|---|---|
| `src/llmserve/` | package code | yes |
| `tests/` | pytest; tiny random models by default, real checkpoints only under `@pytest.mark.slow` | yes |
| `benchmarks/` | benchmark configs and scripts | yes |
| `results/` | result JSONs, tables, charts | yes |
| `docs/` | report and design notes | yes |
| `models/`, `data/raw/`, `runs/` | weights, downloads, scratch | **no** |

## End of every working block
- Update `PROGRESS.md` (what was done, results with paths, what's next, blockers).
- Tick finished checkboxes in `PLAN.md`. Commit and push.

## Commands
- setup: `uv venv --python python3.13 .venv && uv sync --extra dev` then `scripts/install_hooks.sh`
- test / lint: `.venv/bin/pytest` (add `-m slow` for real-checkpoint tests) and
  `.venv/bin/ruff check . && .venv/bin/ruff format .`
