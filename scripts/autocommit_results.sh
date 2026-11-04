#!/usr/bin/env bash
# Commit and push new benchmark result files every 10 minutes while queues run.
cd "$(dirname "$0")/.."
while true; do
  if [ -n "$(git status --porcelain results)" ]; then
    git add results && git commit -qm "Add benchmark results ($(date -u +%H:%M) UTC)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JPU6WKG14a8adFhWPrtjdh" && git push -q origin main
  fi
  sleep 600
done
