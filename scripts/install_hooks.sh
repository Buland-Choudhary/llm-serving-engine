#!/usr/bin/env bash
# Install the repo-safety pre-commit hook.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
cat > "$root/.git/hooks/pre-commit" <<'HOOK'
#!/usr/bin/env bash
exec python3 "$(git rev-parse --show-toplevel)/scripts/check_repo_safety.py"
HOOK
chmod +x "$root/.git/hooks/pre-commit"
echo "pre-commit hook installed"
