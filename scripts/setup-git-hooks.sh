#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

resolve_pre_commit() {
  if [[ -n "${RFX_VENV_PATH:-}" && -x "${RFX_VENV_PATH}/bin/pre-commit" ]]; then
    echo "${RFX_VENV_PATH}/bin/pre-commit"
    return
  fi

  if [[ -x "$ROOT/.venv/bin/pre-commit" ]]; then
    echo "$ROOT/.venv/bin/pre-commit"
    return
  fi

  if command -v pre-commit >/dev/null 2>&1; then
    command -v pre-commit
    return
  fi

  echo "Missing required command: pre-commit" >&2
  echo "Install dev dependencies first: bash scripts/setup-from-source.sh" >&2
  exit 1
}

chmod +x .githooks/pre-commit .githooks/pre-push
chmod +x scripts/python-checks.sh

PRE_COMMIT_BIN="$(resolve_pre_commit)"

# Pre-commit refuses to install when core.hooksPath is set.
git config --local --unset-all core.hooksPath >/dev/null 2>&1 || true

if git config --get-all core.hooksPath >/dev/null 2>&1; then
  echo "core.hooksPath is set outside this repo; pre-commit install will refuse." >&2
  echo "Unset it (for example: git config --global --unset-all core.hooksPath)." >&2
  exit 1
fi

"$PRE_COMMIT_BIN" install --install-hooks --hook-type pre-commit --hook-type pre-push

echo "Git hooks installed via pre-commit"
echo "pre-commit: cargo fmt + ruff (staged Python files)"
echo "pre-push: fmt, clippy, tests, ruff, mypy subset, pytest"
echo "Run push-equivalent local gates with: $PRE_COMMIT_BIN run --all-files --hook-stage pre-push"
