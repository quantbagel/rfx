#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

chmod +x .githooks/pre-commit .githooks/pre-push
chmod +x scripts/python-checks.sh
git config core.hooksPath .githooks

echo "Git hooks enabled via core.hooksPath=.githooks"
echo "pre-commit: cargo fmt + ruff (staged Python files)"
echo "pre-push: fmt, clippy, tests, ruff, mypy subset, pytest"
