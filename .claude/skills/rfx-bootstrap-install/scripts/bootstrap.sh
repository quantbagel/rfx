#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"

echo "[bootstrap] repo root: $ROOT"

echo "[bootstrap] setting up source environment"
bash "$ROOT/scripts/setup-from-source.sh"

echo "[bootstrap] enabling git hooks"
./scripts/setup-git-hooks.sh

echo "[bootstrap] complete"
