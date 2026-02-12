#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

require_cmd() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required command: $tool" >&2
    exit 1
  fi
}

require_cmd cargo
require_cmd uv

VENV_PATH="${RFX_VENV_PATH:-$ROOT/.venv}"
PYTHON_BIN="$VENV_PATH/bin/python"

echo "[setup] creating uv virtualenv at $VENV_PATH"
uv venv "$VENV_PATH"

echo "[setup] installing requirements-dev.txt"
uv pip install --python "$PYTHON_BIN" -r "$ROOT/requirements-dev.txt"

echo "[setup] installing rfx from source (editable)"
uv pip install --python "$PYTHON_BIN" -e "$ROOT"

echo "[setup] fetching Rust crates"
cargo fetch

echo "[setup] complete"
echo "[setup] activate with: source $VENV_PATH/bin/activate"
