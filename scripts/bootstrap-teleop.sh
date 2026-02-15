#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_PATH="${RFX_VENV_PATH:-$ROOT/.venv}"
PYTHON_BIN="$VENV_PATH/bin/python"

echo "[bootstrap-teleop] running source setup"
bash scripts/setup-from-source.sh

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[bootstrap-teleop] missing python at $PYTHON_BIN" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  echo "[bootstrap-teleop] installing teleop extras"
  uv pip install --python "$PYTHON_BIN" -e "$ROOT[teleop]"
else
  echo "[bootstrap-teleop] installing teleop extras with pip"
  "$PYTHON_BIN" -m pip install -e "$ROOT[teleop]"
fi

echo "[bootstrap-teleop] running diagnostics"
bash scripts/doctor-teleop.sh

echo "[bootstrap-teleop] complete"

