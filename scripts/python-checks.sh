#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

resolve_python() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "$ROOT/.venv/bin/python"
    return
  fi

  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi

  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi

  echo "No Python interpreter found. Create .venv or install python3." >&2
  exit 1
}

require_python_module() {
  local python_bin="$1"
  local module_name="$2"
  if ! "$python_bin" -m "$module_name" --version >/dev/null 2>&1; then
    echo "Missing Python tool: $module_name" >&2
    echo "Install dev tools with: uv pip install -e '.[dev]'" >&2
    exit 1
  fi
}

resolve_maturin() {
  if [[ -x "$ROOT/.venv/bin/maturin" ]]; then
    echo "$ROOT/.venv/bin/maturin"
    return
  fi

  if command -v maturin >/dev/null 2>&1; then
    command -v maturin
    return
  fi

  echo "Missing Python tool: maturin" >&2
  echo "Install build tools with: uv pip install maturin" >&2
  exit 1
}

run_lint() {
  local python_bin="$1"
  require_python_module "$python_bin" ruff

  "$python_bin" -m ruff check --select E9,F63,F7,F82 src/python/
  "$python_bin" -m ruff format --check src/python/
}

run_typecheck() {
  local python_bin="$1"
  require_python_module "$python_bin" mypy

  "$python_bin" -m mypy \
    --follow-imports=skip \
    --ignore-missing-imports \
    src/python/rfx/observation.py \
    src/python/rfx/utils/padding.py \
    src/python/rfx/utils/transforms.py
}

run_typecheck_full() {
  local python_bin="$1"
  require_python_module "$python_bin" mypy

  "$python_bin" -m mypy src/python/rfx/ --ignore-missing-imports
}

run_test() {
  local python_bin="$1"
  require_python_module "$python_bin" pytest

  PYTHONPATH="$ROOT/src/python:${PYTHONPATH:-}" "$python_bin" -m pytest src/tests/ -q
}

run_build() {
  local maturin_bin
  maturin_bin="$(resolve_maturin)"

  "$maturin_bin" develop --release
}

usage() {
  cat <<'USAGE'
Usage: scripts/python-checks.sh <lint|typecheck|typecheck-full|test|build|ci>
USAGE
}

main() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 1
  fi

  local mode="$1"
  local python_bin
  python_bin="$(resolve_python)"

  case "$mode" in
    lint)
      run_lint "$python_bin"
      ;;
    typecheck)
      run_typecheck "$python_bin"
      ;;
    typecheck-full)
      run_typecheck_full "$python_bin"
      ;;
    test)
      run_test "$python_bin"
      ;;
    build)
      run_build
      ;;
    ci)
      run_lint "$python_bin"
      run_typecheck "$python_bin"
      run_test "$python_bin"

      if [[ "${RFX_TYPECHECK_FULL:-0}" == "1" ]]; then
        run_typecheck_full "$python_bin"
      fi
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
