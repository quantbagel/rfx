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

run_lint_ratchet() {
  local python_bin="$1"
  local lint_targets_file="$ROOT/scripts/python-lint-ratchet.txt"
  local lint_targets=()

  if [[ ! -f "$lint_targets_file" ]]; then
    echo "Target file not found: $lint_targets_file" >&2
    exit 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    lint_targets+=("$line")
  done < "$lint_targets_file"

  if (( ${#lint_targets[@]} == 0 )); then
    echo "No targets found in: $lint_targets_file" >&2
    exit 1
  fi

  "$python_bin" -m ruff check "${lint_targets[@]}"
}

run_typecheck_targets() {
  local python_bin="$1"
  local target_file="$2"
  local targets=()

  if [[ ! -f "$target_file" ]]; then
    echo "Target file not found: $target_file" >&2
    exit 1
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    targets+=("$line")
  done < "$target_file"

  if (( ${#targets[@]} == 0 )); then
    echo "No targets found in: $target_file" >&2
    exit 1
  fi

  "$python_bin" -m mypy --follow-imports=skip --ignore-missing-imports "${targets[@]}"
}

run_lint() {
  local python_bin="$1"
  require_python_module "$python_bin" ruff

  "$python_bin" -m ruff check --select E9,F63,F7,F82 rfx/python/ rfxJIT/
  "$python_bin" -m ruff format --check rfx/python/ rfxJIT/

  case "${RFX_LINT_STAGE:-ratchet}" in
    baseline)
      ;;
    ratchet)
      run_lint_ratchet "$python_bin"
      ;;
    full)
      "$python_bin" -m ruff check rfx/python/ rfxJIT/
      ;;
    *)
      echo "Invalid RFX_LINT_STAGE=${RFX_LINT_STAGE:-}. Expected baseline|ratchet|full." >&2
      exit 1
      ;;
  esac
}

run_typecheck() {
  local python_bin="$1"
  require_python_module "$python_bin" mypy

  run_typecheck_targets "$python_bin" "$ROOT/scripts/python-typecheck-baseline.txt"

  case "${RFX_TYPECHECK_STAGE:-ratchet}" in
    baseline)
      ;;
    ratchet)
      run_typecheck_targets "$python_bin" "$ROOT/scripts/python-typecheck-ratchet.txt"
      ;;
    full)
      run_typecheck_targets "$python_bin" "$ROOT/scripts/python-typecheck-ratchet.txt"
      run_typecheck_full "$python_bin"
      ;;
    *)
      echo "Invalid RFX_TYPECHECK_STAGE=${RFX_TYPECHECK_STAGE:-}. Expected baseline|ratchet|full." >&2
      exit 1
      ;;
  esac
}

run_typecheck_full() {
  local python_bin="$1"
  require_python_module "$python_bin" mypy

  "$python_bin" -m mypy rfx/python/rfx/ --ignore-missing-imports
}

run_test() {
  local python_bin="$1"
  require_python_module "$python_bin" pytest

  PYTHONPATH="$ROOT:$ROOT/rfx/python:${PYTHONPATH:-}" "$python_bin" -m pytest rfx/tests/ rfxJIT/tests/ -q
}

run_build() {
  local maturin_bin
  local uv_cache_dir
  local install_mode
  maturin_bin="$(resolve_maturin)"
  uv_cache_dir="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
  install_mode="${RFX_MATURIN_INSTALL_MODE:-auto}"

  mkdir -p "$uv_cache_dir"

  case "$install_mode" in
    install)
      UV_CACHE_DIR="$uv_cache_dir" "$maturin_bin" develop --release
      ;;
    skip)
      UV_CACHE_DIR="$uv_cache_dir" "$maturin_bin" develop --release --skip-install
      ;;
    auto)
      if ! UV_CACHE_DIR="$uv_cache_dir" "$maturin_bin" develop --release; then
        echo "maturin develop install failed; retrying with --skip-install" >&2
        UV_CACHE_DIR="$uv_cache_dir" "$maturin_bin" develop --release --skip-install
      fi
      ;;
    *)
      echo "Invalid RFX_MATURIN_INSTALL_MODE=${RFX_MATURIN_INSTALL_MODE:-}. Expected install|skip|auto." >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<'USAGE'
Usage: scripts/python-checks.sh <lint|typecheck|typecheck-full|test|build|ci>

Stages:
  RFX_LINT_STAGE=baseline|ratchet|full          (default: ratchet)
  RFX_TYPECHECK_STAGE=baseline|ratchet|full     (default: ratchet)
  UV_CACHE_DIR=/path/to/cache                   (default: .cache/uv in repo)
  RFX_MATURIN_INSTALL_MODE=install|skip|auto    (default: auto)
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

      if [[ "${RFX_TYPECHECK_FULL:-0}" == "1" && "${RFX_TYPECHECK_STAGE:-ratchet}" != "full" ]]; then
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
