#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'USAGE'
Usage: cli/rfx.sh <doctor|bootstrap|setup-source|check>
USAGE
}

doctor() {
  for tool in cargo git bash; do
    if command -v "$tool" >/dev/null 2>&1; then
      echo "[ok] $tool"
    else
      echo "[missing] $tool"
    fi
  done

  for tool in python3 python uv moon; do
    if command -v "$tool" >/dev/null 2>&1; then
      echo "[ok] $tool"
    fi
  done
}

bootstrap() {
  bash .claude/skills/rfx-bootstrap-install/scripts/bootstrap.sh
}

setup_source() {
  bash scripts/setup-from-source.sh
}

check() {
  cargo fmt --all -- --check
  cargo clippy --workspace --all-targets --all-features -- -D warnings
  scripts/python-checks.sh ci
}

main() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 1
  fi

  case "$1" in
    doctor)
      doctor
      ;;
    bootstrap)
      bootstrap
      ;;
    setup-source)
      setup_source
      ;;
    check)
      check
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
