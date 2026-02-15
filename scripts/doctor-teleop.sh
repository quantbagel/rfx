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

  echo "No Python interpreter found." >&2
  exit 1
}

PYTHON_BIN="$(resolve_python)"

echo "[doctor-teleop] python: $PYTHON_BIN"

"$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import sys

status = {
    "python_ok": True,
    "rfx_importable": False,
    "rust_bindings": False,
    "torch_available": False,
    "opencv_available": False,
    "jit_backends": {},
    "camera_probe": [],
}

status["torch_available"] = importlib.util.find_spec("torch") is not None
status["opencv_available"] = importlib.util.find_spec("cv2") is not None

try:
    import rfx

    status["rfx_importable"] = True
    status["rust_bindings"] = getattr(rfx, "_RUST_AVAILABLE", False)
except Exception as exc:
    print(f"[doctor-teleop] error: failed to import rfx: {exc}", file=sys.stderr)
    print(json.dumps(status, indent=2, sort_keys=True))
    sys.exit(1)

try:
    from rfx.jit import available_backends

    status["jit_backends"] = available_backends()
except Exception:
    status["jit_backends"] = {"cpu": True, "cuda": False, "metal": False}

if status["opencv_available"]:
    try:
        import cv2
    except Exception:
        cv2 = None
    if cv2 is not None:
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            opened = cap.isOpened()
            status["camera_probe"].append({"index": idx, "opened": bool(opened)})
            cap.release()

print(json.dumps(status, indent=2, sort_keys=True))

errors = []
if not status["rfx_importable"]:
    errors.append("rfx import failed")
if not status["rust_bindings"]:
    errors.append("rfx Rust extension not available (build with maturin develop)")
if not status["torch_available"]:
    errors.append("torch is unavailable (required for SO-101 teleop actions)")

if errors:
    print("[doctor-teleop] failures:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    sys.exit(2)

print("[doctor-teleop] ready")
PY

