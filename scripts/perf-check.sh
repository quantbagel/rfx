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

usage() {
  cat <<'USAGE'
Usage: scripts/perf-check.sh --baseline <path> [options]

Options:
  --baseline <path>        Baseline benchmark JSON (required)
  --output <path>          Path for current benchmark JSON
  --backend <name>         cpu|cuda|metal|auto (default: cpu)
  --size <int>             Benchmark tensor size (default: 65536)
  --iterations <int>       Timing iterations (default: 200)
  --warmup <int>           Warmup iterations (default: 10)
  --seed <int>             RNG seed (default: 42)
  --threshold-pct <float>  Regression warning threshold percent (default: 10)
  --fail-on-regression     Exit non-zero when regressions exceed threshold
  -h, --help               Show this help
USAGE
}

BASELINE_PATH=""
OUTPUT_PATH=""
BACKEND="cpu"
SIZE=65536
ITERATIONS=200
WARMUP=10
SEED=42
THRESHOLD_PCT=10
FAIL_ON_REGRESSION=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline)
      BASELINE_PATH="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --size)
      SIZE="${2:-}"
      shift 2
      ;;
    --iterations)
      ITERATIONS="${2:-}"
      shift 2
      ;;
    --warmup)
      WARMUP="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --threshold-pct)
      THRESHOLD_PCT="${2:-}"
      shift 2
      ;;
    --fail-on-regression)
      FAIL_ON_REGRESSION=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$BASELINE_PATH" ]]; then
  echo "Missing required argument: --baseline <path>" >&2
  usage
  exit 1
fi

if [[ ! -f "$BASELINE_PATH" ]]; then
  echo "Baseline file not found: $BASELINE_PATH" >&2
  exit 1
fi

if [[ -z "$OUTPUT_PATH" ]]; then
  OUTPUT_PATH="$(mktemp "${TMPDIR:-/tmp}/rfxjit-perf.XXXXXX")"
fi

PYTHON_BIN="$(resolve_python)"
ABS_BASELINE="$(cd "$(dirname "$BASELINE_PATH")" && pwd)/$(basename "$BASELINE_PATH")"
ABS_OUTPUT="$(cd "$(dirname "$OUTPUT_PATH")" && pwd)/$(basename "$OUTPUT_PATH")"

echo "[perf] running benchmark (backend=$BACKEND, size=$SIZE, iterations=$ITERATIONS)"
PYTHONPATH="$ROOT:$ROOT/rfx/python:${PYTHONPATH:-}" "$PYTHON_BIN" -m rfxJIT.runtime.benchmark \
  --size "$SIZE" \
  --iterations "$ITERATIONS" \
  --warmup "$WARMUP" \
  --seed "$SEED" \
  --backend "$BACKEND" \
  --json-out "$ABS_OUTPUT"

set +e
COMPARE_LOG="$("$PYTHON_BIN" - "$ABS_BASELINE" "$ABS_OUTPUT" "$THRESHOLD_PCT" <<'PY'
import json
import math
import sys
from pathlib import Path

baseline_path = Path(sys.argv[1])
current_path = Path(sys.argv[2])
threshold_pct = float(sys.argv[3])
threshold = threshold_pct / 100.0

with baseline_path.open("r", encoding="utf-8") as handle:
    baseline = json.load(handle)
with current_path.open("r", encoding="utf-8") as handle:
    current = json.load(handle)

metrics_to_compare = [
    "lowered_slowdown_x",
    "lowered_optimized_slowdown_x",
    "queue_slowdown_x",
]

baseline_metrics = baseline.get("metrics", {})
current_metrics = current.get("metrics", {})

regressions = []
for metric_name in metrics_to_compare:
    if metric_name not in baseline_metrics or metric_name not in current_metrics:
        continue
    baseline_value = float(baseline_metrics[metric_name])
    current_value = float(current_metrics[metric_name])

    if not math.isfinite(baseline_value) or baseline_value <= 0:
        continue
    if not math.isfinite(current_value):
        regressions.append((metric_name, baseline_value, current_value, math.inf))
        continue

    delta = (current_value - baseline_value) / baseline_value
    if delta > threshold:
        regressions.append((metric_name, baseline_value, current_value, delta))

if regressions:
    print(
        f"::warning title=rfxJIT perf regression::Detected {len(regressions)} "
        f"regression(s) above {threshold_pct:.2f}%"
    )
    for metric_name, baseline_value, current_value, delta in regressions:
        if math.isfinite(delta):
            pct = delta * 100.0
            detail = f"+{pct:.2f}%"
        else:
            detail = "non-finite"
        print(
            f"[perf] regression {metric_name}: baseline={baseline_value:.4f}, "
            f"current={current_value:.4f}, delta={detail}"
        )
    sys.exit(2)

print(f"[perf] no regressions above {threshold_pct:.2f}%")
sys.exit(0)
PY
)"
COMPARE_STATUS=$?
set -e

echo "$COMPARE_LOG"
echo "[perf] baseline=$ABS_BASELINE"
echo "[perf] current=$ABS_OUTPUT"

if [[ "$COMPARE_STATUS" -eq 2 ]]; then
  if [[ "$FAIL_ON_REGRESSION" -eq 1 ]]; then
    echo "[perf] failing due to --fail-on-regression" >&2
    exit 3
  fi
  echo "[perf] soft warning only; continuing"
  exit 0
fi

if [[ "$COMPARE_STATUS" -ne 0 ]]; then
  echo "[perf] comparison failed" >&2
  exit "$COMPARE_STATUS"
fi
