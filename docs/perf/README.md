# Performance Workflow

`rfxJIT` perf tracking uses one benchmark schema and local regression gates.

## Baselines

- CI baseline: `docs/perf/baselines/rfxjit_microkernels_cpu.json`
- Optional GPU baselines:
  - `docs/perf/baselines/rfxjit_microkernels_cuda.json`
  - `docs/perf/baselines/rfxjit_microkernels_metal.json`
- Initial target kernel: affine+relu phase0/2 microkernel benchmark (`rfxJIT/runtime/benchmark.py`)

## Run Locally

Generate a structured benchmark report:

```bash
python -m rfxJIT.runtime.benchmark \
  --backend cpu \
  --size 65536 \
  --iterations 200 \
  --warmup 10 \
  --json-out /tmp/rfxjit-current.json
```

Compare against baseline with soft warnings (default):

```bash
bash scripts/perf-check.sh \
  --baseline docs/perf/baselines/rfxjit_microkernels_cpu.json \
  --output /tmp/rfxjit-current.json \
  --backend cpu \
  --threshold-pct 10
```

Fail on regression (strict mode):

```bash
bash scripts/perf-check.sh \
  --baseline docs/perf/baselines/rfxjit_microkernels_cpu.json \
  --backend cpu \
  --threshold-pct 10 \
  --fail-on-regression
```

## Hook-Driven Guard (Default)

`pre-push` runs a local perf gate:

- always checks `cpu` and blocks push on regressions over threshold (default `10%`)
- checks `cuda` and `metal` when available on your machine
- GPU backend regressions block push when the check runs successfully
- transient GPU backend runtime failures are warning-only by default
- stores local per-machine baselines in `.rfx/perf-baselines/`
- bootstraps missing local baselines automatically on first run

Manual run:

```bash
bash scripts/perf-gate.sh
```

Tune locally with env vars:

```bash
RFX_PERF_ITERATIONS=100 RFX_PERF_THRESHOLD_PCT=12 bash scripts/perf-gate.sh
```

Force GPU backend failures to block push:

```bash
RFX_PERF_STRICT_GPU=1 bash scripts/perf-gate.sh
```

## Baseline Refresh Policy

Refresh CPU baseline when any of these change:

- kernel IR/lowering semantics
- runtime execution strategy for `cpu`
- benchmark shape/iteration defaults

Suggested update flow:

1. Run benchmark on a quiet machine.
2. Refresh baseline file(s):
   - CI/repo baseline: `bash scripts/perf-baseline.sh --backend cpu --output-dir docs/perf/baselines`
   - local hook baseline(s): `bash scripts/perf-baseline.sh --backend all --output-dir .rfx/perf-baselines`
3. Commit with message like `perf: refresh cpu microkernel baseline`.
