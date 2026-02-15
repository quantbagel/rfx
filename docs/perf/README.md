# Performance Workflow

`rfxJIT` perf tracking uses one benchmark schema and one soft regression gate.

## Baselines

- CPU baseline: `docs/perf/baselines/rfxjit_microkernels_cpu.json`
- Initial target: affine+relu phase0/2 microkernel benchmark (`rfxJIT/runtime/benchmark.py`)

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

Fail on regression (for strict local checks):

```bash
bash scripts/perf-check.sh \
  --baseline docs/perf/baselines/rfxjit_microkernels_cpu.json \
  --backend cpu \
  --threshold-pct 10 \
  --fail-on-regression
```

## Baseline Refresh Policy

Refresh CPU baseline when any of these change:

- kernel IR/lowering semantics
- runtime execution strategy for `cpu`
- benchmark shape/iteration defaults

Suggested update flow:

1. Run benchmark on a quiet machine.
2. Replace `docs/perf/baselines/rfxjit_microkernels_cpu.json`.
3. Commit with message like `perf: refresh cpu microkernel baseline`.
