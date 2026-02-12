# rfxJIT

Workstream for JIT-related systems in rfx:

- kernel lowering ideas
- runtime execution plans
- performance experiments

## Current Focus

- define a small, explicit IR for kernels
- prototype scheduling + fusion passes
- benchmark generated kernels against baseline paths

## Layout

- `notes/`: architecture notes and design records
- `runtime/`: runtime execution plumbing prototypes
- `kernels/`: kernel generation and lowering experiments
- `ROADMAP.md`: milestone plan

## Phase 0 Status

- typed elementwise kernel IR: `rfxJIT/kernels/ir.py`
- reference interpreter: `rfxJIT/runtime/interpreter.py`
- benchmark harness: `rfxJIT/runtime/benchmark.py`
- tests: `rfxJIT/tests/test_ir.py`

Run the benchmark:

```bash
python -m rfxJIT.runtime.benchmark --size 65536 --iterations 200
```
