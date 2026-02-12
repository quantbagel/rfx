# rfxJIT

Performance engine for JIT-related systems in rfx:

- kernel lowering and scheduling
- runtime execution paths
- optimization and profiling passes

rfxJIT is inspired by PyTorch (ergonomics), JAX (functional transforms and
IR-based AD), and TVM (scheduling and codegen), while staying intentionally
tiny and hackable.

## How It Compares

### PyTorch

- Similar: eager Tensor API, autograd, optim, and basic datasets/layers.
- Similar: familiar training loops.
- Different: the full compiler and IR are visible and easy to modify.

### JAX

- Similar: IR-based autodiff over primitives.
- Similar: function-level JIT via TinyJIT-style capture and replay.
- Different: fewer transforms today (for example, no full `vmap`/`pmap` yet),
  but much smaller and easier to read.

### TVM

- Similar: multiple lowering passes, scheduling, and beam-search-style kernel
  exploration.
- Similar: device graph style batched execution.
- Different: rfxJIT is coupled with a front-end framework (`rfx`) rather than
  only being a compiler stack.

## Current Focus

- define a small, explicit IR for kernels
- prototype scheduling + fusion passes
- benchmark generated kernels against baseline paths

## Layout

- `notes/`: architecture notes and design records
- `runtime/`: runtime execution components
- `kernels/`: kernel IR, transforms, and lowering logic
- `ROADMAP.md`: milestone plan

## Phase 0 Status

- typed elementwise kernel IR: `rfxJIT/kernels/ir.py`
- reference interpreter: `rfxJIT/runtime/interpreter.py`
- benchmark harness: `rfxJIT/runtime/benchmark.py`
- tests: `rfxJIT/tests/test_ir.py`

## Phase 1 Status

- IR lowering to slot-based executable form: `rfxJIT/kernels/lowering.py`
- lowered-kernel executor: `rfxJIT/runtime/executor.py`
- single-worker dispatch queue: `rfxJIT/runtime/queue.py`
- lowering/queue tests: `rfxJIT/tests/test_lowering_queue.py`

## Phase 2 Status

- optimization passes: `rfxJIT/kernels/optimize.py`
- constant folding + dead-op elimination + simple chain fusion
- optimization tests: `rfxJIT/tests/test_optimize.py`
- benchmark now reports op count before/after optimization

## Transform Status

- tracer API for Python expression capture: `rfxJIT/kernels/trace.py`
- tinyJIT-style cache+replay runtime: `rfxJIT/runtime/tinyjit.py`
- IR autodiff + functional transforms: `grad`, `value_and_grad`
- transform tests: `rfxJIT/tests/test_tinyjit.py`, `rfxJIT/tests/test_grad_transforms.py`

Run the benchmark:

```bash
python -m rfxJIT.runtime.benchmark --size 65536 --iterations 200
```
