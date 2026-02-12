# rfxJIT Roadmap

## Phase 0: Foundations

- Define IR surface for simple elementwise kernels.
- Document constraints (tensor shapes, dtypes, memory model).
- Add benchmark harness for baseline vs JIT candidate.

## Phase 1: Lowering + Runtime

- IR to executable kernel lowering prototype.
- Runtime queue for kernel dispatch.
- Validation tests on deterministic kernels.

## Phase 2: Optimization

- Simple fusion pass (elementwise chains).
- Constant folding and dead op elimination.
- Cost model for launch vs fusion decisions.

## Phase 3: Integration

- Wire JIT path into selected `rfx` execution points behind feature flags.
- Add profiling and regression tracking in CI/perf jobs.
- Promote stable components from `rfxJIT` into core modules.
