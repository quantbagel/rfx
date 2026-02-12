# rfxJIT Roadmap

## Phase 0: Foundations

- Define IR surface for simple elementwise kernels.
- Document constraints (tensor shapes, dtypes, memory model).
- Add benchmark harness for baseline vs JIT candidate.

## Phase 1: Lowering + Runtime

- IR to executable kernel lowering prototype. (initial version landed)
- Runtime queue for kernel dispatch. (single-worker version landed)
- Validation tests on deterministic kernels. (lowering + queue tests landed)

## Phase 2: Optimization

- Simple fusion pass (elementwise chains). (initial version landed)
- Constant folding and dead op elimination. (landed)
- Cost model for launch vs fusion decisions.

## Phase 3: Integration

- Wire JIT path into selected `rfx` execution points behind feature flags.
- Add profiling and regression tracking in CI/perf jobs.
- Promote stable components from `rfxJIT` into core modules.
