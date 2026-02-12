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
