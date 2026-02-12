# Runtime Prototypes

Execution orchestration experiments for JIT-generated kernels.

Scope examples:
- dispatch queues
- buffer lifetime tracking
- kernel launch interfaces

Current files:
- `interpreter.py`: phase 0 reference execution path
- `executor.py`: lowered phase 1 executor
- `queue.py`: dispatch worker queue for lowered kernels
- `benchmark.py`: baseline vs IR/lowered/optimized timing harness
- `tinyjit.py`: tinyJIT-style runtime and transforms (`grad`, `value_and_grad`)
