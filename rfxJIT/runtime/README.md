# Runtime Engine

Execution orchestration for JIT-generated kernels.

Scope examples:
- dispatch queues
- buffer lifetime tracking
- backend launch interfaces (`cpu`, `cuda`, `metal`)

Current files:
- `interpreter.py`: phase 0 reference execution path
- `executor.py`: lowered phase 1 executor
- `opcode.py`: stable opcode tape contract + serialization for lowered kernels
- `queue.py`: dispatch worker queue for lowered kernels
- `benchmark.py`: baseline vs IR/lowered/optimized timing harness by backend + JSON reports
- `tinyjit.py`: tinyJIT-style runtime and transforms (`grad`, `value_and_grad`)
