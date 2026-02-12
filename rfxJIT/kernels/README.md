# Kernel Engine

Kernel IR, transforms, and lowering logic live here.

Scope examples:
- IR node definitions
- lowering rules
- generated kernel validation fixtures

Current files:
- `ir.py`: typed phase 0 kernel IR
- `lowering.py`: phase 1 lowering into slot-based instructions
- `optimize.py`: phase 2 optimization passes and redundant-kernel fixture
- `trace.py`: expression tracer to generate IR from Python tensor-style code
- `ad.py`: IR autodiff transforms to generate gradient kernels
