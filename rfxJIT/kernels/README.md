# Kernel Prototypes

Kernel IR and lowering experiments live here.

Scope examples:
- IR node definitions
- lowering rules
- generated kernel validation fixtures

Current files:
- `ir.py`: typed phase 0 kernel IR
- `lowering.py`: phase 1 lowering into slot-based instructions
- `optimize.py`: phase 2 optimization passes and redundant-kernel fixture
