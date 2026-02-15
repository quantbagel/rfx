<div align="center">

[<img alt="rfx logo" src="docs/assets/logo.png" width="180" style="border-radius: 20px;" />](https://github.com/quantbagel/rfx)

rfx: For something between PyTorch and a robotics runtime stack.

<h3>

[Homepage](https://github.com/quantbagel/rfx) | [Documentation](https://deepwiki.com/quantbagel/rfx) | [Discord](https://discord.gg/xV8bAGM8WT)

</h3>

[![CI](https://github.com/quantbagel/rfx/actions/workflows/ci.yml/badge.svg)](https://github.com/quantbagel/rfx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/xV8bAGM8WT)

</div>

---

rfx is an end-to-end robotics stack:

- Python robotics API for simulation, real hardware, teleoperation, and policy execution
- `rfxJIT` IR/compiler/runtime that lowers and executes kernels across `cpu`/`cuda`/`metal`
- Rust core primitives and drivers exposed to Python through PyO3 bindings
- Local quality and perf gates designed to keep control-loop and kernel regressions visible

It is inspired by PyTorch (ergonomics), JAX (functional transforms and IR-based AD), and TVM (scheduling/codegen), while staying practical for robot workflows.

---

## How rfx compares

### PyTorch

- Similar: familiar Python-first APIs and policy/training workflows.
- Similar: easy integration with existing model code.
- Different: rfx ships robotics-native interfaces (`observe`/`act`/`reset`) and runtime hooks for deployment.

### JAX

- Similar: IR-centered runtime approach and JIT-style execution paths.
- Similar: transform-friendly kernel representation in `rfxJIT`.
- Different: rfx focuses on robotics control loops and deployment ergonomics over broad functional transform coverage.

### TVM

- Similar: lowering, optimization passes, and backend code generation.
- Similar: explicit execution/runtime separation.
- Different: rfx includes the full robotics-facing product surface, not only compiler infrastructure.

---

## Core interface

All robots in rfx implement the same three-method protocol:

```python
observation = robot.observe()
robot.act(action)
robot.reset()
```

This interface is consistent across simulation, real hardware, and teleoperation.

## Installation

The recommended install for contributors is from source.

### From source

```bash
git clone https://github.com/quantbagel/rfx.git
cd rfx
bash scripts/setup-from-source.sh
```

### Direct (GitHub)

```bash
uv pip install git+https://github.com/quantbagel/rfx.git
```

### Direct (local path)

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e /absolute/path/to/rfx
```

## Runtime switches (`rfxJIT`)

```bash
export RFX_JIT=1
export RFX_JIT_BACKEND=auto  # auto|cpu|cuda|metal
export RFX_JIT_STRICT=0      # 1 to raise if requested backend fails
```

With `RFX_JIT=1`, `@rfx.policy(jit=True)` can route NumPy policy calls through `rfxJIT` while preserving fallback behavior.

## Quality and performance checks

Run local pre-push checks:

```bash
./.venv/bin/pre-commit run --all-files --hook-stage pre-push
```

Run the CPU perf gate used in CI:

```bash
bash scripts/perf-check.sh \
  --baseline docs/perf/baselines/rfxjit_microkernels_cpu.json \
  --backend cpu \
  --threshold-pct 10
```

## Documentation

- Docs entrypoint: `docs/README.md`
- Contributor workflow: `docs/workflow.md`
- Performance workflow: `docs/perf/README.md`
- Contributing guide: `CONTRIBUTING.md`

## Community and support

- Issues: https://github.com/quantbagel/rfx/issues
- Discussions: https://github.com/quantbagel/rfx/discussions
- Pull requests: https://github.com/quantbagel/rfx/pulls
- Community expectations: `CODE_OF_CONDUCT.md`

## License

MIT. See `LICENSE`.
