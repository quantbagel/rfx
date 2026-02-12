# rfx

## Project North Star

- `rfxJIT` is the ultra-performance kernel engine for the project.
- `rfx` remains the product surface, while stable JIT/compiler pieces are promoted into it.
- Long term, execution should be `rfxJIT`-first, reducing reliance on Python-to-Rust bindings.
- Default collaboration flow is atomic commits pushed directly to `main`.

`rfxJIT` runtime path is feature-flagged:

```bash
export RFX_JIT=1
```

With `RFX_JIT=1`, `@rfx.policy(jit=True)` can route NumPy policy calls through
`rfxJIT` while preserving existing fallback behavior.

## Install From Source (Recommended)

Prerequisites:
- `uv`
- `cargo`/Rust toolchain
- `git`

Clone + setup:

```bash
git clone https://github.com/quantbagel/rfx.git
cd rfx
bash scripts/setup-from-source.sh
```

Manual setup equivalent:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
uv pip install --python .venv/bin/python -e .
cargo fetch
./.venv/bin/pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
```

Direct path install (from another repo or workspace):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e /absolute/path/to/rfx
```

Git URL install (without local source checkout):

```bash
uv pip install git+https://github.com/quantbagel/rfx.git
```

`Direct path` and `Git URL` installs are best for consumption, not contributor setup.
Use source install when you want local hooks, CI-parity checks, and editable development.

## Monorepo Layout

- `rfx/`: source tree (`crates/`, `python/`, `tests/`, `configs/`, `examples/`)
- `docs/`: documentation
- `docs/workflow.md`: contributor and OSS workflow
- `rfxJIT/`: ultra-performance kernel and compiler engine
- `cli/`: command-line tooling
- `.claude/skills/rfx-bootstrap-install/`: Claude skill for agent bootstrap

## Claude Skill Bootstrap

To bootstrap this repo with an agent-oriented workflow:

```bash
bash .claude/skills/rfx-bootstrap-install/scripts/bootstrap.sh
```

CLI shortcut:

```bash
./cli/rfx.sh bootstrap
```

Direct source setup shortcut:

```bash
./cli/rfx.sh setup-source
```

## Local Quality Gates (pre-commit + pre-push)

Install hooks once per clone:

```bash
bash scripts/setup-git-hooks.sh
```

`scripts/setup-from-source.sh` already runs this step.

Dev dependencies are defined in `requirements-dev.txt` for local source installs.

What runs:
- `pre-commit`: `cargo fmt --all -- --check`, then Ruff on staged Python files.
- `pre-push`: Rust fmt/clippy/tests plus Python Ruff/mypy subset/pytest.
- Optionally block local direct pushes from `main` by setting `RFX_BLOCK_MAIN_PUSH=1`.

Run all local gates manually:

```bash
./.venv/bin/pre-commit run --all-files --hook-stage pre-push
```

## CI

- Single CI workflow: `.github/workflows/ci.yml`
- CI runs on pushes to `main` and on pull requests.

## Moon (Monorepo Task Runner)

This repo now has a Moon workspace:
- Workspace config: `.moon/workspace.yml`
- Project configs: `rfx/crates/rfx-core/moon.yml`, `rfx/crates/rfx-python/moon.yml`, `rfx/python/moon.yml`

Typical commands:
- `moon run :format`
- `moon run :lint`
- `moon run :typecheck`
- `moon run :test`
- `moon run :build`

Python Moon tasks are backed by `scripts/python-checks.sh`.
Use `scripts/python-checks.sh typecheck-full` (or `RFX_TYPECHECK_FULL=1 scripts/python-checks.sh ci`) for full-package mypy.
