# rfx

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
./scripts/setup-git-hooks.sh
```

Manual setup equivalent:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
uv pip install --python .venv/bin/python -e .
cargo fetch
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

## Monorepo Layout

- `rfx/`: source tree (`crates/`, `python/`, `tests/`, `configs/`, `examples/`)
- `docs/`: documentation
- `docs/workflow.md`: contributor and OSS workflow
- `rfxJIT/`: JIT-related work area
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

## Git Hooks (Quality Gates)

Enable repo-managed hooks once per clone:

```bash
./scripts/setup-git-hooks.sh
```

Dev dependencies are defined in `requirements-dev.txt` for local source installs.

What runs:
- `pre-commit`: `cargo fmt --all -- --check`, then Ruff on staged Python files.
- `pre-push`: Rust fmt/clippy/tests plus Python Ruff/mypy subset/pytest.
- Optionally block local direct pushes from `main` by setting `RFX_BLOCK_MAIN_PUSH=1`.

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
