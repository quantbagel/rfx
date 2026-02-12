# rfx

```bash
uv pip install git+https://github.com/quantbagel/rfx.git
```

## Monorepo Layout

- `src/`: source tree (`crates/`, `python/`, `tests/`, `configs/`, `examples/`)
- `docs/`: documentation
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

## Git Hooks (Quality Gates)

Enable repo-managed hooks once per clone:

```bash
./scripts/setup-git-hooks.sh
```

Install dev tools used by hooks:

```bash
uv pip install -e '.[dev]'
```

What runs:
- `pre-commit`: `cargo fmt --all -- --check`, then Ruff on staged Python files.
- `pre-push`: Rust fmt/clippy/tests plus Python Ruff/mypy subset/pytest.
- `pre-push` also blocks direct pushes from `main` (override with `RFX_ALLOW_MAIN_PUSH=1`).

## Moon (Monorepo Task Runner)

This repo now has a Moon workspace:
- Workspace config: `.moon/workspace.yml`
- Project configs: `src/crates/rfx-core/moon.yml`, `src/crates/rfx-python/moon.yml`, `src/python/moon.yml`

Typical commands:
- `moon run :format`
- `moon run :lint`
- `moon run :typecheck`
- `moon run :test`
- `moon run :build`

Python Moon tasks are backed by `scripts/python-checks.sh`.
Use `scripts/python-checks.sh typecheck-full` (or `RFX_TYPECHECK_FULL=1 scripts/python-checks.sh ci`) for full-package mypy.
