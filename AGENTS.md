# AGENTS.md

## Monorepo Task Runner

Use Moon as the top-level task runner for this Rust + Python monorepo.

- Workspace: `.moon/workspace.yml`
- Rust project config: `src/crates/rfx-core/moon.yml`
- Rust bindings config: `src/crates/rfx-python/moon.yml`
- Python project config: `src/python/moon.yml`

## Moon Commands

Run from repo root:

```bash
moon run :format
moon run :lint
moon run :typecheck
moon run :test
moon run :build
```

## Python Task Backend

Python Moon tasks use:

```bash
scripts/python-checks.sh <lint|typecheck|typecheck-full|test|build|ci>
```

This script resolves Python from `.venv` first, then `python3`/`python`.

It runs checks against:
- `src/python/`
- `src/tests/`

For full-package mypy, use `scripts/python-checks.sh typecheck-full` (or set `RFX_TYPECHECK_FULL=1` with `ci`).

## Git Quality Gates

Repo-managed hooks are configured via `.githooks`:

- `pre-commit`: Rust format check + Ruff on staged Python files
- `pre-push`: Rust fmt/clippy/test + Python lint/typecheck subset/test
- `pre-push` blocks direct pushes from `main` unless `RFX_ALLOW_MAIN_PUSH=1` is set.

Enable hooks once per clone:

```bash
./scripts/setup-git-hooks.sh
```

## Repo Layout

- `src/`: core source tree (Rust crates, Python package, tests, configs, examples)
- `docs/`: project documentation and architecture notes
- `rfxJIT/`: JIT-focused experiments and prototypes
- `cli/`: command-line tooling surface
- `.claude/skills/rfx-bootstrap-install/`: Claude skill for agent bootstrap/install

## Bootstrap Command

Use either:

```bash
bash .claude/skills/rfx-bootstrap-install/scripts/bootstrap.sh
```

or:

```bash
./cli/rfx.sh bootstrap
```
