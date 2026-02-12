# AGENTS.md

## Monorepo Task Runner

Use Moon as the top-level task runner for this Rust + Python monorepo.

- Source setup script: `scripts/setup-from-source.sh`
- Workspace: `.moon/workspace.yml`
- Rust project config: `rfx/crates/rfx-core/moon.yml`
- Rust bindings config: `rfx/crates/rfx-python/moon.yml`
- Python project config: `rfx/python/moon.yml`

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
- `rfx/python/`
- `rfx/tests/`
- `rfxJIT/`

For full-package mypy, use `scripts/python-checks.sh typecheck-full` (or set `RFX_TYPECHECK_FULL=1` with `ci`).

## Git Quality Gates

Repo-managed hooks are configured via `.githooks`:

- `pre-commit`: Rust format check + Ruff on staged Python files
- `pre-push`: Rust fmt/clippy/test + Python lint/typecheck subset/test
- Optional local main-push block: set `RFX_BLOCK_MAIN_PUSH=1`.

Enable hooks once per clone:

```bash
./scripts/setup-git-hooks.sh
```

Recommended first-time local setup:

```bash
bash scripts/setup-from-source.sh
./scripts/setup-git-hooks.sh
```

## Repo Layout

- `rfx/`: core source tree (Rust crates, Python package, tests, configs, examples)
- `docs/`: project documentation and architecture notes
- `docs/workflow.md`: OSS workflow and PR hygiene
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
