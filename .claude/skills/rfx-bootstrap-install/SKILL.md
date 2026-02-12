---
name: rfx-bootstrap-install
description: Bootstrap the rfx monorepo for local development. Use when asked to install dependencies, enable hooks, or prepare this repo for an agent to work safely.
---

# rfx Bootstrap Install

Use this skill when a user asks to:

- install dependencies for this repo
- set up local dev tooling
- enable repo quality gates before coding

## Workflow

1. Run the bootstrap script:

```bash
bash .claude/skills/rfx-bootstrap-install/scripts/bootstrap.sh
```

2. If the user asks for verification, run:

```bash
cargo fmt --all -- --check
scripts/python-checks.sh lint
```

## Notes

- The script enables repo-managed git hooks (`.githooks`).
- Python checks run against `rfx/python` and `rfx/tests`.
- If `uv` is not installed, the script falls back to `python -m pip`.
