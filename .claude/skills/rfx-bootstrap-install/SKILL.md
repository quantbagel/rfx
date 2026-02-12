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

This delegates to `scripts/setup-from-source.sh` (uv venv + requirements + editable install + cargo fetch).

2. If the user asks for verification, run:

```bash
cargo fmt --all -- --check
scripts/python-checks.sh lint
```

## Notes

- The script enables repo-managed git hooks (`.githooks`).
- Source dependencies are tracked in `requirements.txt` and `requirements-dev.txt`.
- Python checks run against `rfx/python`, `rfx/tests`, and `rfxJIT`.
- `uv` is required for the source setup flow.
