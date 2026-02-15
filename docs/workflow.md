# OSS Workflow

This repo follows a tinygrad-style flow: fast local iteration on `main`,
atomic commits, direct pushes to `main`, and strong local/CI gates.

## Default Loop (No Branch Churn)

1. Sync `main`:

```bash
git switch main
git pull --ff-only origin main
```

2. Setup once per machine:

```bash
bash scripts/setup-from-source.sh
```

3. Build and check while coding:

```bash
moon run :format
moon run :lint
moon run :test
```

4. Commit atomically with short messages:

```bash
git add -p
git commit -m "jit: add kernel ir nodes"
```

5. Push:

```bash
git push origin main
```

## When to Use a Branch + PR

Branch + PR is optional and used only when it lowers risk:

- large refactors
- breaking API changes
- partial spikes you may discard

```bash
git switch -c feat/<short-name>
git push -u origin feat/<short-name>
```

## Quality Gates

- Local hooks are installed via `pre-commit` and run at `pre-commit` and `pre-push`.
- CI is a single workflow at `.github/workflows/ci.yml`.
- CI runs on `main` and pull requests.
- `pre-push` includes local `rfxJIT` perf regression checks (`cpu` enforced, `cuda`/`metal` attempted when available).
- CI includes a warning-only `rfxJIT` CPU perf check for visibility.
- Local perf baselines are kept in `.rfx/perf-baselines/`.
- Optional local main-push block:

```bash
export RFX_BLOCK_MAIN_PUSH=1
```

Install hooks manually (if you did not use setup script):

```bash
./.venv/bin/pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
```

## Perf Regression Check

Run the same CPU perf comparison used by CI:

```bash
bash scripts/perf-check.sh \
  --baseline docs/perf/baselines/rfxjit_microkernels_cpu.json \
  --backend cpu \
  --threshold-pct 10
```

Run the exact pre-push perf gate:

```bash
bash scripts/perf-gate.sh
```
