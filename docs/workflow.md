# OSS Workflow

This repo now follows a tinygrad-style flow: fast local iteration on `main`, atomic commits, and strong hooks/CI.

## Default Loop (No Branch Churn)

1. Sync `main`:

```bash
git switch main
git pull --ff-only origin main
```

2. Setup once per machine:

```bash
bash scripts/setup-from-source.sh
./scripts/setup-git-hooks.sh
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

Use a short-lived branch only for bigger/riskier work:

- large refactors
- breaking API changes
- experimental spikes you may discard

```bash
git switch -c feat/<short-name>
git push -u origin feat/<short-name>
```

## Quality Gates

- Local hooks (`pre-commit`, `pre-push`) prevent low-quality pushes.
- CI runs on `main` and pull requests.
- Optional local main-push block:

```bash
export RFX_BLOCK_MAIN_PUSH=1
```

Use GitHub branch protection if you want hard server-side blocking of direct pushes.
