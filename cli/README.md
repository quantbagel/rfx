# CLI

Shell helper for common rfx tasks. All commands run via `cli/rfx.sh <command>`.

## Environment & diagnostics

| Command | Description |
|---------|-------------|
| `doctor` | Environment and dependency checks |
| `doctor-teleop` | Teleop-specific environment checks |
| `doctor-so101` | SO101-specific environment and port checks |

## SO-101 hardware

| Command | Description |
|---------|-------------|
| `so101-demo [--port ...]` | Run a minimal SO101 hardware quickstart |
| `so101-bimanual` | Bimanual SO101 teleop using `rfx/configs/so101_bimanual.yaml` |
| `so101-setup` | Motor ID and baud rate setup wizard |

## Workflow (wraps `rfx` Python CLI)

| Command | Description |
|---------|-------------|
| `record [--robot ...] [--repo-id ...]` | Collect robot observations into a dataset |
| `train [...]` | Train a policy |
| `deploy <policy> [--robot ...]` | Deploy a trained policy to hardware |
| `runs [...]` | Browse saved training runs |

## Platform

| Command | Description |
|---------|-------------|
| `register --url <dashboard> --api-key <key>` | Register robot with the platform |
| `probe [--regions ...]` | Measure candidate AWS region latency |
| `connect --url <dashboard> --api-key <key>` | Keep a registered robot online with heartbeats |

## Setup

| Command | Description |
|---------|-------------|
| `bootstrap` | Local project setup |
| `bootstrap-teleop` | Teleop-specific bootstrap |
| `setup-source` | Run `scripts/setup-from-source.sh` |
| `check` | Run monorepo quality checks |
