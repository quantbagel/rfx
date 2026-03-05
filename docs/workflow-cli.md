# CLI Reference

The supported public CLI surface is small:

- `rfx record`
- `rfx deploy`
- `rfx doctor`

These commands sit on top of the middleware primitives. In `rfx`, simulation and collection are part of the framework contract, not optional side workflows.

There are additional workflow utilities such as `rfx train` and `rfx runs`, but they are secondary to the core SDK/runtime loop.

All commands work with `uv run`:

```bash
uv run rfx record --robot so101 --repo-id demos --episodes 10
uv run rfx deploy runs/my-policy --robot so101
uv run rfx doctor
```

Or directly if rfx is installed:

```bash
rfx record ...
rfx deploy ...
rfx doctor
```

## `rfx record`

Collect robot observations into a LeRobot dataset.

```bash
rfx record --robot so101 --repo-id my-org/demos --episodes 10 --duration 30
rfx record --robot so101 --repo-id demos --duration 15 --fps 30 --port /dev/ttyACM0
rfx record --robot so101 --repo-id demos --mock --duration 5
rfx record --robot go2 --repo-id demos --duration 20 --push
```

| Flag | Default | Description |
|------|---------|-------------|
| `--robot` | required | Robot type (`so101`, `go2`, `g1`, `innate`) |
| `--repo-id` | required | Dataset name or HuggingFace repo ID |
| `--episodes` | `1` | Number of episodes to record |
| `--duration` | interactive | Episode duration in seconds. Omit to stop with Ctrl+C. |
| `--fps` | `30` | Recording frame rate |
| `--rate-hz` | `fps` | Sampling rate override for `robot.observe()` |
| `--config` | none | Robot YAML config override |
| `--port` | auto | Serial port or IP override |
| `--camera-id` | config | Camera device id override (repeatable) |
| `--mock` | `false` | Use `MockRobot` for dry-run recording |
| `--push` | `false` | Push dataset to HuggingFace Hub after recording |
| `--mcap` | `false` | Write MCAP sidecar alongside dataset |
| `--state-dim` | config | Override state dimension |

## `rfx deploy`

Load a trained policy and run it on hardware.

```bash
# From a saved checkpoint
rfx deploy runs/my-policy --robot so101

# From HuggingFace Hub
rfx deploy hf://rfx-community/go2-walk-v1 --robot go2

# From a Python file with @rfx.policy
rfx deploy my_policy.py --robot so101

# Test without hardware
rfx deploy runs/my-policy --robot so101 --mock

# With options
rfx deploy runs/my-policy --robot so101 --duration 60 --rate-hz 50 --port /dev/ttyACM0
```

| Flag | Default | Description |
|------|---------|-------------|
| `policy` | required | Path to policy dir, `hf://org/repo`, or `.py` file |
| `--robot` | auto | Robot type (`so101`, `go2`, `g1`, `innate`). Auto-detected from policy config if saved. |
| `--config` | none | Path to robot YAML config (overrides `--robot`) |
| `--port` | auto | Serial port or IP override |
| `--rate-hz` | auto | Control loop frequency (defaults to robot config) |
| `--duration` | infinite | Run time in seconds. Ctrl+C to stop if not set. |
| `--mock` | `false` | Use MockRobot instead of real hardware |
| `--device` | `cpu` | Torch device for inference |
| `--warmup` | `0.5` | Seconds to sleep after reset before starting |

Deploy handles: load weights, resolve robot config, connect hardware, run control loop with rate control and jitter tracking, clean shutdown on Ctrl+C.

## `rfx doctor`

Check your environment.

```bash
rfx doctor
rfx doctor --strict
```

Checks: Python version, cargo, uv, core imports, Rust extension, simulation backends, rfxJIT backends, serial ports.

Use `--strict` to make missing required dependencies fail with a non-zero exit code.

## Python API

Everything the CLI does is available from Python:

```python
import rfx

# Record
rfx.collection.collect("so101", "my-org/demos", episodes=10)

# Deploy
stats = rfx.deploy("runs/my-policy", robot="so101")
stats = rfx.deploy("hf://user/policy", robot="go2", duration=30)
stats = rfx.deploy("my_policy.py", robot="so101")
```


## Secondary Commands

`rfx train` and `rfx runs` remain available for lightweight workflow metadata and artifact lineage.

## Run Registry

Browse saved training runs:

```bash
rfx runs list --limit 20
rfx runs show <run_id>
```
