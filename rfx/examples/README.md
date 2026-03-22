# rfx Examples

Canonical scripts demonstrating the core rfx user workflows.

## Scripts

- **`so101_quickstart.py`** -- Connect to a real SO-101 and run a gentle movement smoke test
- **`deploy_real.py`** -- Deploy a policy to real SO-101 hardware (or mock mode)
- **`universal_go2.py`** -- Command-level Go2 control across `mock`/`genesis`/`mjx`/`real`
- **`genesis_viewer.py`** -- Visual inspection and debugging in Genesis with a live viewer
- **`train_vla.py`** -- End-to-end training loop in simulation with parallel environments
- **`teleop_record.py`** -- Teleoperation data collection with optional LeRobot/MCAP export

## Running

```bash
uv run rfx/examples/<script>.py
```

## TestPyPI Quickstart

```bash
uv venv .venv
source .venv/bin/activate
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
  rfx-sdk rfx-sdk-sim rfx-sdk-go2 torch
python rfx/examples/universal_go2.py --backend mock
```

Notes:

- Canonical examples now focus on deploy, Go2, Genesis, train, and teleop workflows.
- Most examples run without repo-local `rfx/configs/*.yaml` by using built-in SDK defaults.
- Go2 Genesis examples still require user-provided Go2 URDF/mesh assets at `rfx/assets/robots/go2/urdf/`.
