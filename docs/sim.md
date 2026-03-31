# Simulation

Three backends: Genesis (GPU), MJX (JAX-accelerated MuJoCo), and Mock (zero deps).

## Quickstart (Genesis)

```bash
uv pip install --python 3.13 torch
uv run --python 3.13 rfx/examples/genesis_viewer.py --auto-install
```

This starts the Genesis backend, opens the viewer, and auto-installs `genesis-world` if missing.

## Python API

```python
from rfx.sim import SimRobot

# Genesis with viewer
robot = SimRobot.from_config("rfx/configs/so101.yaml", backend="genesis", viewer=True, auto_install=True)
obs = robot.reset()

# MJX with batched envs
robot = SimRobot.from_config("rfx/configs/go2.yaml", backend="mjx", num_envs=4096)

# Mock (zero deps, for testing)
from rfx.sim import MockRobot
robot = MockRobot(state_dim=12, action_dim=6)
```

## Go2 URDF Integration

Go2 assets are not bundled. Place your own URDF + meshes under `rfx/assets/robots/go2/`:

```text
rfx/assets/robots/go2/
├── urdf/
│   ├── go2.urdf
│   └── meshes/...
└── mjcf/
    └── go2.xml
```

Then:

```bash
uv run --python 3.13 rfx/examples/universal_go2.py --backend genesis --auto-install
```

## Run modes

Headless (no viewer):

```python
robot = SimRobot.from_config("rfx/configs/so101.yaml", backend="genesis", viewer=False)
```

Runtime knobs:

```bash
uv run --python 3.13 rfx/examples/genesis_viewer.py \
  --num-envs 1 \
  --steps 2000 \
  --substeps 4
```

## Auto-install

If Genesis is not installed, rfx tries `uv pip install genesis-world`, falling back to `pip install genesis-world`.

Genesis requires CPython <= 3.13 (no 3.14 wheels yet).

Opt in globally:

```bash
export RFX_AUTO_INSTALL_GENESIS=1
```

## Troubleshooting

- Genesis install fails: `uv pip install genesis-world` manually.
- No viewer: check graphics support, retry with `--device cpu`, or run headless first.
