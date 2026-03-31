# Python SDK

## Design Goals

- One simple Python API for AI robotics workloads.
- Three methods per robot: `observe()`, `act()`, `reset()`.
- Simulation and collection are first-class SDK surfaces.
- Every saved model is self-describing: load and deploy with zero context.
- HuggingFace Hub native: push and pull policies like datasets.

## Install

```bash
uv pip install rfx-sdk
```

Or add to your project:

```bash
uv add rfx-sdk
```

## Quick Start

```python
import rfx

# Sim and mock use the same robot contract as real hardware.
robot = rfx.MockRobot(state_dim=12, action_dim=6)
obs = robot.observe()

# Deploy a policy
rfx.deploy("runs/my-policy", robot="so101")
```

## No hardware? No problem

You can explore the entire rfx workflow on your laptop — no robot required.

```python
# Mock robot: no sim engine deps, spring-damper physics
robot = rfx.MockRobot(state_dim=12, action_dim=6)
obs = robot.observe()
robot.act(torch.zeros(1, 64))

# Simulation: GPU-accelerated with viewer
robot = rfx.SimRobot.from_config("so101.yaml", backend="genesis", viewer=True)

# Batched RL training: thousands of parallel envs
robot = rfx.SimRobot.from_config("go2.yaml", backend="mjx", num_envs=4096)
```

Test the full deploy pipeline without hardware:

```bash
rfx deploy runs/my-policy --robot so101 --mock
```

Write a policy, save it, deploy with `--mock`, iterate. Switch to real hardware when you're ready.

## The Robot Interface

Every robot -- simulated or real -- implements three methods:

```python
obs = robot.observe()    # {"state": Tensor(1, 64), "images": ...}
robot.act(action)        # Tensor(1, 64)
robot.reset()
```

Built-in robots:

```python
robot = rfx.SimRobot.from_config("so101.yaml", backend="genesis", viewer=True)
robot = rfx.MockRobot(state_dim=12, action_dim=6)   # no sim engine deps, for testing
robot = rfx.RealRobot(rfx.SO101_CONFIG)              # real hardware (SO-101)
robot = rfx.RealRobot(rfx.GO2_CONFIG)                # real hardware (Go2)
robot = rfx.RealRobot(rfx.INNATE_CONFIG)             # real hardware (Innate, Zenoh-native)
```

## Collection

Collection is a primitive in the SDK:

```python
dataset = rfx.collection.collect(
    "so101",
    "my-org/demos",
    episodes=2,
    duration_s=10,
)
```

The collection API records observations through the same robot interface used by deployment and simulation.

## Write a Policy

A policy is any callable `Dict[str, Tensor] -> Tensor`. Use `@rfx.policy` to make it deployable from the CLI:

```python
# my_policy.py
import torch
import rfx

@rfx.policy
def hold_position(obs):
    return torch.zeros(1, 64)  # hold still
```

```bash
uv run rfx deploy my_policy.py --robot so101
```

For named joint control instead of raw tensor indices:

```python
@rfx.policy
def grasp(obs):
    return rfx.MotorCommands(
        {"gripper": 0.8, "wrist_pitch": -0.2},
        config=rfx.SO101_CONFIG,
    ).to_tensor()
```

`MotorCommands` resolves joint names from any `RobotConfig`'s joint list. No hardcoded indices.

## Model Management

Policies are saved as self-describing directories. Every saved model bundles weights, architecture, robot config, and normalizer state.

### Save

```python
policy.save("runs/go2-walk-v1",
    robot_config=config,
    normalizer=normalizer,
    training_info={"total_steps": 50000})
```

Creates:

```
runs/go2-walk-v1/
  rfx_config.json       # architecture + robot + training metadata
  model.safetensors     # weights
  normalizer.json       # observation normalizer state
```

### Load

```python
loaded = rfx.load_policy("runs/go2-walk-v1")
loaded = rfx.load_policy("hf://rfx-community/go2-walk-v1")

loaded.policy           # reconstructed policy
loaded.robot_config     # RobotConfig or None
loaded.normalizer       # normalizer or None
loaded.policy_type      # "MLP", "ActorCritic", etc.
```

`LoadedPolicy` is callable and handles torch/tinygrad conversion automatically.

### Inspect

```python
config = rfx.inspect_policy("runs/go2-walk-v1")
print(config["policy_type"])     # "MLP"
print(config["policy_config"])   # {"obs_dim": 48, ...}
```

### Share via HuggingFace Hub

```python
rfx.push_policy("runs/go2-walk-v1", "rfx-community/go2-walk-v1")
```

## Deploy API

`rfx.deploy()` is the main entry point. It handles everything: load policy, resolve robot config, create robot, run the control loop.

```python
stats = rfx.deploy(
    "runs/my-policy",      # path, hf:// URL, or .py file
    robot="so101",          # robot type or YAML path
    config=None,            # YAML config path (overrides robot)
    port="/dev/ttyACM0",    # optional hardware override
    rate_hz=50,             # control frequency
    duration=30,            # seconds (None = infinite)
    mock=False,             # use MockRobot instead
    device="cpu",           # torch device
    warmup_s=0.5,           # seconds to sleep after reset
    verbose=True,           # print status during deployment
)

# stats contains timing info
print(stats.iterations, stats.overruns)
print(stats.p50_jitter_s, stats.p95_jitter_s)
```

## MotorCommands

Build actions from named joints instead of raw tensor indices:

```python
cmd = rfx.MotorCommands(
    {"gripper": 0.8, "elbow": -0.3},
    config=rfx.SO101_CONFIG,
)

action = cmd.to_tensor()          # shape: (1, 64)
positions = cmd.to_list()         # flat list, length = action_dim

# With batch size
action = cmd.to_tensor(batch_size=4)  # shape: (4, 64)

# Factory method
cmd = rfx.MotorCommands.from_positions(
    {"elbow": 1.0},
    config=rfx.SO101_CONFIG,
    kp=30.0,
)
```

Works with any robot config -- joint names are resolved from `config.joints`.

## Built-in Robot Configs

```python
rfx.SO101_CONFIG    # 6 DOF arm, 50 Hz, joints: shoulder_pan, shoulder_lift, elbow, wrist_pitch, wrist_roll, gripper
rfx.GO2_CONFIG      # 12 DOF quadruped, 200 Hz
rfx.G1_CONFIG       # 29 DOF humanoid, 50 Hz
rfx.INNATE_CONFIG   # 6 DOF Innate bot, 50 Hz, Zenoh-native (no Rust RobotNode)
```

## Package Layout

```
rfx/python/rfx/
├── robot/          # Robot protocol, config, URDF
├── collection/     # Collection and dataset contracts
├── real/           # Real hardware backends (SO-101, Go2, G1, Innate)
├── sim/            # Simulation backends (MuJoCo, Genesis, mock)
├── teleop/         # Teleoperation sessions, transport, recording
├── runtime/        # Lifecycle, CLI, health, runtime helpers
├── workflow/       # Quality gates, stage registry
├── utils/          # Padding, transforms, normalizers
├── nn.py           # Policy definitions (MLP, ActorCritic, save/load)
├── hub.py          # Model save/load/push (HuggingFace Hub)
├── session.py      # Rate-controlled control loop
├── deploy.py       # rfx.deploy() implementation
├── decorators.py   # @policy, MotorCommands
├── node.py         # Zenoh transport factory & discovery
└── observation.py  # Observation spec & padding
```
