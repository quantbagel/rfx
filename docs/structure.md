# Repository Structure

```
rfx/                    # Core source tree
├── crates/             # Rust crates (rfx-core, rfx-python bindings)
├── python/rfx/         # Python SDK
│   ├── robot/          # Robot protocol, config, URDF
│   ├── real/           # Hardware backends (SO-101, Go2, G1, Innate)
│   ├── sim/            # Simulation backends (Genesis, MJX, Mock)
│   ├── teleop/         # Teleoperation sessions, transport, recording
│   ├── collection/     # Dataset recording, hub integration
│   ├── runtime/        # CLI, node lifecycle, health, otel
│   ├── workflow/       # Quality gates, stage registry
│   └── utils/          # Padding, transforms, normalizers
├── tests/              # Python test suite
├── configs/            # Robot YAML configs (so101, go2, so101_bimanual)
├── assets/             # Robot URDF/MJCF assets
└── examples/           # Runnable examples

packages/               # Extension Python packages
├── rfx-sim/            # rfx-sdk-sim (simulation extras)
├── rfx-go2/            # rfx-sdk-go2 (Go2-specific extras)
└── rfx-lerobot/        # rfx-sdk-lerobot (LeRobot integration)

docs/                   # Documentation
cli/                    # Command-line tooling
```

Quality/tooling:

- Git hooks: `.githooks/`, `scripts/setup-git-hooks.sh`
- Moon workspace: `.moon/workspace.yml`
- Python checks: `scripts/python-checks.sh`
- Source setup: `scripts/setup-from-source.sh`
- CI: `.github/workflows/ci.yml`
