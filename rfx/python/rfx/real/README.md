# rfx.real

Real hardware backends for rfx. Each backend implements the Robot protocol for physical robots.

## Modules

- **`base.py`** -- RealRobot: auto-detects robot type from config and wraps the appropriate backend
- **`so101.py`** -- So101Backend: 6-DOF arm via USB serial using the Rust driver at 1 Mbaud. Includes So101LeaderFollower for teleoperation pairing.
- **`go2.py`** -- Go2Backend: 12-DOF Unitree Go2 quadruped via Zenoh transport (Rust RobotNode). Legacy unitree_sdk2py/subprocess backends kept for backwards compat. Provides stand/sit/walk commands.
- **`innate.py`** -- InnateBackend: Zenoh-native backend for Innate bots. The bot publishes its own state — no Rust RobotNode needed. Supports JSON and CDR (ROS 2 JointState) message formats.
- **`camera.py`** -- Camera (OpenCV) and RealSenseCamera (Intel RealSense depth camera)
