#!/usr/bin/env python3
"""Minimal SO-101 hardware smoke test.

Connects to the arm, reads the current joint positions, runs a gentle
motion test, then returns home on exit.

Examples:
    uv run --python 3.13 rfx/examples/so101_quickstart.py --port auto
    uv run --python 3.13 rfx/examples/so101_quickstart.py --port /dev/ttyACM0 --motion extend
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import rfx


EXTEND_DELTAS = {
    1: 0.95,   # shoulder_lift
    2: -1.05,  # elbow
    3: -0.45,  # wrist_pitch
}


def _resolve_robot(port: str):
    if port == "auto":
        return rfx.So101Robot()
    return rfx.So101Robot(port=port)


def _build_action(
    *,
    baseline: torch.Tensor,
    robot,
    phase: float,
    amplitude: float,
    joint_indices: list[int],
) -> torch.Tensor:
    action = torch.zeros((1, robot.max_action_dim), dtype=torch.float32)
    target = baseline.clone()
    offset = amplitude * math.sin(phase)
    for joint_idx in joint_indices:
        target[joint_idx] += offset
    action[0, : robot.action_dim] = target
    return action


def _joint_name(robot, joint_idx: int) -> str:
    if hasattr(robot, "config") and robot.config.joints and joint_idx < len(robot.config.joints):
        return robot.config.joints[joint_idx].name
    return f"joint_{joint_idx}"


def _joint_limits(robot, joint_idx: int) -> tuple[float, float]:
    if hasattr(robot, "config") and robot.config.joints and joint_idx < len(robot.config.joints):
        joint = robot.config.joints[joint_idx]
        return float(joint.position_min), float(joint.position_max)
    return -3.14159, 3.14159


def _clamp_target(robot, joint_idx: int, value: float, margin: float = 0.08) -> float:
    lower, upper = _joint_limits(robot, joint_idx)
    return float(min(max(value, lower + margin), upper - margin))


def _build_extend_target(*, baseline: torch.Tensor, robot, extension_scale: float) -> torch.Tensor:
    target = baseline.clone()
    for joint_idx, delta in EXTEND_DELTAS.items():
        if joint_idx >= robot.action_dim:
            continue
        target[joint_idx] = _clamp_target(
            robot,
            joint_idx,
            float(baseline[joint_idx]) + (delta * extension_scale),
        )
    return target


def _build_extend_action(
    *,
    baseline: torch.Tensor,
    target: torch.Tensor,
    robot,
    elapsed: float,
    duration: float,
) -> tuple[torch.Tensor, float]:
    progress = 0.5 - 0.5 * math.cos((elapsed / duration) * 2.0 * math.pi)
    blended = baseline + ((target - baseline) * progress)
    action = torch.zeros((1, robot.max_action_dim), dtype=torch.float32)
    action[0, : robot.action_dim] = blended
    return action, progress


def main() -> None:
    parser = argparse.ArgumentParser(description="Gentle SO-101 movement smoke test")
    parser.add_argument("--port", default="auto", help="Serial port or 'auto'")
    parser.add_argument("--duration", type=float, default=8.0, help="Motion time in seconds")
    parser.add_argument("--rate", type=float, default=25.0, help="Control loop frequency in Hz")
    parser.add_argument(
        "--motion",
        choices=["wiggle", "extend"],
        default="extend",
        help="Motion preset: local wiggle or visible extend/retract cycle",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.15,
        help="Joint offset amplitude in radians",
    )
    parser.add_argument(
        "--cycles",
        type=float,
        default=2.0,
        help="How many sine cycles to run over the motion duration",
    )
    parser.add_argument(
        "--joints",
        default="0,1",
        help="Comma-separated joint indices to move gently",
    )
    parser.add_argument(
        "--extension-scale",
        type=float,
        default=1.0,
        help="Scale factor for the extend/retract preset",
    )
    args = parser.parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.rate <= 0:
        raise ValueError("--rate must be > 0")

    robot = _resolve_robot(args.port)
    joint_indices = [int(raw.strip()) for raw in args.joints.split(",") if raw.strip()]
    invalid_joints = [idx for idx in joint_indices if idx < 0 or idx >= robot.action_dim]
    if invalid_joints:
        raise ValueError(
            f"--joints contains invalid indices {invalid_joints}; "
            f"SO-101 supports 0..{robot.action_dim - 1}"
        )

    print(f"Connected: {robot}")
    if args.motion == "wiggle":
        print(f"Moving joints {joint_indices} for {args.duration:.1f}s at {args.rate:.1f} Hz")
    else:
        print(f"Running extend/retract preset for {args.duration:.1f}s at {args.rate:.1f} Hz")

    try:
        obs = robot.reset()
        baseline = obs["state"][0, : robot.action_dim].detach().cpu().float()
        print(f"Baseline joints: {[round(float(v), 4) for v in baseline.tolist()]}")
        extend_target = _build_extend_target(
            baseline=baseline,
            robot=robot,
            extension_scale=args.extension_scale,
        )
        if args.motion == "extend":
            named_targets = ", ".join(
                f"{_joint_name(robot, idx)}={float(extend_target[idx]):+.3f}"
                for idx in sorted(EXTEND_DELTAS)
                if idx < robot.action_dim
            )
            print(f"Extension target: {named_targets}")

        period = 1.0 / args.rate
        start = time.perf_counter()
        step = 0

        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= args.duration:
                break

            if args.motion == "wiggle":
                phase = (elapsed / args.duration) * (2.0 * math.pi * args.cycles)
                action = _build_action(
                    baseline=baseline,
                    robot=robot,
                    phase=phase,
                    amplitude=args.amplitude,
                    joint_indices=joint_indices,
                )
                status = f"offset={args.amplitude * math.sin(phase):+.3f} rad"
            else:
                action, progress = _build_extend_action(
                    baseline=baseline,
                    target=extend_target,
                    robot=robot,
                    elapsed=elapsed,
                    duration=args.duration,
                )
                status = f"extension={progress * 100.0:5.1f}%"
            robot.act(action)

            if step % max(int(args.rate), 1) == 0:
                print(f"t={elapsed:4.1f}s {status}")

            step += 1
            deadline = start + step * period
            sleep_s = deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nInterrupted, returning home...")
    finally:
        try:
            robot.go_home()
            time.sleep(1.0)
        finally:
            robot.disconnect()

    print("Done.")


if __name__ == "__main__":
    main()
