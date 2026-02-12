#!/usr/bin/env python3
"""
Teleoperation Recording

Usage:
    python teleop_record.py --leader /dev/ttyACM0 --follower /dev/ttyACM1 --output demos/
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime

import torch

from rfx.real.so101 import So101LeaderFollower


def run_teleop(args):
    print("Connecting to arms...")
    teleop = So101LeaderFollower(leader_port=args.leader, follower_port=args.follower)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = []
    recording = False
    episode = 0

    print("Controls: Enter=start/stop recording, q=quit, h=home")
    print("Move the leader arm to control the follower.")

    import threading

    quit_flag = [False]
    toggle_record = [False]
    go_home = [False]

    def input_thread():
        while not quit_flag[0]:
            try:
                cmd = input()
                if cmd == "q":
                    quit_flag[0] = True
                elif cmd == "h":
                    go_home[0] = True
                else:
                    toggle_record[0] = True
            except:
                break

    threading.Thread(target=input_thread, daemon=True).start()

    step = 0
    try:
        while not quit_flag[0]:
            if go_home[0]:
                go_home[0] = False
                teleop.follower.go_home()
                time.sleep(0.5)
                continue

            if toggle_record[0]:
                toggle_record[0] = False
                if recording:
                    recording = False
                    if len(trajectory) >= 10:
                        episode += 1
                        filename = (
                            output_dir
                            / f"demo_{episode:04d}_{datetime.now().strftime('%H%M%S')}.json"
                        )
                        with open(filename, "w") as f:
                            json.dump(
                                {"trajectory": [{"positions": t.tolist()} for t in trajectory]}, f
                            )
                        print(f"Saved: {filename}")
                    trajectory = []
                else:
                    recording = True
                    print("Recording...")

            positions = teleop.step()
            if recording:
                trajectory.append(positions.clone())

            if step % 25 == 0:
                status = "REC" if recording else "   "
                print(f"[{status}] Step {step} | pos: {positions[:3].numpy()}")

            step += 1
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass

    teleop.disconnect()
    print(f"Recorded {episode} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leader", default="/dev/ttyACM0")
    parser.add_argument("--follower", default="/dev/ttyACM1")
    parser.add_argument("--output", default="demos")
    run_teleop(parser.parse_args())
