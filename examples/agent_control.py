#!/usr/bin/env python3
"""
LLM agent control example for the Unitree Go2

This example demonstrates:
- Defining skills with the @rfx.skill decorator
- Creating an LLM agent with available skills
- Natural language control of the robot

NOTE: Requires an API key for Anthropic or OpenAI.
Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.
"""

import os
import time

import rfx
from rfx.agent import Agent

# Global robot instance for skills to use
go2 = None


@rfx.skill
def walk_forward(distance: float = 1.0):
    """Walk forward by the specified distance in meters"""
    if go2 is None:
        return "Robot not connected"

    speed = 0.3  # m/s
    duration = distance / speed

    go2.walk(vx=speed, vy=0, vyaw=0)
    time.sleep(duration)
    go2.stand()

    return f"Walked forward {distance:.1f} meters"


@rfx.skill
def walk_backward(distance: float = 1.0):
    """Walk backward by the specified distance in meters"""
    if go2 is None:
        return "Robot not connected"

    speed = 0.2  # m/s (slower for safety)
    duration = distance / speed

    go2.walk(vx=-speed, vy=0, vyaw=0)
    time.sleep(duration)
    go2.stand()

    return f"Walked backward {distance:.1f} meters"


@rfx.skill
def turn_left(angle: float = 90.0):
    """Turn left by the specified angle in degrees"""
    if go2 is None:
        return "Robot not connected"

    import math
    rad = math.radians(angle)
    angular_speed = 0.5  # rad/s
    duration = rad / angular_speed

    go2.walk(vx=0, vy=0, vyaw=angular_speed)
    time.sleep(duration)
    go2.stand()

    return f"Turned left {angle:.0f} degrees"


@rfx.skill
def turn_right(angle: float = 90.0):
    """Turn right by the specified angle in degrees"""
    if go2 is None:
        return "Robot not connected"

    import math
    rad = math.radians(angle)
    angular_speed = 0.5  # rad/s
    duration = rad / angular_speed

    go2.walk(vx=0, vy=0, vyaw=-angular_speed)
    time.sleep(duration)
    go2.stand()

    return f"Turned right {angle:.0f} degrees"


@rfx.skill
def look_around():
    """Rotate in place to survey surroundings"""
    if go2 is None:
        return "Robot not connected"

    # Turn 360 degrees slowly
    go2.walk(vx=0, vy=0, vyaw=0.3)
    time.sleep(6.28 / 0.3)  # Full rotation
    go2.stand()

    return "Completed 360 degree survey"


@rfx.skill
def stand():
    """Make the robot stand still"""
    if go2 is None:
        return "Robot not connected"

    go2.stand()
    return "Standing"


@rfx.skill
def sit():
    """Make the robot sit down"""
    if go2 is None:
        return "Robot not connected"

    go2.sit()
    return "Sitting"


@rfx.skill
def get_status() -> str:
    """Get the current robot status including IMU and position"""
    if go2 is None:
        return "Robot not connected"

    state = go2.state()
    return (
        f"Position: {state.position}, "
        f"IMU: roll={state.imu.roll_deg:.1f}°, pitch={state.imu.pitch_deg:.1f}°, yaw={state.imu.yaw_deg:.1f}°"
    )


def main():
    global go2

    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        print("Running in demo mode without actual LLM calls.")
        demo_mode = True
    else:
        demo_mode = False

    # Connect to robot
    print("Connecting to Go2...")
    go2 = rfx.Go2.connect()

    if not go2.is_connected():
        print("Failed to connect to robot!")
        print("Running in simulation mode...")

    try:
        # Create agent with skills
        skills = [
            walk_forward,
            walk_backward,
            turn_left,
            turn_right,
            look_around,
            stand,
            sit,
            get_status,
        ]

        if demo_mode:
            # Use mock agent for testing
            from rfx.agent import MockAgent
            agent = MockAgent(skills=skills)

            print("\nAvailable skills:")
            print(agent.describe_skills())
            print()

            # Demo: execute skills directly
            print("Demo: Executing skills directly")
            print()
            print(f"  walk_forward: {agent.execute_skill('walk_forward', distance=1.0)}")
            print(f"  turn_left: {agent.execute_skill('turn_left', angle=45)}")
            print(f"  get_status: {agent.execute_skill('get_status')}")
        else:
            # Create real agent
            agent = Agent(
                model="claude-sonnet-4-20250514",  # or "gpt-4" for OpenAI
                skills=skills,
                robot=go2,
            )

            print("\nAgent created with skills:")
            print(agent.describe_skills())
            print()

            # Interactive mode
            print("Enter commands in natural language (or 'quit' to exit):")
            print("Examples:")
            print("  - walk forward 2 meters")
            print("  - turn left 90 degrees")
            print("  - look around and tell me what you see")
            print("  - what is the robot's current status?")
            print()

            while True:
                try:
                    command = input("> ").strip()
                    if not command:
                        continue
                    if command.lower() in ("quit", "exit", "q"):
                        break

                    print(f"\nExecuting: {command}")
                    result = agent.execute(command)
                    print(f"Result: {result}\n")

                except EOFError:
                    break

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        if go2 and go2.is_connected():
            print("Standing and disconnecting...")
            go2.stand()
            go2.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
