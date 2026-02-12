"""
rfx.sim - Simulation backends for parallel training

Supports: Genesis (GPU), MuJoCo MJX (JAX), Mock (CPU testing)
"""

from .base import SimRobot
from .mock import MockRobot

__all__ = ["SimRobot", "MockRobot"]
