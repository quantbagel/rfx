"""
rfx.real - Real hardware robot backends

Same interface as simulation - no ROS, no middleware.
"""

from .base import RealRobot
from .camera import Camera, RealSenseCamera
from .go2 import Go2Robot
from .so101 import So101Robot

__all__ = ["RealRobot", "So101Robot", "Go2Robot", "Camera", "RealSenseCamera"]
