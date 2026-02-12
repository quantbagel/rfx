"""
rfx.real - Real hardware robot backends

Same interface as simulation - no ROS, no middleware.
"""

from .base import RealRobot
from .so101 import So101Robot
from .go2 import Go2Robot
from .camera import Camera, RealSenseCamera

__all__ = ["RealRobot", "So101Robot", "Go2Robot", "Camera", "RealSenseCamera"]
