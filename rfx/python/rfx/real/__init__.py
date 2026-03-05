"""
rfx.real - Real hardware robot backends

Same interface as simulation - no ROS, no middleware.
"""

from .base import RealRobot
from .so101 import So101Robot

try:
    from .go2 import Go2Robot
except ModuleNotFoundError:
    Go2Robot = None

try:
    from .g1 import G1Robot
except ModuleNotFoundError:
    G1Robot = None

try:
    from .innate import InnateRobot
except ModuleNotFoundError:
    InnateRobot = None

try:
    from .camera import Camera, RealSenseCamera
except ModuleNotFoundError:
    # Keep non-camera backends importable when optional camera deps (e.g. torch) are absent.
    Camera = None
    RealSenseCamera = None

__all__ = [
    "RealRobot",
    "So101Robot",
    "Go2Robot",
    "G1Robot",
    "InnateRobot",
    "Camera",
    "RealSenseCamera",
]
