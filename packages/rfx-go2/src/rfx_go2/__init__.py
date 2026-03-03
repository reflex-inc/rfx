"""rfx-sdk-go2 — Unitree Go2 integrations for rfx.

Convenience package that re-exports Go2-specific components from the core
``rfx-sdk`` so users can ``pip install rfx-sdk-go2`` for a focused install.

Example::

    from rfx_go2 import Go2Robot, Go2Backend
    robot = Go2Robot(ip_address="192.168.123.161")
    obs = robot.observe()
"""

from __future__ import annotations

from rfx.envs import Go2Env

try:
    from rfx.real.go2 import Go2Backend, Go2Robot
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "rfx-sdk-go2 requires torch. Install with: pip install rfx-sdk[teleop]"
    ) from exc

try:
    from rfx.robot.lerobot import go2 as make_go2
except ModuleNotFoundError:
    make_go2 = None  # type: ignore[assignment]

__all__ = [
    "Go2Backend",
    "Go2Env",
    "Go2Robot",
    "make_go2",
]
