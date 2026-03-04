"""rfx-sdk-sim — Simulation adapters and controllers for rfx.

Convenience package that re-exports simulation components from the core
``rfx-sdk`` so users can ``pip install rfx-sdk-sim`` for a focused install.

Example::

    from rfx_sim import SimRobot, MockRobot
    robot = SimRobot.from_config("so101.yaml", backend="mock", num_envs=16)
    obs = robot.observe()
"""

from __future__ import annotations

from rfx.envs import BaseEnv, Box, VecEnv, make_vec_env

try:
    from rfx.sim import MockRobot, SimRobot
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "rfx-sdk-sim requires torch. Install with: pip install rfx-sdk[sim-mock]"
    ) from exc

__all__ = [
    "BaseEnv",
    "Box",
    "MockRobot",
    "SimRobot",
    "VecEnv",
    "make_vec_env",
]
