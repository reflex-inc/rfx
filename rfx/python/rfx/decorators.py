"""
rfx.decorators - @policy decorator and MotorCommands.

@policy marks a function as deployable by `rfx deploy my_file.py`.
MotorCommands lets you build actions from named joints instead of raw indices.

Example:
    # my_policy.py
    import rfx

    @rfx.policy
    def hold(obs):
        return rfx.MotorCommands({"gripper": 0.8}, config=rfx.SO101_CONFIG).to_tensor()

    # Then:
    #   rfx deploy my_policy.py --robot so101
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from .robot.config import RobotConfig


def policy(fn: Callable | None = None) -> Callable:
    """Mark a function as an rfx policy.

    A policy is any callable: ``Dict[str, Tensor] -> Tensor``.
    This decorator stamps ``_rfx_policy = True`` so that
    ``rfx deploy my_file.py`` can discover it automatically.

    Can be used with or without parentheses::

        @rfx.policy
        def my_policy(obs):
            return model(obs["state"])

        @rfx.policy()
        def my_policy(obs):
            return model(obs["state"])

    Args:
        fn: The function to decorate (when used without parentheses).

    Returns:
        The decorated function, unchanged except for the ``_rfx_policy`` marker.
    """
    if fn is None:
        # Called as @policy() with parens
        return policy

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    wrapper._rfx_policy = True  # type: ignore[attr-defined]
    return wrapper


class MotorCommands:
    """Build action tensors from named joint positions.

    Instead of writing ``action[0, 5] = 0.8`` and hoping index 5 is the gripper,
    use joint names from the robot config::

        cmd = MotorCommands({"gripper": 0.8, "elbow": -0.3}, config=SO101_CONFIG)
        action = cmd.to_tensor()  # shape: (1, max_action_dim)

    Works with any robot — resolves joint names from the config's joint list.

    Args:
        positions: Dict mapping joint name to target position.
        config: RobotConfig with joints list. Required for to_tensor().
        kp: Position gain (for backends that use it).
        kd: Damping gain (for backends that use it).
    """

    def __init__(
        self,
        positions: dict[str, float] | None = None,
        *,
        config: RobotConfig | None = None,
        kp: float = 20.0,
        kd: float = 0.5,
    ) -> None:
        self.positions = positions or {}
        self.config = config
        self.kp = kp
        self.kd = kd

    def to_tensor(self, batch_size: int = 1) -> torch.Tensor:
        """Convert named positions to a padded action tensor.

        Returns a tensor of shape ``(batch_size, max_action_dim)`` with joint
        positions placed at the correct indices from the robot config.

        Args:
            batch_size: Leading batch dimension (default: 1).

        Returns:
            Action tensor ready to pass to ``robot.act()``.

        Raises:
            ValueError: If config is not set or a joint name is not found.
        """
        import torch

        if self.config is None:
            raise ValueError(
                "MotorCommands.to_tensor() requires a config. Pass config= to the constructor."
            )

        joint_map = {j.name: j.index for j in self.config.joints}
        action = torch.zeros(batch_size, self.config.max_action_dim)

        for name, value in self.positions.items():
            if name not in joint_map:
                available = ", ".join(sorted(joint_map.keys()))
                raise ValueError(
                    f"Joint '{name}' not found in {self.config.name} config. Available: {available}"
                )
            action[:, joint_map[name]] = value

        return action

    def to_list(self) -> list[float]:
        """Convert to a flat position list using config joint ordering.

        Returns a list of length ``action_dim`` with positions at the correct
        indices. Joints not specified default to 0.0.

        Raises:
            ValueError: If config is not set.
        """
        if self.config is None:
            raise ValueError(
                "MotorCommands.to_list() requires a config. Pass config= to the constructor."
            )

        joint_map = {j.name: j.index for j in self.config.joints}
        result = [0.0] * self.config.action_dim

        for name, value in self.positions.items():
            if name in joint_map:
                idx = joint_map[name]
                if idx < len(result):
                    result[idx] = value

        return result

    @classmethod
    def from_positions(
        cls,
        positions: dict[str, float],
        config: RobotConfig | None = None,
        kp: float = 20.0,
        kd: float = 0.5,
    ) -> MotorCommands:
        """Create commands from named positions."""
        return cls(positions=positions, config=config, kp=kp, kd=kd)

    def __repr__(self) -> str:
        config_name = self.config.name if self.config else "no config"
        if self.positions:
            joints = ", ".join(f"{k}={v:.2f}" for k, v in self.positions.items())
            return f"MotorCommands({joints}, config={config_name})"
        return f"MotorCommands(empty, config={config_name})"
