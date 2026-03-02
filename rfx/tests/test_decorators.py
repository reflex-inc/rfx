"""Tests for rfx.decorators module."""

from typing import Any

import pytest

from rfx.decorators import MotorCommands, policy
from rfx.robot.config import JointConfig, RobotConfig

# ---------------------------------------------------------------------------
# @policy decorator
# ---------------------------------------------------------------------------


class TestPolicyDecorator:

    def test_policy_bare(self) -> None:
        """@rfx.policy without parentheses."""

        @policy
        def my_policy(obs: Any) -> Any:
            return obs

        assert hasattr(my_policy, "_rfx_policy")
        assert my_policy._rfx_policy is True

    def test_policy_with_parens(self) -> None:
        """@rfx.policy() with parentheses."""

        @policy()
        def my_policy(obs: Any) -> Any:
            return obs

        assert my_policy._rfx_policy is True

    def test_policy_preserves_function(self) -> None:
        @policy
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10

    def test_policy_preserves_docstring(self) -> None:
        @policy
        def documented(obs: Any) -> Any:
            """My docstring."""
            return obs

        assert documented.__doc__ == "My docstring."

    def test_policy_preserves_name(self) -> None:
        @policy
        def original_name(obs: Any) -> Any:
            return obs

        assert original_name.__name__ == "original_name"


# ---------------------------------------------------------------------------
# MotorCommands — config-aware
# ---------------------------------------------------------------------------


def _make_config(joints: list[tuple[str, int]], action_dim: int = 6) -> RobotConfig:
    return RobotConfig(
        name="test-robot",
        state_dim=action_dim * 2,
        action_dim=action_dim,
        max_state_dim=64,
        max_action_dim=64,
        joints=[JointConfig(name=n, index=i) for n, i in joints],
    )


SO101_JOINTS = [
    ("shoulder_pan", 0),
    ("shoulder_lift", 1),
    ("elbow", 2),
    ("wrist_pitch", 3),
    ("wrist_roll", 4),
    ("gripper", 5),
]


class TestMotorCommands:

    def test_empty(self) -> None:
        cmd = MotorCommands()
        assert cmd.positions == {}

    def test_positions_stored(self) -> None:
        cmd = MotorCommands({"gripper": 0.8, "elbow": -0.3})
        assert cmd.positions == {"gripper": 0.8, "elbow": -0.3}

    def test_repr_with_config(self) -> None:
        config = _make_config(SO101_JOINTS)
        cmd = MotorCommands({"gripper": 0.8}, config=config)
        r = repr(cmd)
        assert "gripper" in r
        assert "test-robot" in r

    def test_repr_empty(self) -> None:
        r = repr(MotorCommands())
        assert "empty" in r

    def test_from_positions(self) -> None:
        config = _make_config(SO101_JOINTS)
        cmd = MotorCommands.from_positions({"elbow": 1.0}, config=config, kp=30.0)
        assert cmd.positions == {"elbow": 1.0}
        assert cmd.kp == 30.0
        assert cmd.config is config


class TestMotorCommandsToTensor:

    def test_to_tensor_basic(self) -> None:
        config = _make_config(SO101_JOINTS)
        cmd = MotorCommands({"gripper": 0.8, "elbow": -0.3}, config=config)
        t = cmd.to_tensor()

        assert t.shape == (1, 64)
        assert t[0, 5].item() == pytest.approx(0.8)  # gripper at index 5
        assert t[0, 2].item() == pytest.approx(-0.3)  # elbow at index 2
        assert t[0, 0].item() == pytest.approx(0.0)  # shoulder_pan untouched

    def test_to_tensor_batch(self) -> None:
        config = _make_config(SO101_JOINTS)
        cmd = MotorCommands({"gripper": 1.0}, config=config)
        t = cmd.to_tensor(batch_size=4)

        assert t.shape == (4, 64)
        assert t[3, 5].item() == pytest.approx(1.0)

    def test_to_tensor_unknown_joint_raises(self) -> None:
        config = _make_config(SO101_JOINTS)
        cmd = MotorCommands({"nonexistent": 0.5}, config=config)

        with pytest.raises(ValueError, match="nonexistent"):
            cmd.to_tensor()

    def test_to_tensor_no_config_raises(self) -> None:
        cmd = MotorCommands({"gripper": 0.5})

        with pytest.raises(ValueError, match="requires a config"):
            cmd.to_tensor()


class TestMotorCommandsToList:

    def test_to_list_basic(self) -> None:
        config = _make_config(SO101_JOINTS)
        cmd = MotorCommands({"gripper": 0.8, "elbow": -0.3}, config=config)
        result = cmd.to_list()

        assert len(result) == 6
        assert result[5] == pytest.approx(0.8)
        assert result[2] == pytest.approx(-0.3)
        assert result[0] == pytest.approx(0.0)

    def test_to_list_no_config_raises(self) -> None:
        cmd = MotorCommands({"gripper": 0.5})

        with pytest.raises(ValueError, match="requires a config"):
            cmd.to_list()
