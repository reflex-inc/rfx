"""
rfx.real.innate - Innate bot hardware backend via Zenoh.

The Innate bot publishes state and accepts commands over Zenoh topics.
This backend subscribes to state, deserializes the payload (JSON or
CDR-encoded sensor_msgs/JointState), and publishes commands back.

No Rust RobotNode needed — the bot itself is the Zenoh publisher.

Topic names and message format are configurable via the hardware config:

    hardware = {
        "zenoh_state_topic": "innate/joint_states",
        "zenoh_cmd_topic": "innate/joint_commands",
        "msg_format": "json",          # "json" or "cdr"
        "zenoh_endpoint": "tcp/...",   # optional router address
    }
"""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
from typing import TYPE_CHECKING, Any

import torch

from ..observation import make_observation
from ..robot.config import RobotConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default Zenoh topic names — override via hardware config
_DEFAULT_STATE_TOPIC = "innate/joint_states"
_DEFAULT_CMD_TOPIC = "innate/joint_commands"


def _decode_cdr_joint_state(data: bytes) -> dict[str, list[float]]:
    """Minimal CDR deserializer for sensor_msgs/msg/JointState.

    ROS 2 CDR layout (little-endian, after 4-byte RTPS header):
        Header header           (stamp + frame_id string)
        string[] name           (sequence of joint name strings)
        float64[] position
        float64[] velocity
        float64[] effort

    We skip the header and names and just extract the three float64 arrays.
    """
    offset = 4  # skip CDR encapsulation header (0x00 0x01 0x00 0x00)

    def _align(off: int, alignment: int) -> int:
        return (off + alignment - 1) & ~(alignment - 1)

    def _read_uint32(off: int) -> tuple[int, int]:
        val = struct.unpack_from("<I", data, off)[0]
        return val, off + 4

    def _skip_string(off: int) -> int:
        str_len, off = _read_uint32(off)
        return off + str_len  # includes null terminator

    def _read_float64_array(off: int) -> tuple[list[float], int]:
        off = _align(off, 4)
        count, off = _read_uint32(off)
        if count == 0:
            return [], off
        off = _align(off, 8)
        values = list(struct.unpack_from(f"<{count}d", data, off))
        return values, off + count * 8

    # --- Header ---
    # stamp: sec (int32) + nanosec (uint32)
    offset = _align(offset, 4)
    offset += 8  # skip stamp
    # frame_id string
    offset = _skip_string(offset)

    # --- name[] (string sequence) ---
    offset = _align(offset, 4)
    name_count, offset = _read_uint32(offset)
    for _ in range(name_count):
        offset = _skip_string(offset)

    # --- position[], velocity[], effort[] ---
    position, offset = _read_float64_array(offset)
    velocity, offset = _read_float64_array(offset)
    effort, offset = _read_float64_array(offset)

    return {"position": position, "velocity": velocity, "effort": effort}


class InnateBackend:
    """Innate bot backend — reads state and sends commands via Zenoh.

    The bot already publishes its own state to Zenoh topics. This backend
    simply subscribes and forwards commands. No Rust RobotNode needed.

    Args:
        config: RobotConfig for the innate bot.
        zenoh_state_topic: Zenoh key expression for state subscription.
        zenoh_cmd_topic: Zenoh key expression for command publishing.
        msg_format: Message format — "json" or "cdr" (CDR JointState).
        zenoh_endpoint: Optional Zenoh router endpoint (e.g. "tcp/10.0.0.1:7447").
        name: Robot instance name (for logging).
    """

    def __init__(
        self,
        config: RobotConfig,
        zenoh_state_topic: str | None = None,
        zenoh_cmd_topic: str | None = None,
        msg_format: str = "json",
        zenoh_endpoint: str | None = None,
        name: str = "innate",
        **kwargs: Any,
    ):
        self.config = config
        self._name = name
        self._msg_format = msg_format.lower()
        self._state_topic = zenoh_state_topic or _DEFAULT_STATE_TOPIC
        self._cmd_topic = zenoh_cmd_topic or _DEFAULT_CMD_TOPIC

        self._latest_state: dict[str, list[float]] | None = None
        self._state_lock = threading.Lock()
        self._connected = False

        try:
            import zenoh
        except ImportError as err:
            raise ImportError(
                "zenoh Python package required for Innate backend. "
                "Install with: pip install eclipse-zenoh"
            ) from err

        # Open Zenoh session
        zenoh_config = zenoh.Config()
        if zenoh_endpoint:
            zenoh_config.insert_json5("connect/endpoints", json.dumps([zenoh_endpoint]))

        logger.info(
            "Innate[%s] opening Zenoh session (state=%s, cmd=%s, format=%s)",
            name,
            self._state_topic,
            self._cmd_topic,
            self._msg_format,
        )
        self._session = zenoh.open(zenoh_config)

        # Publisher for commands
        self._cmd_publisher = self._session.declare_publisher(self._cmd_topic)

        # Subscriber for state
        self._subscriber = self._session.declare_subscriber(self._state_topic, self._on_state)
        self._connected = True
        logger.info("Innate[%s] connected — listening on %s", name, self._state_topic)

    def _on_state(self, sample: Any) -> None:
        """Callback invoked by Zenoh when a state message arrives."""
        try:
            payload = bytes(sample.payload)
            if self._msg_format == "cdr":
                parsed = _decode_cdr_joint_state(payload)
            else:
                raw = json.loads(payload)
                # Accept both rfx RobotState format and raw JointState-like dicts
                if "joint_positions" in raw:
                    parsed = {
                        "position": raw["joint_positions"],
                        "velocity": raw.get("joint_velocities", []),
                        "effort": raw.get("joint_torques", []),
                    }
                else:
                    parsed = {
                        "position": raw.get("position", []),
                        "velocity": raw.get("velocity", []),
                        "effort": raw.get("effort", []),
                    }
            with self._state_lock:
                self._latest_state = parsed
        except Exception:
            logger.warning("Innate[%s] failed to parse state message", self._name, exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        return self._connected

    def observe(self) -> dict[str, torch.Tensor]:
        with self._state_lock:
            state = self._latest_state

        n_joints = self.config.action_dim

        if state is not None:
            pos = state.get("position", [])
            vel = state.get("velocity", [])
            # Pad or truncate to expected joint count
            positions = torch.tensor((pos + [0.0] * n_joints)[:n_joints], dtype=torch.float32)
            velocities = torch.tensor((vel + [0.0] * n_joints)[:n_joints], dtype=torch.float32)
        else:
            positions = torch.zeros(n_joints, dtype=torch.float32)
            velocities = torch.zeros(n_joints, dtype=torch.float32)

        raw_state = torch.cat([positions, velocities]).unsqueeze(0)

        return make_observation(
            state=raw_state,
            state_dim=self.config.state_dim,
            max_state_dim=self.config.max_state_dim,
            device="cpu",
        )

    def act(self, action: torch.Tensor) -> None:
        targets = action[0, : self.config.action_dim].cpu().tolist()

        if self._msg_format == "cdr":
            # Send as raw float64 array (simple binary protocol)
            payload = struct.pack(f"<{len(targets)}d", *targets)
        else:
            payload = json.dumps({"position": targets}).encode()

        self._cmd_publisher.put(payload)

    def reset(self) -> dict[str, torch.Tensor]:
        home = [0.0] * self.config.action_dim
        if self._msg_format == "cdr":
            payload = struct.pack(f"<{len(home)}d", *home)
        else:
            payload = json.dumps({"position": home}).encode()
        self._cmd_publisher.put(payload)
        time.sleep(0.1)
        return self.observe()

    def go_home(self) -> None:
        home = [0.0] * self.config.action_dim
        if self._msg_format == "cdr":
            payload = struct.pack(f"<{len(home)}d", *home)
        else:
            payload = json.dumps({"position": home}).encode()
        self._cmd_publisher.put(payload)

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        try:
            self._subscriber.undeclare()
        except Exception:
            pass
        try:
            self._cmd_publisher.undeclare()
        except Exception:
            pass
        try:
            self._session.close()
        except Exception:
            pass
        logger.info("Innate[%s] disconnected", self._name)


class InnateRobot:
    """Convenience factory for Innate bot.

    Examples:
        >>> robot = InnateRobot()
        >>> robot = InnateRobot(zenoh_endpoint="tcp/10.0.0.1:7447")
    """

    def __new__(cls, **kwargs):
        from .base import RealRobot
        from ..robot.config import INNATE_CONFIG

        return RealRobot(config=INNATE_CONFIG, robot_type="innate", **kwargs)
