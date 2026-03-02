"""
rfx.deploy - Deploy a trained policy to a robot.

The missing piece: load a model, connect to hardware, run inference.

    $ rfx deploy runs/my-policy --robot so101
    $ rfx deploy hf://user/my-policy --robot go2 --duration 60
    $ rfx deploy my_policy.py --robot so101
    $ rfx deploy runs/my-policy --mock

Python API:
    >>> rfx.deploy("runs/my-policy", robot="so101")
    >>> rfx.deploy("hf://user/policy", robot="go2", port="/dev/ttyACM0")
    >>> rfx.deploy("my_policy.py", robot="so101")
"""

from __future__ import annotations

import importlib.util
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .hub import LoadedPolicy, load_policy
from .robot.config import GO2_CONFIG, SO101_CONFIG, G1_CONFIG, RobotConfig
from .session import Session, SessionStats

# Robot type -> built-in config
_BUILTIN_CONFIGS: dict[str, RobotConfig] = {
    "so101": SO101_CONFIG,
    "go2": GO2_CONFIG,
    "g1": G1_CONFIG,
}


def deploy(
    policy_source: str | Path,
    *,
    robot: str | None = None,
    config: str | Path | None = None,
    port: str | None = None,
    rate_hz: float | None = None,
    duration: float | None = None,
    mock: bool = False,
    device: str = "cpu",
    warmup_s: float = 0.5,
    verbose: bool = True,
) -> SessionStats:
    """Deploy a trained policy to a robot. One function, zero ceremony.

    Args:
        policy_source: Path to saved policy directory, or "hf://org/repo".
        robot: Robot type ("so101", "go2", "g1"). Auto-detected from policy config if saved.
        config: Path to robot YAML config. Overrides ``robot`` if provided.
        port: Serial port or IP override (e.g. "/dev/ttyACM0", "192.168.123.161").
        rate_hz: Control loop frequency. Defaults to robot config's control_freq_hz.
        duration: Run time in seconds. None = infinite (Ctrl+C to stop).
        mock: If True, use MockRobot instead of real hardware.
        device: Torch device for policy inference.
        warmup_s: Seconds to sleep after reset before starting loop.
        verbose: Print status messages and stats.

    Returns:
        SessionStats with timing and jitter information.

    Example:
        >>> stats = rfx.deploy("runs/my-policy", robot="so101")
        >>> stats = rfx.deploy("hf://user/policy", robot="go2", duration=30)
    """
    # 1. Load the policy
    if verbose:
        print(f"[rfx] Loading policy from {policy_source}...")

    source_path = Path(str(policy_source))
    if source_path.suffix == ".py" and source_path.exists():
        # Load @rfx.policy-decorated function from a Python file
        policy_fn = _load_policy_from_py(source_path)
        loaded = _WrapCallable(policy_fn)
        if verbose:
            print(f"[rfx] Policy function: {policy_fn.__name__}")
    else:
        loaded = load_policy(policy_source)
        if verbose:
            print(f"[rfx] Policy type: {loaded.policy_type}")
            if loaded.robot_config:
                print(f"[rfx] Bundled robot config: {loaded.robot_config.name}")

    # 2. Resolve robot config
    robot_config = _resolve_robot_config(loaded, robot=robot, config=config)

    if port is not None:
        robot_config.hardware["port"] = port
        robot_config.hardware["ip_address"] = port

    if rate_hz is None:
        rate_hz = float(robot_config.control_freq_hz)

    # 3. Create the robot
    robot_instance = _create_robot(robot_config, mock=mock, device=device)

    if verbose:
        print(f"[rfx] Robot: {robot_instance}")
        label = f"{rate_hz} Hz"
        if duration:
            label += f" for {duration}s"
        else:
            label += ", Ctrl+C to stop"
        print(f"[rfx] Running at {label}")
        print()

    # 4. Run the control loop
    stats = _run_deploy_loop(robot_instance, loaded, rate_hz, duration, warmup_s, verbose)

    # 5. Print results
    if verbose:
        _print_stats(stats)

    # 6. Cleanup
    if hasattr(robot_instance, "disconnect"):
        robot_instance.disconnect()

    return stats


def _resolve_robot_config(
    loaded: LoadedPolicy,
    *,
    robot: str | None = None,
    config: str | Path | None = None,
) -> RobotConfig:
    """Resolve robot config from CLI args, policy metadata, or defaults."""
    # Explicit config file wins
    if config is not None:
        return RobotConfig.from_yaml(config)

    # Explicit robot type
    if robot is not None:
        robot_key = robot.lower().replace("-", "").replace("_", "")
        if robot_key in _BUILTIN_CONFIGS:
            return _BUILTIN_CONFIGS[robot_key]
        # Try as YAML path
        path = Path(robot)
        if path.exists() and path.suffix in (".yaml", ".yml"):
            return RobotConfig.from_yaml(path)
        raise ValueError(
            f"Unknown robot type: '{robot}'. "
            f"Available: {', '.join(_BUILTIN_CONFIGS.keys())}. "
            f"Or pass a path to a YAML config."
        )

    # Auto-detect from policy's bundled config
    if loaded.robot_config is not None:
        return loaded.robot_config

    # Last resort: check policy config metadata
    policy_config = loaded.config
    if "robot_config" in policy_config:
        return RobotConfig.from_dict(dict(policy_config["robot_config"]))

    raise ValueError(
        "Cannot determine robot type. Either:\n"
        "  1. Pass --robot so101 (or go2, g1)\n"
        "  2. Pass --config path/to/robot.yaml\n"
        "  3. Save your policy with rfx_config.json containing robot_config"
    )


def _create_robot(
    config: RobotConfig,
    *,
    mock: bool = False,
    device: str = "cpu",
) -> Any:
    """Create a Robot instance from config."""
    if mock:
        from .sim import MockRobot

        return MockRobot(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            max_state_dim=config.max_state_dim,
            max_action_dim=config.max_action_dim,
        )

    from .real import RealRobot

    return RealRobot(config)


def _run_deploy_loop(
    robot: Any,
    policy: LoadedPolicy,
    rate_hz: float,
    duration: float | None,
    warmup_s: float,
    verbose: bool,
) -> SessionStats:
    """Run the deploy control loop with live status output."""
    session = Session(robot, policy, rate_hz=rate_hz, warmup_s=warmup_s)

    # Handle Ctrl+C gracefully
    interrupted = False

    def _sigint_handler(signum, frame):
        nonlocal interrupted
        if interrupted:
            # Second Ctrl+C: hard exit
            sys.exit(1)
        interrupted = True
        if verbose:
            print("\n[rfx] Stopping...")
        session.stop()

    old_handler = signal.signal(signal.SIGINT, _sigint_handler)

    try:
        session.start()

        if verbose and duration:
            # Timed run with progress
            start = time.perf_counter()
            while session.is_running and not interrupted:
                elapsed = time.perf_counter() - start
                if elapsed >= duration:
                    break
                remaining = duration - elapsed
                steps = session.step_count
                actual_hz = steps / elapsed if elapsed > 0 else 0
                sys.stdout.write(
                    f"\r[rfx] {steps} steps | {actual_hz:.0f} Hz | {remaining:.0f}s remaining  "
                )
                sys.stdout.flush()
                time.sleep(0.25)
            print()
        elif verbose:
            # Infinite run with live Hz
            while session.is_running and not interrupted:
                steps = session.step_count
                sys.stdout.write(f"\r[rfx] {steps} steps  ")
                sys.stdout.flush()
                time.sleep(0.25)
            print()
        else:
            # Silent run
            session.run(duration=duration)

        session.stop()
        session.check_health()
    finally:
        signal.signal(signal.SIGINT, old_handler)

    return session.stats


class _WrapCallable:
    """Thin wrapper so a raw callable looks like LoadedPolicy to deploy()."""

    def __init__(self, fn: Any) -> None:
        self._fn = fn
        self.robot_config = None
        self.config: dict[str, Any] = {}

    @property
    def policy_type(self) -> str:
        return "python_function"

    def __call__(self, obs: Any) -> Any:
        return self._fn(obs)


def _load_policy_from_py(path: Path) -> Any:
    """Import a .py file and find the @rfx.policy-decorated function."""
    spec = importlib.util.spec_from_file_location("_rfx_user_policy", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the @policy-decorated function
    candidates = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, "_rfx_policy", False):
            candidates.append((name, obj))

    if len(candidates) == 1:
        return candidates[0][1]
    elif len(candidates) > 1:
        names = ", ".join(n for n, _ in candidates)
        raise ValueError(
            f"Multiple @rfx.policy functions found in {path.name}: {names}. "
            f"Use only one @rfx.policy per file."
        )

    # No @policy found — look for any callable named "policy"
    if hasattr(module, "policy") and callable(module.policy):
        return module.policy

    raise ValueError(
        f"No @rfx.policy function found in {path.name}. "
        f"Decorate your policy function with @rfx.policy."
    )


def _print_stats(stats: SessionStats) -> None:
    """Print deploy session results."""
    print(f"[rfx] Done - {stats.iterations} steps, {stats.overruns} overruns")
    if stats.iterations > 0:
        print(
            f"[rfx]   avg period:  {stats.avg_period_s * 1000:.2f} ms"
            f"  (target: {stats.target_period_s * 1000:.2f} ms)"
        )
        print(f"[rfx]   jitter p50:  {stats.p50_jitter_s * 1000:.2f} ms")
        print(f"[rfx]   jitter p95:  {stats.p95_jitter_s * 1000:.2f} ms")
        print(f"[rfx]   jitter p99:  {stats.p99_jitter_s * 1000:.2f} ms")
