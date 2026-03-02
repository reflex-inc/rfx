"""
rfx CLI - Three commands. That's it.

    rfx record   - Record demonstrations from hardware
    rfx train    - Train a policy from recorded data
    rfx deploy   - Deploy a trained policy to a robot

    rfx doctor   - Check your setup
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# deploy
# ---------------------------------------------------------------------------


def cmd_deploy(args: argparse.Namespace) -> int:
    """Deploy a trained policy to a robot."""
    from rfx.deploy import deploy

    try:
        deploy(
            policy_source=args.policy,
            robot=args.robot,
            config=args.config,
            port=args.port,
            rate_hz=args.rate_hz,
            duration=args.duration,
            mock=args.mock,
            warmup_s=args.warmup,
            verbose=True,
        )
    except KeyboardInterrupt:
        pass
    except FileNotFoundError as exc:
        print(f"[rfx] Error: {exc}")
        return 1
    except ValueError as exc:
        print(f"[rfx] Error: {exc}")
        return 1
    except Exception as exc:
        print(f"[rfx] Deploy failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def cmd_record(args: argparse.Namespace) -> int:
    """Record demonstrations from a robot."""
    from rfx.collection._cli import run_collection

    try:
        result = run_collection(args)
        print(f"[rfx] Recorded {result['episodes']} episodes, {result['total_frames']} frames")
        print(f"[rfx] Saved to {result['root']}")
    except KeyboardInterrupt:
        print("\n[rfx] Recording stopped.")
    except Exception as exc:
        print(f"[rfx] Record failed: {type(exc).__name__}: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    """Train a policy from collected data."""
    from rfx.workflow.registry import (
        create_run_record,
        generate_run_id,
        materialize_refs,
        snapshot_config,
    )
    from rfx.workflow.stages import execute_stage

    run_id = generate_run_id("train")
    config_data = snapshot_config(args.config)
    input_refs = materialize_refs(list(args.input))
    output_refs = materialize_refs(list(args.output))

    metadata: dict[str, Any] = {}
    if args.data:
        input_refs = materialize_refs([args.data]) + input_refs
        metadata["data"] = args.data

    try:
        result = execute_stage(
            stage="train",
            run_id=run_id,
            root=Path.cwd(),
            config_snapshot_data=config_data,
            input_refs=input_refs,
            output_refs=output_refs,
            metadata=metadata,
        )
        if result.generated_outputs:
            output_refs = output_refs + materialize_refs(result.generated_outputs)

        create_run_record(
            run_id=run_id,
            stage="train",
            status=result.status,
            invocation_argv=list(sys.argv) if sys.argv else ["rfx"],
            config_snapshot_data=config_data,
            input_refs=input_refs,
            output_refs=output_refs,
            metadata=result.metadata,
            reports=result.reports,
            artifacts=result.artifacts,
        )
    except Exception as exc:
        print(f"[rfx] Train failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[rfx] train run_id={run_id} status={result.status}")
    if result.message:
        print(f"[rfx] {result.message}")
    for artifact in result.artifacts:
        print(f"[rfx] artifact: {artifact.get('ref')}")
    return 0 if result.status == "succeeded" else 1


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(_args: argparse.Namespace) -> int:
    """Check system setup."""
    import shutil

    checks = [
        ("python", shutil.which("python3") or shutil.which("python")),
        ("cargo", shutil.which("cargo")),
        ("uv", shutil.which("uv")),
    ]

    # Check Python imports
    optional_imports = [
        ("torch", "PyTorch (policy inference)"),
        ("tinygrad", "tinygrad (lightweight policies)"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML (config loading)"),
    ]

    print("[rfx] System check\n")

    all_ok = True
    for name, path in checks:
        status = "ok" if path else "missing"
        if not path:
            all_ok = False
        print(f"  {name:12s}  {status:8s}  {path or ''}")

    print()

    for module, desc in optional_imports:
        try:
            __import__(module)
            print(f"  {desc:36s}  ok")
        except ImportError:
            print(f"  {desc:36s}  not installed")

    # Check Rust extension
    print()
    try:
        from rfx import _rfx  # noqa: F401

        print(f"  {'rfx Rust extension':36s}  ok")
    except ImportError:
        print(f"  {'rfx Rust extension':36s}  not built (run: maturin develop)")

    # Check serial ports
    print()
    try:
        from rfx.robot.discovery import discover_ports

        ports = discover_ports()
        if ports:
            print("  Serial ports:")
            for p in ports:
                print(f"    {p}")
        else:
            print("  No serial ports found")
    except Exception:
        print("  Serial port detection: unavailable")

    print()
    if all_ok:
        print("[rfx] All good.")
    else:
        print("[rfx] Some tools missing. See above.")

    return 0


# ---------------------------------------------------------------------------
# runs (lightweight registry query)
# ---------------------------------------------------------------------------


def cmd_runs_list(args: argparse.Namespace) -> int:
    from rfx.workflow.registry import list_runs

    runs = list_runs(stage=args.stage, status=args.status, limit=args.limit)
    if not runs:
        print("No runs found.")
        return 0
    for run in runs:
        print(
            f"{run.get('run_id')}  {run.get('stage')}  {run.get('status')}  {run.get('finished_at')}"
        )
    return 0


def cmd_runs_show(args: argparse.Namespace) -> int:
    from rfx.workflow.registry import load_run

    run = load_run(args.run_id)
    print(json.dumps(run, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rfx",
        description="rfx - The PyTorch for Robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  rfx record --robot so101 --repo-id my-org/demos --episodes 10
  rfx train  --data demos/ --config train.yaml
  rfx deploy runs/my-policy --robot so101
  rfx deploy hf://user/my-policy --robot go2 --duration 60
  rfx deploy runs/my-policy --mock
""",
    )
    p.add_argument("--version", action="version", version="%(prog)s 0.2.0")
    sp = p.add_subparsers(dest="cmd", required=True)

    # --- deploy ---
    s = sp.add_parser(
        "deploy",
        help="deploy a trained policy to a robot",
        description="Load a policy and run it on hardware (or mock). "
        "This is the main command — train, then deploy.",
    )
    s.add_argument(
        "policy",
        help='path to saved policy directory, or "hf://org/repo" for HuggingFace Hub',
    )
    s.add_argument(
        "--robot",
        "-r",
        default=None,
        help="robot type (so101, go2, g1) or path to YAML config",
    )
    s.add_argument("--config", default=None, help="path to robot YAML config (overrides --robot)")
    s.add_argument("--port", "-p", default=None, help="serial port or IP address override")
    s.add_argument(
        "--rate-hz", type=float, default=None, help="control loop Hz (default: from robot config)"
    )
    s.add_argument(
        "--duration",
        "-d",
        type=float,
        default=None,
        help="run time in seconds (default: infinite, Ctrl+C to stop)",
    )
    s.add_argument("--mock", action="store_true", help="use MockRobot (no hardware needed)")
    s.add_argument("--warmup", type=float, default=0.5, help="warmup sleep after reset (seconds)")
    s.set_defaults(fn=cmd_deploy)

    # --- record ---
    s = sp.add_parser(
        "record",
        help="record demonstrations from a robot",
        description="Collect teleoperation demos into a LeRobot dataset.",
    )
    s.add_argument("--robot", required=True, help="robot type (e.g. so101)")
    s.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    s.add_argument("--output", "-o", default="datasets", help="output root directory")
    s.add_argument("--episodes", "-n", type=int, default=1, help="number of episodes to collect")
    s.add_argument(
        "--duration", "-d", type=float, default=None, help="duration per episode in seconds"
    )
    s.add_argument("--task", default="default", help="task label for episodes")
    s.add_argument("--fps", type=int, default=30, help="recording frame rate")
    s.add_argument("--push", action="store_true", help="push to Hub after collection")
    s.add_argument("--mcap", action="store_true", help="also log MCAP sidecar")
    s.add_argument("--state-dim", type=int, default=6, help="state dimension")
    s.set_defaults(fn=cmd_record)

    # --- train ---
    s = sp.add_parser(
        "train",
        help="train a policy from collected data",
        description="Run a training stage and register the resulting artifact.",
    )
    s.add_argument("--data", default=None, help="path to training data directory or dataset")
    s.add_argument("--config", default=None, help="training config file")
    s.add_argument(
        "--input", action="append", default=[], help="additional input refs (repeatable)"
    )
    s.add_argument(
        "--output", action="append", default=[], help="additional output refs (repeatable)"
    )
    s.set_defaults(fn=cmd_train)

    # --- doctor ---
    s = sp.add_parser("doctor", help="check system setup and dependencies")
    s.set_defaults(fn=cmd_doctor)

    # --- runs ---
    s = sp.add_parser("runs", help="query the run registry")
    runs_sp = s.add_subparsers(dest="runs_cmd", required=True)

    rs = runs_sp.add_parser("list", help="list past runs")
    rs.add_argument("--stage", default=None)
    rs.add_argument("--status", default=None)
    rs.add_argument("--limit", type=int, default=20)
    rs.set_defaults(fn=cmd_runs_list)

    rs = runs_sp.add_parser("show", help="show a run record")
    rs.add_argument("run_id")
    rs.set_defaults(fn=cmd_runs_show)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(args.fn(args))


if __name__ == "__main__":
    main()
