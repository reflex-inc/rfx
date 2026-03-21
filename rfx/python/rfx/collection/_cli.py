"""rfx.collection._cli — CLI integration for rfx collect."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def add_collect_args(parser: argparse.ArgumentParser) -> None:
    """Add args to the collect subcommand."""
    parser.add_argument("--robot", required=True, help="robot type (e.g. so101)")
    parser.add_argument("--repo-id", required=True, help="HuggingFace dataset repo ID")
    parser.add_argument("--output", "-o", default="datasets", help="output root directory")
    parser.add_argument(
        "--episodes", "-n", type=int, default=1, help="number of episodes to collect"
    )
    parser.add_argument(
        "--duration", "-d", type=float, default=None, help="duration per episode in seconds"
    )
    parser.add_argument("--task", default="default", help="task label for episodes")
    parser.add_argument("--fps", type=int, default=30, help="recording frame rate")
    parser.add_argument("--rate-hz", type=float, default=None, help="sampling rate override")
    parser.add_argument("--config", default=None, help="path to robot YAML config")
    parser.add_argument("--port", default=None, help="serial port or IP override")
    parser.add_argument(
        "--camera-id",
        action="append",
        default=[],
        help="camera device id (repeatable, overrides config cameras)",
    )
    parser.add_argument("--mock", action="store_true", help="use MockRobot for dry-run recording")
    parser.add_argument("--push", action="store_true", help="push to Hub after collection")
    parser.add_argument("--mcap", action="store_true", help="also log MCAP sidecar")
    parser.add_argument("--state-dim", type=int, default=None, help="state dimension override")


def run_collection(args: argparse.Namespace) -> dict[str, Any]:
    """Execute collection. Called by workflow/stages.py or CLI."""
    from . import collect

    dataset = collect(
        args.robot,
        args.repo_id,
        output=args.output,
        episodes=args.episodes,
        duration_s=args.duration,
        task=args.task,
        fps=args.fps,
        state_dim=args.state_dim,
        config=args.config,
        port=args.port,
        rate_hz=args.rate_hz,
        mock=bool(args.mock),
        camera_ids=list(args.camera_id),
        push_to_hub=bool(args.push),
        mcap=args.mcap,
    )

    return {
        "repo_id": args.repo_id,
        "episodes": int(dataset.num_episodes),
        "total_frames": int(dataset.num_frames),
        "root": str(Path(args.output).resolve()),
    }
