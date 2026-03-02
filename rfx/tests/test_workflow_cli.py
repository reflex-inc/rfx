from __future__ import annotations

import json
from pathlib import Path

from rfx.runtime.cli import build_parser
from rfx.workflow.registry import list_runs


def _invoke(argv: list[str]) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    return int(ns.fn(ns))


def _write_dataset(path: Path) -> None:
    rows = [
        {
            "timestamp_ns": 1_000,
            "camera_timestamp_ns": 1_010,
            "control_timestamp_ns": 1_000,
            "observation": {"state": [0.0]},
            "action": {"joint": [0.1]},
        },
        {
            "timestamp_ns": 2_000,
            "camera_timestamp_ns": 2_010,
            "control_timestamp_ns": 2_000,
            "observation": {"state": [0.0]},
            "action": {"joint": [0.1]},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_train_creates_run_record(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    assert _invoke(["train", "--data", str(dataset)]) == 0

    runs = list_runs(stage="train", limit=1)
    assert len(runs) == 1
    assert runs[0]["status"] == "succeeded"


def test_train_with_config(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    config = tmp_path / "robot.json"
    config.write_text(json.dumps({"name": "SO-101"}))

    assert _invoke(["train", "--data", str(dataset), "--config", str(config)]) == 0

    runs = list_runs(stage="train", limit=1)
    assert len(runs) == 1


def test_runs_list_empty(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert _invoke(["runs", "list"]) == 0


def test_deploy_missing_policy_returns_error(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    rc = _invoke(["deploy", "nonexistent/path", "--robot", "so101", "--mock"])
    assert rc == 1


def test_deploy_parser_accepts_all_flags() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "deploy", "runs/my-policy",
        "--robot", "so101",
        "--port", "/dev/ttyACM0",
        "--rate-hz", "100",
        "--duration", "10",
        "--mock",
        "--warmup", "1.0",
    ])
    assert ns.policy == "runs/my-policy"
    assert ns.robot == "so101"
    assert ns.port == "/dev/ttyACM0"
    assert ns.rate_hz == 100.0
    assert ns.duration == 10.0
    assert ns.mock is True
    assert ns.warmup == 1.0


def test_record_parser_accepts_all_flags() -> None:
    parser = build_parser()
    ns = parser.parse_args([
        "record",
        "--robot", "so101",
        "--repo-id", "my-org/demos",
        "--episodes", "5",
        "--duration", "30",
        "--fps", "60",
        "--push",
        "--mcap",
    ])
    assert ns.robot == "so101"
    assert ns.repo_id == "my-org/demos"
    assert ns.episodes == 5
    assert ns.duration == 30.0
    assert ns.fps == 60
    assert ns.push is True
    assert ns.mcap is True


def test_doctor_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert _invoke(["doctor"]) == 0
