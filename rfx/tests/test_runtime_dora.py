from __future__ import annotations

import pytest

from rfx.runtime.dora_bridge import DoraCliError, build_dataflow, run_dataflow


def test_build_dataflow_raises_when_dora_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(DoraCliError, match="Dora CLI not found"):
        build_dataflow("graph.yml")


def test_run_dataflow_raises_when_dora_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(DoraCliError, match="Dora CLI not found"):
        run_dataflow("graph.yml")


def test_dora_bridge_build_runs() -> None:
    """Verify dora_bridge.build_dataflow is importable and callable."""
    assert callable(build_dataflow)


def test_dora_bridge_run_runs() -> None:
    """Verify dora_bridge.run_dataflow is importable and callable."""
    assert callable(run_dataflow)
