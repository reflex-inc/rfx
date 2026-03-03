"""rfx-sdk-lerobot — LeRobot integration package for rfx.

Convenience package that re-exports LeRobot data-collection and dataset
components from the core ``rfx-sdk`` so users can
``pip install rfx-sdk-lerobot`` for a focused install.

Example::

    from rfx_lerobot import collect, Dataset
    dataset = collect("so101", "my-org/demos", episodes=10)
    dataset.push()
"""

from __future__ import annotations

from rfx.collection import Dataset, Recorder, collect, from_hub, open_dataset, pull, push
from rfx.teleop.lerobot_writer import LeRobotExportConfig, LeRobotPackageWriter

__all__ = [
    "Dataset",
    "LeRobotExportConfig",
    "LeRobotPackageWriter",
    "Recorder",
    "collect",
    "from_hub",
    "open_dataset",
    "pull",
    "push",
]
