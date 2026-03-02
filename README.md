<div align="center">

[<img alt="rfx logo" src="docs/assets/logo.svg" width="220" />](https://github.com/quantbagel/rfx)

**The robotics framework for the foundation model era.**

[Documentation](https://deepwiki.com/quantbagel/rfx) | [Discord](https://discord.gg/xV8bAGM8WT)

[![CI](https://github.com/quantbagel/rfx/actions/workflows/ci.yml/badge.svg)](https://github.com/quantbagel/rfx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)
[![Discord](https://img.shields.io/badge/Discord-Join%20Chat-5865F2?logo=discord&logoColor=white)](https://discord.gg/xV8bAGM8WT)

</div>

---

```bash
uv pip install rfx-sdk
```

Three commands. Any robot. Any policy.

```bash
rfx record --robot so101 --repo-id my-org/demos --episodes 10
rfx deploy runs/my-policy --robot so101
rfx deploy hf://user/my-policy --robot go2 --duration 60
```

---

## Why rfx

ROS was built for message passing between components. We're in a different era -- the workflow is **collect demos, train a policy, deploy, iterate**. rfx is built from scratch for that loop.

- **Three commands** -- `rfx record`, `rfx deploy`, `rfx doctor` -- that's the whole CLI
- **Three-method robot interface** -- `observe()`, `act()`, `reset()` -- same API for sim and real
- **Self-describing models** -- save once, load anywhere, deploy with zero config
- **HuggingFace Hub native** -- push and pull policies like you push datasets
- **Rust core** for real-time control, Python SDK for fast research
- **Zenoh transport** underneath -- invisible plumbing, there when you need it

## Install

```bash
uv pip install rfx-sdk
```

Or with pip:

```bash
pip install rfx-sdk
```

From source:

```bash
git clone https://github.com/quantbagel/rfx.git && cd rfx
bash scripts/setup-from-source.sh
```

All CLI commands can also be run directly with `uv`:

```bash
uv run rfx deploy runs/my-policy --robot so101
uv run rfx record --robot so101 --repo-id demos --episodes 10
```

## Record demos

Collect teleoperation demonstrations into a LeRobot dataset:

```bash
rfx record --robot so101 --repo-id my-org/demos --episodes 10
```

```python
# Or from Python
rfx.collection.collect("so101", "my-org/demos", episodes=10)
```

## Deploy a policy

Load a trained policy and run it on hardware. One command:

```bash
# From a saved checkpoint
rfx deploy runs/my-policy --robot so101

# From HuggingFace Hub
rfx deploy hf://rfx-community/go2-walk-v1 --robot go2

# From a Python file with @rfx.policy
rfx deploy my_policy.py --robot so101

# Test without hardware
rfx deploy runs/my-policy --robot so101 --mock
```

```python
# Or from Python
rfx.deploy("runs/my-policy", robot="so101")
rfx.deploy("hf://user/policy", robot="go2", duration=30)
```

Deploy handles everything: load weights, resolve robot config, connect hardware, run the control loop with rate control and jitter tracking, clean shutdown on Ctrl+C.

## The robot interface

Every robot -- simulated or real -- implements three methods:

```python
obs = robot.observe()    # {"state": Tensor(1, 64), "images": ...}
robot.act(action)        # Tensor(1, 64)
robot.reset()
```

## Write a policy

A policy is any callable `Dict[str, Tensor] -> Tensor`. Use `@rfx.policy` to make it deployable from the CLI:

```python
# my_policy.py
import torch
import rfx

@rfx.policy
def hold_position(obs):
    return torch.zeros(1, 64)  # hold still
```

```bash
rfx deploy my_policy.py --robot so101
```

For named joint control instead of raw tensor indices:

```python
@rfx.policy
def grasp(obs):
    return rfx.MotorCommands(
        {"gripper": 0.8, "wrist_pitch": -0.2},
        config=rfx.SO101_CONFIG,
    ).to_tensor()
```

## Save and share models

Every saved model is a self-describing directory:

```python
policy.save("runs/go2-walk-v1",
    robot_config=config,
    normalizer=normalizer,
    training_info={"total_steps": 50000})
```

```
runs/go2-walk-v1/
  rfx_config.json       # architecture + robot + training metadata
  model.safetensors     # weights
  normalizer.json       # observation normalizer state
```

Push to Hub, load anywhere:

```python
rfx.push_policy("runs/go2-walk-v1", "rfx-community/go2-walk-v1")
loaded = rfx.load_policy("hf://rfx-community/go2-walk-v1")
```

## Supported hardware

| Robot | Type | Interface | Status |
|-------|------|-----------|--------|
| **SO-101** | 6-DOF arm | USB serial (Rust driver) | Ready |
| **Unitree Go2** | Quadruped | Ethernet (Zenoh transport) | Ready |
| **Unitree G1** | Humanoid | Ethernet (Zenoh transport) | In progress |

Custom robots: implement `observe()` / `act()` / `reset()` or write a YAML config.

## Simulation

```python
robot = rfx.SimRobot.from_config("so101.yaml", backend="genesis", viewer=True)
robot = rfx.SimRobot.from_config("go2.yaml", backend="mjx", num_envs=4096)
robot = rfx.MockRobot(state_dim=12, action_dim=6)  # zero deps, for testing
```

## Docs

- [Full documentation](https://deepwiki.com/quantbagel/rfx)
- [SO-101 quickstart](docs/so101.md)
- [Simulation guide](docs/sim.md)
- [Python SDK reference](docs/python-sdk.md)

## Community

- [Issues](https://github.com/quantbagel/rfx/issues)
- [Discussions](https://github.com/quantbagel/rfx/discussions)
- [Discord](https://discord.gg/xV8bAGM8WT)

## License

MIT
