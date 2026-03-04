# Extension Packages

This folder contains Python package splits for provider-specific integrations.
Each package re-exports the relevant subset of the core `rfx-sdk` so users can
install only what they need.

| Package | Install | Provides |
|---------|---------|----------|
| `rfx-sdk-go2` | `pip install rfx-sdk-go2` | `Go2Robot`, `Go2Backend`, `Go2Env`, `make_go2` factory |
| `rfx-sdk-sim` | `pip install rfx-sdk-sim` | `SimRobot`, `MockRobot`, `BaseEnv`, `VecEnv`, `make_vec_env` |
| `rfx-sdk-lerobot` | `pip install rfx-sdk-lerobot` | `collect`, `Dataset`, `Recorder`, `LeRobotPackageWriter`, Hub helpers |

The base Python import remains `rfx` (published as `rfx-sdk`).
