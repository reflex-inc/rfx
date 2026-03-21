# Extension Packages

This folder contains adapter-oriented package splits for integrations that sit on
top of the core `rfx-sdk`.

The framework itself is not defined by these packages. They exist so users can
bring specific robots, simulators, or dataset integrations into an otherwise
generic middleware/runtime stack.

| Package | Install | Provides |
|---------|---------|----------|
| `rfx-sdk-go2` | `pip install rfx-sdk-go2` | `Go2Robot`, `Go2Backend`, `Go2Env`, `make_go2` factory |
| `rfx-sdk-sim` | `pip install rfx-sdk-sim` | `SimRobot`, `MockRobot`, `BaseEnv`, `VecEnv`, `make_vec_env` |
| `rfx-sdk-lerobot` | `pip install rfx-sdk-lerobot` | `collect`, `Dataset`, `Recorder`, `LeRobotPackageWriter`, Hub helpers |

The base Python import remains `rfx` (published as `rfx-sdk`).
