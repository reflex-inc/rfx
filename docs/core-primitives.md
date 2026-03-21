# Core Primitives

`rfx` organizes its API into focused package surfaces for AI-native robotics middleware:

- `rfx.robot` — robot contract, config, discovery, and hardware-facing abstractions
- `rfx.sim` — simulation and mock execution surfaces
- `rfx.collection` — dataset recording and collection contracts
- `rfx.runtime` — lifecycle, health, CLI, and operational runtime helpers
- `rfx.teleop` — teleoperation and high-rate recording flows built on the same primitives

## Intent

- Keep the middleware core small and explicit.
- Treat simulation as a primitive, not an afterthought.
- Treat collection as a primitive, not a script on the side.
- Keep robot adapters separate from framework identity.
- Let workflow helpers compose on top of the primitives instead of defining them.

## Usage

```python
import rfx

robot = rfx.MockRobot(state_dim=12, action_dim=6)
obs = robot.observe()
```

Or access the package surfaces directly:

```python
import rfx

rfx.robot      # Robot protocol, config, discovery
rfx.sim        # Simulation and mock backends
rfx.collection # Collection and dataset APIs
rfx.runtime    # Runtime and operational helpers
rfx.teleop     # Teleop on top of the same robot/runtime contracts
```
