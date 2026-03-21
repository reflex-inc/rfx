# Architecture

`rfx` is an AI-native robotics middleware.

The design target is not "ROS2 but smaller" and not "a framework for one robot".
The design target is a compact core for modern AI robotics workloads where simulation,
collection, policy execution, and real hardware all share the same runtime contracts.

## Layers

### Core

The Rust core owns the middleware primitives:

- transport
- node/runtime lifecycle
- discovery
- hardware adapters
- typed wire contracts
- realtime/control utilities

### SDK

The Python SDK is the ergonomic surface for researchers and robotics engineers:

- `Robot` interface
- `Session`
- `deploy`
- `collection`
- policy artifacts
- simulation helpers

### Adapters

Robot-specific integrations are adapters, not the identity of the framework.

Examples:

- SO-101
- Go2
- G1
- simulator backends
- camera backends

### Workflows

Workflow commands such as `record`, `deploy`, `train`, and `runs` are built on top of the
core and SDK layers. They are useful entry points, but they are not the whole framework.

## First-Class Primitives

### Simulation

Simulation is a primitive, not an optional extension.

The same observation/action contract should hold across:

- mock robots
- simulators
- real robots

### Collection

Collection is a primitive, not a sidecar script.

The framework should make it straightforward to turn observations into structured datasets
with stable metadata and artifact lineage.

### Artifacts

Saved policies should be self-describing and portable across machines and embodiments.

## Non-Goals

`rfx` is not trying to be:

- a navigation stack
- a SLAM framework
- a planner library
- a full autonomy platform
- a robot-specific SDK disguised as middleware

## Influence From Dora

The useful inspiration from Dora is:

- process/dataflow discipline
- clear runtime boundaries
- low-latency transport
- observability
- small, sharp middleware semantics

The goal is to bring that clarity into an AI-native robotics framework where simulation
and collection are part of the base contract.
