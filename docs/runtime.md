# Legacy Runtime Experiments

This document describes older ROS-like runtime ideas that still exist in parts of the repository.

They are not the primary supported `rfx` product surface.

The supported framework surface is:

- `rfx.record`
- `rfx.deploy`
- `rfx.doctor`
- the Python SDK around `Robot`, `Session`, collection, transport, and policy artifacts

Some runtime/node/launch primitives remain useful as internal building blocks and future design experiments, but they should be treated as experimental until they are backed by a complete, documented CLI and stronger end-to-end contracts.
