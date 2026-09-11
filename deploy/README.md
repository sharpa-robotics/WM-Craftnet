# Real-robot inference

This directory contains the public Sharpa SDK runtime and WM-Craftnet
real-robot inference entry points.

## Requirements

- Linux x86-64
- Python 3.10 (the bundled `sharpa` extension uses the CPython 3.10 ABI)
- A supported Sharpa hand and depth-camera setup
- The shared libraries under `sharpa_sdk/lib`

Run commands from the repository root so the scripts can resolve the bundled
SDK without machine-specific absolute paths.

## Entry points

- `examples/wm_craftnet_infer.py`: local policy inference and actuation.
- `examples/wm_craftnet_policy_server.py`: policy server for split deployment.
- `examples/wm_craftnet_infer_client.py`: robot-side ZMQ client.

Inspect each command with `--help` before connecting hardware. For local
inference, set `no_actuation=True` in `build_hardcoded_config()`; for the ZMQ
client, pass `--no-actuation`. Verify joint ordering and limits before enabling
commands on the robot.

The local entry point and policy server load released checkpoints from
`example_ckpt/`. Each `*.pth` file needs a sibling `*.yaml` of the same stem
(for example `example_ckpt/wm_craftnet_set_z.pth` and
`example_ckpt/wm_craftnet_set_z.yaml`). A real depth camera is required.
Legacy E2E checkpoint compatibility and dummy-depth fallback are not
supported.

The SDK and its bundled third-party libraries retain their own licenses and
notices; the repository's Apache-2.0 license does not replace those terms.
