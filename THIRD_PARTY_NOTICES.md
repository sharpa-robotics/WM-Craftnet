# Third-Party Notices

WM-Craftnet builds on third-party software, simulation tools, robot interfaces,
datasets, and assets. Those materials are **not** relicensed by the
repository-level Apache License, Version 2.0. Each component remains governed by
its own copyright notice, license, and distribution terms.

See also [`NOTICE`](NOTICE) for attribution required by the Apache License and
the complete upstream license texts copied from the linked repositories.

## Open-source code incorporated or adapted in this repository

| Component | Upstream | License | Location / usage |
| --- | --- | --- | --- |
| Isaac Gym Environments | [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) | BSD-3-Clause | `isaacgymenvs/` training stack, task base classes, utilities |
| World Model-based Perception (WMP) | [bytedance/WMP](https://github.com/bytedance/WMP) | Apache-2.0 | World-model training integration patterns, depth preprocessing references |
| Robot Synesthesia / in-hand rotation | [YingYuan0414/in-hand-rotation](https://github.com/YingYuan0414/in-hand-rotation) | MIT | Dexterous in-hand rotation task structure and training workflow |
| DreamerV3 / dreamerv3-torch | [danijar/dreamerv3](https://github.com/danijar/dreamerv3), NM512 | MIT | `wsm/` recurrent state-space model implementation |
| rl_games | [Denys88/rl_games](https://github.com/Denys88/rl_games) | MIT | `rl_games/` PPO / RL training backend |
| Detectron2 flatten utility | Meta Detectron2 | Apache-2.0 | `rl_games/algos_torch/flatten.py` |

File-level copyright and license headers take precedence for the corresponding
source files. Where a directory contains mixed authorship, retain all upstream
notices when redistributing modified source.

## External software obtained separately

The following components are **not** bundled under the WM-Craftnet Apache-2.0
license and must be obtained and used under their own terms:

- **NVIDIA Isaac Gym Preview 4** — download from
  [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym). Governed by
  NVIDIA's license and distribution terms.
- **Sharpa Wave SDK** — Python extension, native libraries, firmware, and
  related vendor materials under `deploy/sharpa_sdk/`. Governed by the terms
  supplied by the hardware vendor.
- **Intel RealSense SDK** — if installed for deployment, governed by Intel's
  license terms.
- **PyTorch, PyTorch3D, Hydra, and other Python dependencies** — each package
  retains its upstream license.

## Robot, object, and dataset assets

Robot descriptions, meshes, URDFs, textures, object models, demonstration data,
and derived assets under `assets/` or other data directories may originate from
multiple sources. Each asset remains governed by its source license or
permission. Inclusion in this repository does not imply that the WM-Craftnet
authors own the asset or can relicense it under Apache-2.0.

Before redistributing a release, verify the provenance and redistribution terms
of every included asset and binary. Remove any material for which
redistribution permission cannot be established, or provide it through an
authorized external download.

## Checkpoints

Released checkpoints under `example_ckpt/` may encode information learned from
simulation assets and datasets. Any checkpoint-specific terms supplied with a
release apply in addition to the repository license.

## Disclaimer

This document is a notice, not legal advice, and does not replace the complete
license texts or vendor terms distributed with third-party materials.
