# WM-Craftnet v1.0.0

Initial public release of **WM-Craftnet**, corresponding to our **CoRL 2026 Spotlight** (Top 4.4%) paper.

WM-Craftnet learns a World Synesthesia Model (WSM) for generalizable and robust dexterous in-hand rotation. It fuses proprioception, noisy wrist depth, tactile contact, and action history into an action-conditioned recurrent state, then passes that state to an asymmetric actor–critic policy as a deployable representation—without imagined rollouts.

## Highlights

- Predictive visuotactile state that reconstructs clean hand–object depth from noisy wrist observations
- Multi-object, multi-axis in-hand rotation in Isaac Gym (`set_z`, `set_x4`, `set_y`)
- Reusable WSM prior: a model pretrained on nine z-axis objects can initialize learning on other object sets
- Sim-to-real transfer to the human-sized, five-finger, 22-DoF [Sharpa Wave](https://www.sharpa.com/pages/wave) hand
- Apache-2.0 license, Copyright 2026 Sharpa Group

## Links

- Website: https://wmcraftnet.github.io
- Paper: https://arxiv.org/abs/2609.07002
- Checkpoint: https://huggingface.co/SharpaIT/WM-Craftnet
- Platform: https://www.sharpa.com/pages/wave

## What's included

- Isaac Gym training and evaluation scripts for z / x / y-axis rotation
- Released z-axis checkpoint config: `example_ckpt/wm_craftnet_set_z.yaml`
- Sharpa Wave deployment example under `deploy/`
- Demo media and documentation

The policy weights (`wm_craftnet_set_z.pth`) exceed GitHub's file-size limit and are hosted on Hugging Face rather than attached here.

## Quick start

```bash
hf download SharpaIT/WM-Craftnet wm_craftnet_set_z.pth --local-dir example_ckpt

CHECKPOINT=example_ckpt/wm_craftnet_set_z.pth \
bash scripts/test_wm_craftnet.sh
```

See the [README](https://github.com/sharpa-robotics/WM-Craftnet#getting-started) for installation, training, and real-robot deployment.

## License

Copyright 2026 Sharpa Group. Released under the [Apache License, Version 2.0](LICENSE). Third-party assets and the Sharpa Wave SDK have separate terms; see [`NOTICE`](NOTICE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@inproceedings{yin2026wmcraftnet,
  title     = {WM-Craftnet: World Synesthesia Model for Generalizable and Robust Dexterous In-Hand Manipulation},
  author    = {Yin, Jie and Zhao, Zeyuan and Tan, Xiaojing and Liu, Yang and Wang, Chiyu and Gu, Xinyang},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```
