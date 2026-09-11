# Installation
WM-Craftnet uses Python 3.8 and NVIDIA Isaac Gym Preview 4.

## Conda environment

```bash
conda create -n wm-craftnet python=3.8
conda activate wm-craftnet
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

## IsaacGym
Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. We provide the bash commands we used.
```bash
pip install scipy imageio ninja
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e . --no-deps
```

## Other dependencies
```bash
pip install hydra-core gym numpy==1.22.2 tensorboardX tensorboard \
  wandb scipy imageio imageio-ffmpeg h5py trimesh rtree pillow
```