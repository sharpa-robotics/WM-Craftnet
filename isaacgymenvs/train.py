# train.py
# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import datetime
import isaacgym

import os
import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import gym
import sys
import os
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    # Ensure local repo package wins over globally installed isaacgymenvs.
    sys.path.insert(0, repo_root)

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict

from isaacgymenvs.utils.utils import set_np_formatting, set_seed
import random
## OmegaConf & Hydra Config


BEIJING_TZ = datetime.timezone(datetime.timedelta(hours=8))


def _safe_segment(value):
    text = str(value).strip()
    if not text:
        return "-"
    return text.replace("/", "-").replace("\\", "-").replace(" ", "-")


def _device_index(device_name, default="0"):
    text = str(device_name).strip()
    if ":" in text:
        return text.split(":")[-1]
    return text if text else str(default)


def _infer_run_prefix_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return ""
    abs_ckpt = os.path.abspath(str(checkpoint_path))
    marker = f"{os.sep}runs{os.sep}"
    marker_idx = abs_ckpt.find(marker)
    if marker_idx < 0:
        return ""
    rel_path = abs_ckpt[marker_idx + len(marker):]
    parts = rel_path.split(os.sep)
    # Supports:
    # - runs/<axis>/<prefix>/nn/<checkpoint>.pth
    # - runs/<obj_set>/<axis>/<prefix>/nn/<checkpoint>.pth
    if len(parts) >= 4 and parts[2] == "nn":
        return parts[1]
    if len(parts) >= 5 and parts[3] == "nn":
        return parts[2]
    return ""


def _safe_close_runner_env(runner):
    """Best-effort explicit env shutdown to avoid exit-time native crashes."""
    player = getattr(runner, "player", None)
    if player is None:
        return
    env_wrapper = getattr(player, "env", None)
    candidates = [env_wrapper, getattr(env_wrapper, "env", None)]
    visited = set()
    for candidate in candidates:
        if candidate is None:
            continue
        obj_id = id(candidate)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        close_fn = getattr(candidate, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                print(f"[shutdown] env close warning: {exc}")


def _get_task_resume_checkpoint(cfg):
    if not hasattr(cfg, "task") or not hasattr(cfg.task, "env"):
        return ""
    resume_checkpoint = cfg.task.env.get("resumeCheckpoint", "")
    if resume_checkpoint is None:
        return ""
    return str(resume_checkpoint).strip()


def _apply_policy_to_rlg_network_cfg(cfg, rlg_config_dict):
    if not hasattr(cfg, "task") or not hasattr(cfg.task, "env"):
        return
    params_cfg = rlg_config_dict.get("params", {})
    if not isinstance(params_cfg, dict):
        return
    network_cfg = params_cfg.setdefault("network", {})
    config_cfg = params_cfg.setdefault("config", {})
    if not isinstance(network_cfg, dict) or not isinstance(config_cfg, dict):
        return

    # WorldModel settings are nested under cameraPolicy.
    cam_cfg = getattr(cfg.task.env, "cameraPolicy", None)
    if cam_cfg is not None:
        wm_cfg = getattr(cam_cfg, "worldModel", None)
        if wm_cfg is not None:
            wm_dict = OmegaConf.to_container(wm_cfg, resolve=True) if isinstance(wm_cfg, DictConfig) else dict(wm_cfg)
            wm_enabled = bool(wm_dict.get("enabled", False))
            network_cfg["world_model_enabled"] = wm_enabled
            network_cfg["wm_latent_dim"] = int(wm_dict.get("wm_latent_dim", 16))
            if "wm_feature_hidden" in wm_dict:
                network_cfg["wm_feature_hidden"] = list(wm_dict["wm_feature_hidden"])
            config_cfg["world_model_enabled"] = wm_enabled
            config_cfg["world_model"] = wm_dict

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
@hydra.main(config_name="config", config_path="./cfg", version_base=None)
def launch_rlg_hydra(cfg: DictConfig):
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    import isaacgymenvs

    time_str = datetime.datetime.now(BEIJING_TZ).strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    if not cfg.checkpoint:
        task_resume_checkpoint = _get_task_resume_checkpoint(cfg)
        if task_resume_checkpoint:
            cfg.checkpoint = task_resume_checkpoint

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    if hasattr(cfg.task, "env"):
        OmegaConf.update(cfg, "task.env.isTestRun", bool(cfg.test), merge=True, force_add=True)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.sim_device = f'cuda:{rank}'
        cfg.rl_device = f'cuda:{rank}'

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
            monitor_gym=True,
        )

    def make_env_thunk(local_cfg):
        def create_env_thunk(**kwargs):
            envs = isaacgymenvs.make(
                local_cfg.seed,
                local_cfg.task_name,
                local_cfg.task.env.numEnvs,
                local_cfg.sim_device,
                local_cfg.rl_device,
                local_cfg.graphics_device_id,
                local_cfg.headless,
                local_cfg.multi_gpu,
                local_cfg.capture_video,
                local_cfg.force_render,
                local_cfg,
                **kwargs,
            )

            if local_cfg.capture_video:
                envs.is_vector_env = True
                envs = gym.wrappers.RecordVideo(
                    envs,
                    f"videos/{run_name}",
                    step_trigger=lambda step: step % local_cfg.capture_video_freq == 0,
                    video_length=local_cfg.capture_video_len,
                )
            return envs

        return create_env_thunk

    create_env_thunk = make_env_thunk(cfg)

    # register the rl-games adapter to use inside the runner
    vecenv.register('RLGPU',
                    lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': create_env_thunk,
    })

    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        # WorldModel-aware PPO: only swap the a2c_continuous builder when the
        # task actually requested the world model, so behavior is unchanged
        # for existing runs.
        wm_enabled = False
        if hasattr(cfg.task, "env") and hasattr(cfg.task.env, "cameraPolicy"):
            _wm_sub = getattr(cfg.task.env.cameraPolicy, "worldModel", None)
            if _wm_sub is not None:
                wm_enabled = bool(getattr(_wm_sub, "enabled", False))
        if wm_enabled:
            from isaacgymenvs.learning import wm_continuous
            runner.algo_factory.register_builder(
                'a2c_continuous', lambda **kwargs: wm_continuous.WMA2CAgent(**kwargs)
            )

        return runner

    time_prefix = time_str
    axis_name = _safe_segment(cfg.task.env.get("axis", "z"))
    obj_set = _safe_segment(cfg.task.env.get("objSet", "0"))
    cam_policy_cfg = cfg.task.env.get("cameraPolicy", {})
    enc_segments = []
    wm_cfg_for_tag = cam_policy_cfg.get("worldModel", {}) if cam_policy_cfg else {}
    if bool(wm_cfg_for_tag.get("enabled", False)):
        wm_parts = ["wm"]
        if bool(wm_cfg_for_tag.get("wm_pose_pred", False)):
            wm_parts.append("pose")
        if bool(wm_cfg_for_tag.get("wm_prop_pred", True)):
            wm_parts.append("prop")
        if bool(wm_cfg_for_tag.get("wm_tac_pred", False)):
            wm_parts.append("tac")
        if bool(wm_cfg_for_tag.get("wm_depth_pred", True)):
            wm_parts.append("depth")
        if bool(wm_cfg_for_tag.get("wm_value_pred", False)):
            wm_parts.append("value")
        if bool(wm_cfg_for_tag.get("wm_obj_pred", False)):
            wm_parts.append("obj")
        enc_segments.append("_".join(wm_parts))
    head_tag = ("_" + "_".join(enc_segments)) if enc_segments else ""
    user_prefix_raw = ""
    if hasattr(cfg.train.params.config, 'user_prefix'):
        user_prefix_raw = cfg.train.params.config.user_prefix
    user_prefix_clean = _safe_segment(user_prefix_raw)
    user_tag = f"_{user_prefix_clean}" if user_prefix_clean != "-" else ""
    inferred_prefix = ""
    if cfg.test and cfg.checkpoint:
        inferred_prefix = _infer_run_prefix_from_checkpoint(cfg.checkpoint)
    if inferred_prefix:
        prefix = inferred_prefix
    else:
        prefix = f"{time_prefix}{head_tag}{user_tag}"
    train_dir = os.path.join("runs", obj_set, axis_name)
    cfg.train.params.config.prefix = prefix

    experiment_dir = os.path.join(train_dir, prefix)
    config_dir = os.path.join(experiment_dir, 'config')
    videos_dir = os.path.join(experiment_dir, 'videos')
    if hasattr(cfg.task, "env") and hasattr(cfg.task.env, "cameraDemo") and hasattr(cfg.task.env.cameraDemo, "record"):
        keep_existing_record_root = False
        if cfg.test:
            existing_record_root = str(cfg.task.env.cameraDemo.record.get("root_dir", "")).strip()
            keep_existing_record_root = bool(existing_record_root)
        if not keep_existing_record_root:
            cfg.task.env.cameraDemo.record.root_dir = videos_dir

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    _apply_policy_to_rlg_network_cfg(cfg, rlg_config_dict)
    # Keep rl_games model/update device aligned with env rl_device.
    # Otherwise rl_games falls back to cuda:0 and can mismatch env tensors on other GPUs.
    rlg_config_dict['params']['config']['device'] = cfg.rl_device
    rlg_config_dict['params']['config']['train_dir'] = train_dir
    print(rlg_config_dict)
    rlg_config_dict['params']['config']['prefix'] = prefix

    # convert CLI arguments into dictionory
    # create runner and set the settings
    runner = build_runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    # dump config dict only for training runs.
    if not cfg.test:
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, 'config.yaml'), 'w') as f:
            f.write(OmegaConf.to_yaml(cfg))

    try:
        runner.run({
            'train': not cfg.test,
            'play': cfg.test,
            'checkpoint' : cfg.checkpoint,
            'sigma' : None
        })
    finally:
        _safe_close_runner_env(runner)
        if cfg.wandb_activate and rank == 0:
            wandb.finish()

if __name__ == "__main__":
    launch_rlg_hydra()
