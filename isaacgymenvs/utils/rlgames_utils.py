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

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
from isaacgymenvs.utils.utils import set_seed
import os
import torch
import numpy as np
from typing import Callable

from isaacgymenvs.tasks import isaacgym_task_map


def get_rlgames_env_creator(
        # used to create the vec task
        seed: int,
        task_config: dict,
        task_name: str,
        sim_device: str,
        rl_device: str,
        graphics_device_id: int,
        headless: bool,
        # Used to handle multi-gpu case
        multi_gpu: bool = False,
        post_create_hook: Callable = None,
        virtual_screen_capture: bool = False,
        force_render: bool = False,
):
    """Parses the configuration parameters for the environment task and creates a VecTask

    Args:
        task_config: environment configuration.
        task_name: Name of the task, used to evaluate based on the imported name (eg 'Trifinger')
        sim_device: The type of env device, eg 'cuda:0'
        rl_device: Device that RL will be done on, eg 'cuda:0'
        graphics_device_id: Graphics device ID.
        headless: Whether to run in headless mode.
        multi_gpu: Whether to use multi gpu
        post_create_hook: Hooks to be called after environment creation.
            [Needed to setup WandB only for one of the RL Games instances when doing multiple GPUs]
        virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
        force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
    Returns:
        A VecTaskPython object.
    """
    def create_rlgpu_env():
        """
        Creates the task from configurations and wraps it using RL-games wrappers if required.
        """

        # create native task and pass custom config
        env = isaacgym_task_map[task_name](
            cfg=task_config,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        if post_create_hook is not None:
            post_create_hook()

        return env
    return create_rlgpu_env


class RLGPUAlgoObserver(AlgoObserver):
    """Allows us to log stats from the env along with the algorithm running stats. """

    def __init__(self, periodic_eval_cfg=None):
        self.periodic_eval_cfg = periodic_eval_cfg or {}
        self.periodic_eval_enabled = bool(self.periodic_eval_cfg.get("enabled", False))
        self.periodic_eval_interval = int(self.periodic_eval_cfg.get("interval", 0) or 0)
        self.periodic_eval_best_ckpt = str(self.periodic_eval_cfg.get("best_checkpoint_path", ""))
        self.periodic_eval_callback = self.periodic_eval_cfg.get("callback", None)
        self._last_periodic_eval_epoch = -1

    def after_init(self, algo):
        self.algo = algo
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer
        if self.periodic_eval_enabled:
            print(
                "[periodic-eval] enabled interval="
                f"{self.periodic_eval_interval}"
            )

    def process_infos(self, infos, done_indices):
        assert isinstance(infos, dict), "RLGPUAlgoObserver expects dict info"
        if isinstance(infos, dict):
            if 'episode' in infos:
                self.ep_infos.append(infos['episode'])

            if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
                self.direct_info = {}
                for k, v in infos.items():
                    # only log scalars
                    if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                        self.direct_info[k] = v

    def after_clear_stats(self):
        self.mean_scores.clear()

    def _run_periodic_eval(self, epoch_num):
        if not self.periodic_eval_enabled or self.periodic_eval_interval <= 0:
            return
        if epoch_num <= 0 or (epoch_num % self.periodic_eval_interval) != 0:
            return
        if self._last_periodic_eval_epoch == epoch_num:
            return
        self._last_periodic_eval_epoch = epoch_num

        best_ckpt = self.periodic_eval_best_ckpt
        if not best_ckpt or not os.path.isfile(best_ckpt):
            print(f"[periodic-eval] skip epoch={epoch_num}: missing best checkpoint {best_ckpt}")
            return

        if not callable(self.periodic_eval_callback):
            print(f"[periodic-eval] skip epoch={epoch_num}: no in-process callback configured")
            return

        try:
            self.periodic_eval_callback(epoch_num, best_ckpt, self.algo)
        except TypeError:
            # Backward compatibility for older 2-arg callbacks.
            self.periodic_eval_callback(epoch_num, best_ckpt)
        except Exception as exc:
            print(f"[periodic-eval] failed at epoch={epoch_num}: {exc}")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._run_periodic_eval(epoch_num)

        if self.ep_infos:
            for key in self.ep_infos[0]:
                    infotensor = torch.tensor([], device=self.algo.device)
                    for ep_info in self.ep_infos:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                    value = torch.mean(infotensor)
                    self.writer.add_scalar('Episode/' + key, value, epoch_num)
            self.ep_infos.clear()
        
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/iter', v, epoch_num)

        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar('scores/iter', mean_scores, epoch_num)


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.env = env_configurations.configurations[config_name]['env_creator'](**kwargs)

    def step(self, actions):
        return self.env.step(actions)

    def reset(self):
        return self.env.reset()
    
    def reset_done(self):
        return self.env.reset_done()

    def close(self):
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()

    def get_number_of_agents(self):
        return self.env.get_number_of_agents()

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space
        if hasattr(self.env, "amp_observation_space"):
            info['amp_observation_space'] = self.env.amp_observation_space

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info
