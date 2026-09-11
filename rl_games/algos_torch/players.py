from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import torch 
from torch import nn
import numpy as np
import os
import time
from isaacgymenvs.utils.recorders.evaluation import Hdf5EpisodeRecorder

try:
    import h5py
except Exception:
    h5py = None

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

class PpoPlayerContinuous(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        self.record_hdf5 = bool(self.player_config.get('record_hdf5', False))
        self.record_hdf5_dir = self.player_config.get('record_hdf5_dir', 'test/hdf5_records')
        self.record_hdf5_episode_idx_base = int(self.player_config.get('record_hdf5_episode_idx_base', 0))
        self.hdf5_recorder = None
        self.replay_hdf5_path = self.player_config.get('replay_hdf5_path', '')
        self.replay_actions = None

        if (self.record_hdf5 or self.replay_hdf5_path) and h5py is None:
            raise RuntimeError("h5py is required for HDF5 record/replay, but it is not available in current environment.")
        if self.record_hdf5:
            self.hdf5_recorder = Hdf5EpisodeRecorder(
                self.record_hdf5_dir,
                episode_idx_base=self.record_hdf5_episode_idx_base,
            )

        if self.replay_hdf5_path:
            with h5py.File(self.replay_hdf5_path, 'r') as h5_file:
                if 'action' in h5_file:
                    actions = np.asarray(h5_file['action'], dtype=np.float32)
                    print(
                        "Replay is using dataset 'action'. "
                        "This is PD target joint-angle trajectory."
                    )
                elif 'policy_action' in h5_file:
                    actions = np.asarray(h5_file['policy_action'], dtype=np.float32)
                    print(
                        "Replay is using dataset 'policy_action'. "
                        "This is policy output and may differ from PD target replay."
                    )
                else:
                    raise KeyError(
                        f"HDF5 replay file missing dataset 'action'/'policy_action': {self.replay_hdf5_path}"
                    )
            if actions.ndim == 3:
                # [T, num_envs, action_dim] -> use env0 by default.
                actions = actions[:, 0, :]
            if actions.ndim != 2:
                raise ValueError(f"Unexpected replay action shape {actions.shape}, expected [T, action_dim].")
            if actions.shape[-1] != self.actions_num:
                raise ValueError(
                    f"Replay action dim mismatch: file has {actions.shape[-1]}, expected {self.actions_num}."
                )
            self.replay_actions = actions
            print(f"Loaded replay actions from {self.replay_hdf5_path}, steps={self.replay_actions.shape[0]}")

    def _resolve_wm_adapter(self):
        queue = [self.env]
        seen = set()
        while queue:
            candidate = queue.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            adapter = getattr(candidate, 'wm_adapter', None)
            if adapter is not None:
                return adapter
            for attr in ('env', 'task', 'vec_env'):
                nested = getattr(candidate, attr, None)
                if nested is not None:
                    queue.append(nested)
        return None

    def _extract_env0(self, tensor_like):
        if isinstance(tensor_like, torch.Tensor):
            array = tensor_like.detach().cpu().numpy()
        else:
            array = np.asarray(tensor_like)
        if array.ndim == 0:
            return np.asarray([array], dtype=np.float32)
        if array.ndim == 1:
            return array.astype(np.float32)
        return array[0].astype(np.float32)

    def _save_episode_hdf5(self, episode_idx, actions, qpos, dones, replay_actions=None, joint_names=None):
        if self.hdf5_recorder is None:
            return
        output_path = self.hdf5_recorder.save_episode(
            episode_idx,
            actions,
            qpos,
            dones,
            replay_actions=replay_actions,
            joint_names=joint_names,
        )
        if output_path:
            print(f"Saved episode trajectory to {output_path}")

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        # print(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        print("Loading checkpoint")
        self.model.load_state_dict(checkpoint['model'])
        if self.config.get('world_model_enabled', False):
            adapter = self._resolve_wm_adapter()
            if adapter is None:
                raise RuntimeError("WorldModel-enabled player could not find the task wm_adapter.")
            if 'world_model' not in checkpoint:
                raise RuntimeError("WM-Craftnet checkpoint does not contain world_model state.")
            adapter.load_state_dict(checkpoint['world_model'], load_optimizer=False)
            print("Loaded world model from player checkpoint")
        if self.normalize_input: 
            if 'running_mean_std.running_mean_std.obs.running_mean' in checkpoint['model']:
                self.model.running_mean_std.running_mean_std.obs.count.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.obs.count'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.obs.running_mean.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.obs.running_mean'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.obs.running_var.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.obs.running_var'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.pointcloud.count.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.pointcloud.count'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.pointcloud.running_mean.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.pointcloud.running_mean'].data).to(self.device)
                self.model.running_mean_std.running_mean_std.pointcloud.running_var.data = (
                    checkpoint['model']['running_mean_std.running_mean_std.pointcloud.running_var'].data).to(self.device)

            else:
                self.model.running_mean_std.count.data = (
                    checkpoint['model']['running_mean_std.count'].data).to(self.device)
                self.model.running_mean_std.running_mean.data = (
                    checkpoint['model']['running_mean_std.running_mean'].data).to(self.device)
                self.model.running_mean_std.running_var.data = (
                    checkpoint['model']['running_mean_std.running_var'].data).to(self.device)
    def reset(self):
        self.init_rnn()

    def _task_in_standalone_test_eval(self):
        """True if the underlying task is running its standalone test-eval
        lifecycle. The task owns the exit decision in that mode; the player
        must yield control of the loop to it (no games_num cap, no force
        reset between games)."""
        for candidate in (
            self.env,
            getattr(self.env, "env", None),
            getattr(self.env, "task", None),
            getattr(getattr(self.env, "env", None), "task", None),
        ):
            if candidate is None:
                continue
            if bool(getattr(candidate, "_test_eval_active", False)):
                return True
        return False

    def _run_standalone_test_eval(self):
        """Loop driver for standalone test runs.

        Termination is owned entirely by the task: it raises SystemExit from
        post_physics_step once all envs have completed their one eval
        episode. Here we just keep stepping the policy until that happens.
        We do NOT call env_reset() inside the loop (force-reset would break
        the per-env pending->recording state machine); the env was already
        reset implicitly before this method runs.
        """
        is_deterministic = self.is_deterministic
        if self.is_rnn:
            self.init_rnn()
        obses = self.env_reset(self.env)
        batch_size = self.get_batch_size(obses, 1)
        # Cosmetic: keep a single line trickling out so the user can see we
        # are alive while the task drains its eval window.
        step_idx = 0
        while True:
            action = self.get_action(obses, is_deterministic)
            obses, _r, _done, _info = self.env_step(self.env, action)
            step_idx += 1
            if step_idx % 256 == 0 and self.print_stats:
                print(f"[test-eval][player] running, steps={step_idx}")
            _ = batch_size

    def run(self):
        # Test-eval mode (driven entirely by the task): the task owns the
        # exit decision (it raises SystemExit once every env has finished one
        # eval episode), so this loop must NOT terminate based on games_num
        # and must NOT call env_reset() at the start of every "game" (that
        # would force-reset all envs and break the per-env pending->recording
        # state machine). Detect the mode once and dispatch to the dedicated
        # loop.
        if self._task_in_standalone_test_eval():
            return self._run_standalone_test_eval()

        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        is_openloop_trajectory_enough = False
        openloop_trajectory_saved = False

        env_dones = np.ones(16) * 2
        env_reward = np.zeros(16)
        end_flag = False

        # Infer object class count from live env/task instead of hard-coded objSet mapping.
        # Fallback remains safe: allocate at least one slot, and grow on demand below.
        n_object = None
        obj_set_key = self.env.cfg['env'].get('objSet', None) if isinstance(getattr(self.env, 'cfg', None), dict) else None
        candidate_objs = [
            self.env,
            getattr(self.env, 'env', None),
            getattr(self.env, 'task', None),
            getattr(getattr(self.env, 'env', None), 'task', None),
            getattr(self.env, '_env', None),
        ]
        for candidate in candidate_objs:
            if candidate is None:
                continue
            if hasattr(candidate, 'num_training_objects'):
                try:
                    n_object = int(getattr(candidate, 'num_training_objects'))
                except Exception:
                    n_object = None
            if n_object is None and hasattr(candidate, 'used_training_objects'):
                try:
                    n_object = len(getattr(candidate, 'used_training_objects'))
                except Exception:
                    n_object = None
            if n_object is None and hasattr(candidate, 'object_sets') and obj_set_key is not None:
                try:
                    object_sets = getattr(candidate, 'object_sets')
                    n_object = len(object_sets[str(obj_set_key)])
                except Exception:
                    n_object = None
            if n_object is not None and n_object > 0:
                break
        if n_object is None or n_object <= 0:
            n_object = 1

        object_count_dict = {
            'game_count': np.zeros(n_object, dtype=np.float64),
            'reward_sum': np.zeros(n_object, dtype=np.float64),
            'step_sum': np.zeros(n_object, dtype=np.float64),
        }
        print(n_games, self.max_steps, self.n_game_life)

        for game_idx in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            open_loop_trajectories_action = []
            open_loop_trajectories_done = []
            open_loop_trajectories_state = []
            episode_actions = []
            episode_qpos = []
            episode_dones = []
            episode_replay_actions = []
            episode_joint_names = None
            replay_idx = 0
            replay_exhausted = False

            for n in range(self.max_steps):
                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    if self.replay_actions is not None:
                        if replay_idx >= self.replay_actions.shape[0]:
                            replay_exhausted = True
                            break
                        replay_action = torch.from_numpy(self.replay_actions[replay_idx]).to(self.device)
                        action = replay_action.unsqueeze(0).repeat(batch_size, 1)
                        replay_idx += 1
                    else:
                        action = self.get_action(obses, is_deterministic)

                next_obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                if self.record_hdf5:
                    qpos = self.get_env_internal_info(self.env, 'qpos')
                    pd_target = self.get_env_internal_info(self.env, 'target')
                    if episode_joint_names is None:
                        episode_joint_names = self.get_env_internal_info(self.env, 'joint_names')
                        if episode_joint_names is None:
                            episode_joint_names = []
                    episode_actions.append(self._extract_env0(pd_target))
                    episode_qpos.append(self._extract_env0(qpos))
                    episode_dones.append(bool(self._extract_env0(done)[0] > 0.5))
                    episode_replay_actions.append(self._extract_env0(action))

                # Record some open loop trajectories.
                if not is_openloop_trajectory_enough:
                    open_loop_trajectories_action.append(action) # [num_envs, action_dim]
                    open_loop_trajectories_done.append(done)
                    open_loop_trajectories_state.append(obses)

                if len(open_loop_trajectories_action) > 400:
                    is_openloop_trajectory_enough = True

                # if is_openloop_trajectory_enough:
                #     if not openloop_trajectory_saved:
                #         open_loop_trajectories_action = torch.stack(open_loop_trajectories_action, dim=0)
                #         open_loop_trajectories_done = torch.stack(open_loop_trajectories_done, dim=0)
                #         open_loop_trajectories_state = torch.stack(open_loop_trajectories_state, dim=0)

                        # pickle_utils.save_data({'obs': open_loop_trajectories_state.detach().cpu().numpy(),
                        #                         'act': open_loop_trajectories_action.detach().cpu().numpy(),
                        #                         'done': open_loop_trajectories_done.detach().cpu().numpy()},
                        #                         self.action_savepath)

                        #print("Data saved.")

                    openloop_trajectory_saved = True

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                obses = next_obses
                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                default_mode = True
                if done_count > 0:
                    if default_mode:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:,all_done_indices, :] * 0.0

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        # Fetch object information
                        obj_idx_info = self.env.get_internal_info('obj')
                        for done_index in done_indices:
                            obj_idx = int(obj_idx_info[done_index][:, 0].item())
                            current_size = object_count_dict['game_count'].shape[0]
                            if obj_idx >= current_size:
                                grow_size = obj_idx + 1 - current_size
                                object_count_dict['game_count'] = np.pad(object_count_dict['game_count'], (0, grow_size))
                                object_count_dict['reward_sum'] = np.pad(object_count_dict['reward_sum'], (0, grow_size))
                                object_count_dict['step_sum'] = np.pad(object_count_dict['step_sum'], (0, grow_size))
                            object_count_dict['game_count'][obj_idx] += 1
                            object_count_dict['reward_sum'][obj_idx] += cr[done_index].item()
                            object_count_dict['step_sum'][obj_idx] += steps[done_index].item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if 'battle_won' in info:
                                print_game_res = True
                                game_res = info.get('battle_won', 0.5)
                            if 'scores' in info:
                                print_game_res = True
                                game_res = info.get('scores', 0.5)

                        if self.print_stats:
                            if print_game_res:
                                print('reward:', cur_rewards/done_count,
                                      'steps:', cur_steps/done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards/done_count,
                                      'steps:', cur_steps/done_count)

                        sum_game_res += game_res
                        print(games_played, object_count_dict['game_count'])
                        if batch_size//self.num_agents == 1 or games_played >= n_games:
                            break
                    else:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                        for d in list(done_indices):
                            if env_dones[d] <= 0:
                                continue
                            else:
                                env_reward[d] += cr[d].item()
                                env_dones[d] -= 1

                        if len(np.where(env_dones > 0)[0]) < 0:
                            break_flag = True
                        else:
                            break_flag = False

                        if break_flag:
                            end_flag = True
                            break

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if self.print_stats:
                            if print_game_res:
                                print(cur_steps)
                                print('reward:', cur_rewards / done_count,
                                      'steps:', cur_steps / done_count, 'w:', game_res)
                            else:
                                print('reward:', cur_rewards / done_count,
                                      'steps:', cur_steps / done_count)

                        sum_game_res += game_res
                        if batch_size // self.num_agents == 1 or games_played >= n_games:
                            break
            if self.record_hdf5 and len(episode_actions) > 0:
                self._save_episode_hdf5(
                    game_idx,
                    episode_actions,
                    episode_qpos,
                    episode_dones,
                    replay_actions=episode_replay_actions,
                    joint_names=episode_joint_names,
                )
            if replay_exhausted:
                print("Replay action sequence exhausted before env reset; stopping play loop.")
                break
            if end_flag:
                break

        mask = np.where(object_count_dict['game_count']>0, 1, 0)
        av_rew = np.where(object_count_dict['game_count']>0, object_count_dict['reward_sum']/object_count_dict['game_count'], 0)
        av_step = np.where(object_count_dict['game_count']>0, object_count_dict['step_sum']/object_count_dict['game_count'], 0)
        print("av of av rewards:", np.average(av_rew, weights=mask),
            "av of av steps:", np.average(av_step, weights=mask))
        print("ALL", env_reward.sum() / 32)
        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

class PpoPlayerContinuousCollect(BasePlayer):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def reset(self):
        self.init_rnn()
        self.collect_trajectories_action = None
        self.collect_trajectories_done = None
        self.collect_trajectories_state = None
        if self.is_rnn:
            self.collect_trajectories_rnn = None
        self.collect_trajectories_gt = None

        self.collect_trajectories_qpos = None
        self.collect_trajectories_target = None
        self.collect_trajectories_contact = None
        self.collect_trajectories_obj_class = None

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn

        collect_trajectories_action = []
        collect_trajectories_done = []
        collect_trajectories_state = []
        if self.is_rnn:
            collect_trajectories_rnn = []
        collect_trajectories_gt = []
        collect_trajectories_target = []
        collect_trajectories_qpos = []
        collect_trajectories_contact = []
        collect_trajectories_obj_class = []
        collect_trajectories_init_quat = []

        for game_idx in range(n_games):
            if games_played >= n_games:
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if self.is_rnn:
                    current_rnn_state = torch.cat(self.states, dim=0)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)

                groundtruth = self.get_env_internal_state(self.env)
                next_obses, r, done, info = self.env_step(self.env, action)

                qpos = self.get_env_internal_info(self.env, 'qpos')
                # print(qpos)
                target = self.get_env_internal_info(self.env, 'target')
                contact = self.get_env_internal_info(self.env, 'contact')
                obj_class = self.get_env_internal_info(self.env, 'obj')
                init_quat = self.get_env_internal_info(self.env, 'qinit')
                # This will record the execution trajectory....

                cr += r
                steps += 1

                # Record some open loop trajectories.
                collect_trajectories_action.append(action.detach().cpu().numpy())  # [num_envs, action_dim]
                collect_trajectories_done.append(done.detach().cpu().numpy())
                if self.is_rnn:
                    collect_trajectories_rnn.append(current_rnn_state.detach().cpu().numpy())
                collect_trajectories_state.append(obses.detach().cpu().numpy())
                collect_trajectories_gt.append(groundtruth.detach().cpu().numpy())
                collect_trajectories_qpos.append(qpos.detach().cpu().numpy())
                collect_trajectories_target.append(target.detach().cpu().numpy())
                collect_trajectories_contact.append(contact.detach().cpu().numpy())
                collect_trajectories_obj_class.append(obj_class.detach().cpu().numpy())
                collect_trajectories_init_quat.append(init_quat.detach().cpu().numpy())

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                obses = next_obses
                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if self.print_stats:
                        if print_game_res:
                            print(cur_steps)
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards / done_count,
                                  'steps:', cur_steps / done_count)

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        self.collect_trajectories_action = np.stack(collect_trajectories_action, axis=0)
        self.collect_trajectories_done = np.stack(collect_trajectories_done, axis=0)
        self.collect_trajectories_state = np.stack(collect_trajectories_state, axis=0)
        if self.is_rnn:
            self.collect_trajectories_rnn = np.stack(collect_trajectories_rnn, axis=0)
        self.collect_trajectories_gt = np.stack(collect_trajectories_gt, axis=0)

        self.collect_trajectories_qpos = np.stack(collect_trajectories_qpos, axis=0)
        self.collect_trajectories_target = np.stack(collect_trajectories_target, axis=0)
        self.collect_trajectories_contact = np.stack(collect_trajectories_contact, axis=0)
        self.collect_trajectories_obj_class = np.stack(collect_trajectories_obj_class, axis=0)
        self.collect_trajectories_init_quat = np.stack(collect_trajectories_init_quat, axis=0)

        print(self.collect_trajectories_state.shape, self.collect_trajectories_action.shape,
              self.collect_trajectories_gt.shape, self.collect_trajectories_done.shape,
              self.collect_trajectories_qpos.shape,
              self.collect_trajectories_target.shape)

        print(sum_rewards)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)

    def post_run(self, runner):
        print("Player post-running procedure, saving record data.")
        if self.is_rnn:
            runner.set_record_data(state=self.collect_trajectories_state,
                                   action=self.collect_trajectories_action,
                                   gt=self.collect_trajectories_gt,
                                   done=self.collect_trajectories_done,
                                   rnn=self.collect_trajectories_rnn,
                                   qpos=self.collect_trajectories_qpos,
                                   target=self.collect_trajectories_target,
                                   contact=self.collect_trajectories_contact,
                                   qinit=self.collect_trajectories_init_quat)
        else:
            runner.set_record_data(state=self.collect_trajectories_state,
                                   action=self.collect_trajectories_action,
                                   gt=self.collect_trajectories_gt,
                                   done=self.collect_trajectories_done,
                                   qpos=self.collect_trajectories_qpos,
                                   target=self.collect_trajectories_target,
                                   contact=self.collect_trajectories_contact,
                                   obj=self.collect_trajectories_obj_class,
                                   qinit=self.collect_trajectories_init_quat)
        return
