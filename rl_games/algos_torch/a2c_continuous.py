from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext

from rl_games.algos_torch import central_value
from rl_games.common import common_losses
from rl_games.common import datasets

from torch import optim
import torch 
from torch import nn
import torch.nn.functional as F
import numpy as np
import gym

class A2CAgent(a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, params):
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        obs_shape = self.obs_shape
        self.distill = False
        main_shape = obs_shape
        build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : main_shape,
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.use_l1 = False
        policy_net = getattr(self.model, 'a2c_network', None)
        self.pos_loss_coef = float(getattr(policy_net, 'pos_loss_coef', 0.0))
        self.rot_loss_coef = float(getattr(policy_net, 'rot_loss_coef', 0.0))
        self.tac_pred_loss_coef = float(getattr(policy_net, 'tac_pred_loss_coef', 0.0))

        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, 
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_len' : self.seq_len,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer,
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
            }
            self.central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_len)
        if self.normalize_value:
            self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std

        self.has_value_loss = (self.has_central_value and self.use_experimental_cv) \
                            or (not self.has_phasic_policy_gradients and not self.has_central_value) 
        self.algo_observer.after_init(self)

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num
        
    def save(self, fn):
        state = self.get_full_state_weights()
        torch_ext.save_checkpoint(fn, state)

    def restore(self, fn, mode='student'):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.set_full_state_weights(checkpoint, mode)

    def get_masked_action_values(self, obs, action_masks):
        assert False

    def recon_criterion(self, out, target):
        if self.use_l1:
            return torch.abs(out - target)
        else:
            return (out - target).pow(2)

    @staticmethod
    def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        a1 = rot6d[:, 0:3]
        a2 = rot6d[:, 3:6]
        b1 = F.normalize(a1, dim=-1, eps=1e-8)
        b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1, eps=1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-1)

    @staticmethod
    def _rotation_matrix_geodesic_angle(pred_rot_mats: torch.Tensor, target_rot_mats: torch.Tensor) -> torch.Tensor:
        # Relative rotation: R_rel = R_pred^T * R_target
        rel = torch.matmul(pred_rot_mats.transpose(1, 2), target_rot_mats)
        trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
        cos_theta = 0.5 * (trace - 1.0)
        cos_theta = torch.clamp(cos_theta, min=-1.0 + 1e-6, max=1.0 - 1e-6)
        return torch.acos(cos_theta)

    @classmethod
    def _rot6d_geodesic_loss(cls, pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor) -> torch.Tensor:
        pred_rot_mats = cls._rot6d_to_matrix(pred_rot6d)
        target_rot_mats = cls._rot6d_to_matrix(target_rot6d)
        return cls._rotation_matrix_geodesic_angle(pred_rot_mats, target_rot_mats)
    
    def calc_gradients(self, input_dict):
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        if self.distill:
            assert isinstance(input_dict['obs'], dict)
            obs_batch = {'obs': input_dict['obs']['student_obs'], 'pointcloud': input_dict['obs']['pointcloud']}  # input_dict['obs']['student_obs']
            teacher_obs_batch = input_dict['obs']['obs']
            teacher_res_dict = self.get_teacher_action_values(input_dict)
            teacher_actions_batch = teacher_res_dict['actions']
        else:
            obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)

        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'obs' : obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            # print("GET RNN STATES:", batch_dict['rnn_states'][0].shape)
            batch_dict['seq_length'] = self.seq_len
            batch_dict['bptt_len'] = self.bptt_len
            batch_dict['dones'] = input_dict['dones']
            
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']
            pose_pred = res_dict.get('pose_pred', None)
            tac_pred_logits = res_dict.get('tac_pred_logits', None)

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            if self.has_value_loss:
                c_loss = common_losses.critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            pose_pred_pos_loss = torch.zeros(1, device=self.ppo_device)
            pose_pred_rot_loss = torch.zeros(1, device=self.ppo_device)
            pose_pred_rot_deg = torch.zeros(1, device=self.ppo_device)
            tac_pred_loss = torch.zeros(1, device=self.ppo_device)
            tac_pred_acc = torch.zeros(1, device=self.ppo_device)
            if (
                pose_pred is not None
                and isinstance(obs_batch, dict)
                and 'pose_target' in obs_batch
                and (self.pos_loss_coef != 0.0 or self.rot_loss_coef != 0.0)
            ):
                pose_target = obs_batch['pose_target']
                pos_loss_raw = (pose_pred[:, :3] - pose_target[:, :3]).pow(2).mean(dim=-1)
                rot_loss_raw = self._rot6d_geodesic_loss(pose_pred[:, 3:], pose_target[:, 3:])
                rot_deg_raw = rot_loss_raw * (180.0 / np.pi)
                pose_losses, _ = torch_ext.apply_masks(
                    [pos_loss_raw.unsqueeze(1), rot_loss_raw.unsqueeze(1), rot_deg_raw.unsqueeze(1)],
                    rnn_masks,
                )
                pose_pred_pos_loss, pose_pred_rot_loss, pose_pred_rot_deg = pose_losses[0], pose_losses[1], pose_losses[2]
            pose_pred_weighted_loss = self.pos_loss_coef * pose_pred_pos_loss + self.rot_loss_coef * pose_pred_rot_loss

            if (
                tac_pred_logits is not None
                and isinstance(obs_batch, dict)
                and 'tac_contact_target' in obs_batch
                and self.tac_pred_loss_coef != 0.0
            ):
                tac_target = obs_batch['tac_contact_target'].to(dtype=tac_pred_logits.dtype)
                tac_loss_raw = F.binary_cross_entropy_with_logits(
                    tac_pred_logits,
                    tac_target,
                    reduction='none',
                ).mean(dim=-1)
                tac_pred_binary = (torch.sigmoid(tac_pred_logits) >= 0.5).to(dtype=tac_target.dtype)
                tac_acc_raw = (tac_pred_binary == tac_target).to(dtype=tac_target.dtype).mean(dim=-1)
                tac_losses, _ = torch_ext.apply_masks(
                    [tac_loss_raw.unsqueeze(1), tac_acc_raw.unsqueeze(1)],
                    rnn_masks,
                )
                tac_pred_loss, tac_pred_acc = tac_losses[0], tac_losses[1]
            tac_pred_weighted_loss = self.tac_pred_loss_coef * tac_pred_loss
            if self.distill:
                bc_loss = self.recon_criterion(torch.clamp(mu, -1, 1),
                                               torch.clamp(teacher_actions_batch, -1, 1)).mean()
                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                    + bc_loss * self.bc_loss_coef
                    + pose_pred_weighted_loss
                    + tac_pred_weighted_loss
                )
            else:
                loss = (
                    a_loss
                    + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef
                    + b_loss * self.bounds_loss_coef
                    + pose_pred_weighted_loss
                    + tac_pred_weighted_loss
                )
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss, pose_pred_pos_loss, pose_pred_rot_loss, pose_pred_weighted_loss, pose_pred_rot_deg, tac_pred_loss, tac_pred_acc)

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result

    def reg_loss(self, mu):
        if self.bounds_loss_coef is not None:
            reg_loss = (mu*mu).sum(axis=-1)
        else:
            reg_loss = 0
        return reg_loss

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss


