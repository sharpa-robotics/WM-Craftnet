# MIT License

# Copyright (c) 2023 NM512

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Additional WM-Craftnet modifications Copyright (c) 2026 The WM-Craftnet Authors.

import copy
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Dict, Optional

from . import tools
from . import networks

to_np = lambda x: x.detach().cpu().numpy()


def _build_mlp(in_dim: int, hidden_dims: list, out_dim: int, act=nn.ELU) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class WorldModel(nn.Module):
    def __init__(self, config, obs_shape, use_camera,
                 wm_cfg: Optional[Dict[str, Any]] = None,
                 tac_pred_n_links: int = 0,
                 obj_pred_dim: int = 0):
        super(WorldModel, self).__init__()
        # self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.device = self._config.device
        if wm_cfg is None:
            wm_cfg = {}

        self.encoder = networks.MultiEncoder(obs_shape, **config.encoder, use_camera=use_camera)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )
        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self._feat_size = feat_size
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, obs_shape, **config.decoder, use_camera=use_camera
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        # self.heads["cont"] = networks.MLP(...)
        for name in config.grad_heads:
            assert name in self.heads, name

        # ── WM decoder loss switches + aux heads ─────────────────────────────
        # Keep decoder modules built for checkpoint compatibility; ablations
        # only skip their corresponding reconstruction losses below.
        self.wm_prop_pred = bool(wm_cfg.get("wm_prop_pred", True))
        self.wm_depth_pred = bool(wm_cfg.get("wm_depth_pred", True))
        self.wm_pose_pred = bool(wm_cfg.get("wm_pose_pred", False))
        self.wm_tac_pred = bool(wm_cfg.get("wm_tac_pred", False)) and tac_pred_n_links > 0
        self.wm_value_pred = bool(wm_cfg.get("wm_value_pred", False))
        self.wm_obj_pred = bool(wm_cfg.get("wm_obj_pred", False)) and obj_pred_dim > 0
        self.tac_pred_n_links = int(tac_pred_n_links)
        self.obj_pred_dim = int(obj_pred_dim)
        self._wm_pos_loss_coef = float(wm_cfg.get("wm_pos_loss_coef", 0.001))
        self._wm_rot_loss_coef = float(wm_cfg.get("wm_rot_loss_coef", 0.01))
        self._wm_tac_loss_coef = float(wm_cfg.get("wm_tac_loss_coef", 0.001))
        self._wm_tac_pos_ema_decay = float(wm_cfg.get("wm_tac_pos_ema_decay", 0.01))
        self._wm_tac_pos_weight_min = float(wm_cfg.get("wm_tac_pos_weight_min", 1.0))
        self._wm_tac_pos_weight_max = float(wm_cfg.get("wm_tac_pos_weight_max", 20.0))
        self._wm_tac_pos_eps = float(wm_cfg.get("wm_tac_pos_eps", 1e-4))
        self._wm_tac_pos_rate_init = float(wm_cfg.get("wm_tac_pos_rate_init", 0.1))
        self.register_buffer(
            "_wm_tac_pos_rate_ema",
            torch.tensor(self._wm_tac_pos_rate_init, dtype=torch.float32),
            persistent=False,
        )
        self._wm_value_loss_coef = float(wm_cfg.get("wm_value_loss_coef", 0.001))
        self._wm_obj_loss_coef = float(wm_cfg.get("wm_obj_loss_coef", 0.01))

        if self.wm_pose_pred:
            pose_hidden = list(wm_cfg.get("wm_pose_pred_hidden", [256, 128]))
            self.wm_pose_head = _build_mlp(feat_size, pose_hidden, 9)
        else:
            self.wm_pose_head = None

        if self.wm_tac_pred:
            tac_hidden = list(wm_cfg.get("wm_tac_pred_hidden", [256, 128]))
            self.wm_tac_head = _build_mlp(feat_size, tac_hidden, self.tac_pred_n_links)
        else:
            self.wm_tac_head = None

        if self.wm_value_pred:
            value_hidden = list(wm_cfg.get("wm_value_pred_hidden", [256, 128]))
            self.wm_value_head = _build_mlp(feat_size, value_hidden, 1)
        else:
            self.wm_value_head = None

        if self.wm_obj_pred:
            obj_hidden = list(wm_cfg.get("wm_obj_pred_hidden", [256, 128]))
            self.wm_obj_head = _build_mlp(feat_size, obj_hidden, self.obj_pred_dim)
        else:
            self.wm_obj_head = None

        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        # can set different scale for terms in decoder here
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            image=1.0,
            # cont=config.cont_head["loss_scale"],
        )

    @staticmethod
    def _rot6d_geodesic_loss(pred_rot6d: torch.Tensor, target_rot6d: torch.Tensor) -> torch.Tensor:
        """Geodesic loss between two rotation matrices represented as 6D vectors.
        pred/target shape: (..., 6).  Returns per-sample scalar loss (...,).
        """
        def _gram_schmidt(v: torch.Tensor) -> torch.Tensor:
            a1 = v[..., :3]
            a2 = v[..., 3:6]
            b1 = F.normalize(a1, dim=-1)
            b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
            b3 = torch.cross(b1, b2, dim=-1)
            return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)

        R_pred = _gram_schmidt(pred_rot6d)
        R_tgt = _gram_schmidt(target_rot6d)
        R_diff = torch.matmul(R_pred.transpose(-1, -2), R_tgt)
        trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
        cos_angle = (trace - 1.0) / 2.0
        cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_angle)

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        wm_tac_pos_rate: Optional[torch.Tensor] = None
        wm_tac_pos_weight_metric: Optional[torch.Tensor] = None
        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                weighted_losses = {}
                for name, pred in preds.items():
                    if name == "prop" and not self.wm_prop_pred:
                        continue
                    if name == "image" and not self.wm_depth_pred:
                        continue
                    # Use clean (noise-free, near=1 far=0) image as decoder target
                    target_key = "image_clean" if name == "image" and "image_clean" in data else name
                    loss = -pred.log_prob(data[target_key])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss

                # ── WM auxiliary heads ──────────────────────────────────────
                feat_all = self.dynamics.get_feat(post)  # (B, T, feat_size)
                if self.wm_pose_pred and self.wm_pose_head is not None and "pose_target" in data:
                    pose_pred = self.wm_pose_head(feat_all)  # (B, T, 9)
                    pose_tgt = data["pose_target"]           # (B, T, 9)
                    pos_loss = (pose_pred[..., :3] - pose_tgt[..., :3]).pow(2).mean(-1)
                    rot_loss = self._rot6d_geodesic_loss(pose_pred[..., 3:], pose_tgt[..., 3:])
                    wm_pose_pos_loss = (self._wm_pos_loss_coef * pos_loss).mean()
                    wm_pose_rot_loss = (self._wm_rot_loss_coef * rot_loss).mean()
                    wm_pose_loss = wm_pose_pos_loss + wm_pose_rot_loss
                    model_loss = model_loss + wm_pose_loss
                    losses["wm_pose_pos"] = pos_loss
                    losses["wm_pose_rot"] = rot_loss
                    losses["wm_pose"] = pos_loss + rot_loss
                    weighted_losses["wm_pose_pos"] = wm_pose_pos_loss
                    weighted_losses["wm_pose_rot"] = wm_pose_rot_loss
                    weighted_losses["wm_pose"] = wm_pose_loss

                if self.wm_tac_pred and self.wm_tac_head is not None and "tac_contact_target" in data:
                    tac_logits = self.wm_tac_head(feat_all)  # (B, T, n_links)
                    tac_tgt = data["tac_contact_target"].to(dtype=tac_logits.dtype)
                    batch_pos_rate = tac_tgt.detach().float().mean()
                    decay = float(min(max(self._wm_tac_pos_ema_decay, 0.0), 1.0))
                    with torch.no_grad():
                        self._wm_tac_pos_rate_ema.mul_(1.0 - decay).add_(batch_pos_rate * decay)
                    pos_rate_ema = torch.clamp(
                        self._wm_tac_pos_rate_ema.to(device=tac_logits.device, dtype=tac_logits.dtype),
                        min=self._wm_tac_pos_eps,
                        max=1.0 - self._wm_tac_pos_eps,
                    )
                    dynamic_pos_weight = (1.0 - pos_rate_ema) / pos_rate_ema
                    dynamic_pos_weight = torch.clamp(
                        dynamic_pos_weight,
                        min=self._wm_tac_pos_weight_min,
                        max=self._wm_tac_pos_weight_max,
                    )
                    pos_weight = torch.full(
                        (self.tac_pred_n_links,),
                        float(dynamic_pos_weight.item()),
                        device=tac_logits.device,
                        dtype=tac_logits.dtype,
                    )
                    tac_loss = F.binary_cross_entropy_with_logits(
                        tac_logits, tac_tgt, reduction="none", pos_weight=pos_weight
                    ).mean(-1)
                    wm_tac_loss = (self._wm_tac_loss_coef * tac_loss).mean()
                    model_loss = model_loss + wm_tac_loss
                    losses["wm_tac"] = tac_loss  # for logging
                    weighted_losses["wm_tac"] = wm_tac_loss
                    wm_tac_pos_rate = batch_pos_rate
                    wm_tac_pos_weight_metric = dynamic_pos_weight.float()

                if self.wm_value_pred and self.wm_value_head is not None and "value_target" in data:
                    value_pred = self.wm_value_head(feat_all).squeeze(-1)  # (B, T)
                    value_tgt = data["value_target"].to(dtype=value_pred.dtype)
                    value_loss = F.smooth_l1_loss(value_pred, value_tgt, reduction="none")
                    wm_value_loss = (self._wm_value_loss_coef * value_loss).mean()
                    model_loss = model_loss + wm_value_loss
                    losses["wm_value"] = value_loss
                    weighted_losses["wm_value"] = wm_value_loss

                if self.wm_obj_pred and self.wm_obj_head is not None and "obj_shape_target" in data:
                    obj_pred = self.wm_obj_head(feat_all)  # (B, T, D)
                    obj_tgt = data["obj_shape_target"].to(dtype=obj_pred.dtype)  # (B, T, D)
                    obj_loss = F.smooth_l1_loss(obj_pred, obj_tgt, reduction="none").mean(-1)
                    wm_obj_loss = (self._wm_obj_loss_coef * obj_loss).mean()
                    model_loss = model_loss + wm_obj_loss
                    losses["wm_obj"] = obj_loss
                    weighted_losses["wm_obj"] = wm_obj_loss

            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss.mean()) for name, loss in losses.items()})
        metrics.update({
            f"{name}_weighted_loss": to_np(loss)
            for name, loss in weighted_losses.items()
        })
        if wm_tac_pos_rate is not None:
            metrics["wm_tac_pos_rate"] = to_np(wm_tac_pos_rate)
        if wm_tac_pos_weight_metric is not None:
            metrics["wm_tac_pos_weight"] = to_np(wm_tac_pos_weight_metric)
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs):
        # obs = obs.copy()
        # obs["image"] = torch.Tensor(obs["image"]) / 255.0

        # discount in obs seems useless
        # if "discount" in obs:
        #     obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            # obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        # assert "is_terminal" in obs
        # obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        result = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self._config.device)
            else:
                result[k] = torch.tensor(v, dtype=torch.float32).to(self._config.device)
        return result

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model], 2)
