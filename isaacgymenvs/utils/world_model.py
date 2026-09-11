# Copyright (c) 2026 The WM-Craftnet Authors
# SPDX-License-Identifier: Apache-2.0

"""World Synesthesia Model adapter for WM-Craftnet.

Owns three responsibilities:
  1. Build the WSM ``WorldModel`` from the task config and WSM defaults.
  2. Maintain per-env recurrent state (``wm_latent``, ``prev_action``,
     ``is_first``) and expose a ``step`` that produces the deter feature ``h``.
  3. Hold a ring replay of ``(prop, image, image_clean, action, reward,
     is_first, pose_target, tac_contact_target)`` rows collected during
     rollouts, and sample contiguous time chunks for training.

The deter feature ``h`` is the only thing exposed to the actor (mirrors WMP's
``ActorCriticWMP`` integration). The actor-side MLP that consumes ``h`` lives
in ``rl_games/algos_torch/network_builder.py``.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# wsm/ is at the repository root and is directly importable.
from wsm.models import WorldModel


_WSM_CONFIG_REL = "wsm/configs.yaml"
_WM_BODY_RESUME_PREFIXES = (
    "encoder.",
    "dynamics.",
    "heads.decoder.",
    "heads.reward.",
)


def _load_wsm_defaults(repo_root: pathlib.Path) -> Dict[str, Any]:
    cfg_path = repo_root / _WSM_CONFIG_REL
    text = cfg_path.read_text()
    raw = yaml.safe_load(text)
    return dict(raw["defaults"])


def build_wm_config(
    repo_root: pathlib.Path,
    device: str,
    num_actions: int,
    overrides: Dict[str, Any],
) -> SimpleNamespace:
    """Merge WSM defaults with task-level overrides into a SimpleNamespace.

    Avoids the upstream argparse-on-sys.argv hack used by WMP's WMPRunner,
    which is incompatible with hydra-driven entrypoints.
    """
    base = _load_wsm_defaults(repo_root)
    base["device"] = device
    base["num_actions"] = int(num_actions)
    allowed = {
        "dyn_deter",
        "dyn_stoch",
        "dyn_discrete",
        "dyn_hidden",
        "dyn_rec_depth",
        "model_lr",
        "grad_clip",
        "batch_size",
        "batch_length",
        "train_steps_per_iter",
        "train_start_steps",
        "kl_free",
        "dyn_scale",
        "rep_scale",
        "weight_decay",
        "opt",
        "opt_eps",
        "unimix_ratio",
        "initial",
    }
    for k, v in overrides.items():
        if k in allowed:
            base[k] = v
    # Coerce numeric strings (yaml may load "1e-4" as a string under some
    # loaders) to floats so the WSM optimizer can compare them.
    _float_keys = ("model_lr", "opt_eps", "grad_clip", "kl_free", "dyn_scale", "rep_scale", "weight_decay", "dyn_min_std")
    for k in _float_keys:
        if k in base:
            try:
                base[k] = float(base[k])
            except (TypeError, ValueError):
                pass
    _int_keys = ("dyn_deter", "dyn_stoch", "dyn_discrete", "dyn_hidden", "dyn_rec_depth", "batch_size", "batch_length", "train_steps_per_iter", "train_start_steps", "pretrain", "train_ratio", "units", "precision")
    for k in _int_keys:
        if k in base and not isinstance(base[k], bool):
            try:
                base[k] = int(base[k])
            except (TypeError, ValueError):
                pass
    return SimpleNamespace(**base)


class WorldModelAdapter:
    """Owns the WorldModel, per-env recurrent state and the ring replay."""

    def __init__(
        self,
        repo_root: pathlib.Path,
        num_envs: int,
        num_actions: int,
        prop_dim: int,
        image_shape: Tuple[int, int, int],
        wm_cfg: Dict[str, Any],
        device: torch.device,
        tac_pred_n_links: int = 0,
        obj_pred_dim: int = 0,
    ) -> None:
        self.repo_root = pathlib.Path(repo_root)
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.prop_dim = int(prop_dim)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.device = torch.device(device)
        self.image_on_cpu = bool(wm_cfg.get("image_on_cpu", True))

        self.wm_config = build_wm_config(
            repo_root=repo_root,
            device=str(self.device),
            num_actions=self.num_actions,
            overrides=wm_cfg,
        )
        obs_shape = {"prop": (self.prop_dim,), "image": self.image_shape}
        self.world_model = WorldModel(
            self.wm_config,
            obs_shape,
            use_camera=True,
            wm_cfg=wm_cfg,
            tac_pred_n_links=tac_pred_n_links,
            obj_pred_dim=obj_pred_dim,
        ).to(self.device)
        self.wm_feature_dim = int(self.wm_config.dyn_deter)

        self._wm_latent: Optional[Dict[str, torch.Tensor]] = None
        self._wm_prev_action = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self._wm_is_first = torch.ones((self.num_envs,), device=self.device)
        self._wm_feature = torch.zeros((self.num_envs, self.wm_feature_dim), device=self.device)

        self.replay_capacity = int(wm_cfg.get("replay_capacity", 4096))
        self.train_after_warmup_steps = int(wm_cfg.get("train_after_warmup_steps", 10000))
        self.train_steps_per_epoch = int(wm_cfg.get("train_steps_per_epoch", 10))
        self.batch_size = int(wm_cfg.get("batch_size", 16))
        self.batch_length = int(wm_cfg.get("batch_length", 32))
        self.tac_pred_n_links = int(tac_pred_n_links)
        self.obj_pred_dim = int(obj_pred_dim)

        # Ring replay tensors. prop/action/reward/is_first stay on the wm
        # device; depth is optionally kept on CPU since it dominates VRAM.
        img_device = torch.device("cpu") if self.image_on_cpu else self.device
        H, W, C = self.image_shape
        self.ring: Dict[str, torch.Tensor] = {
            "prop": torch.zeros((self.replay_capacity, self.num_envs, self.prop_dim), device=self.device),
            "image": torch.zeros((self.replay_capacity, self.num_envs, H, W, C), device=img_device),
            # clean image: crop-only, no noise, near=1 far=0 (WM decoder supervision target)
            "image_clean": torch.zeros((self.replay_capacity, self.num_envs, H, W, C), device=img_device),
            "action": torch.zeros((self.replay_capacity, self.num_envs, self.num_actions), device=self.device),
            "reward": torch.zeros((self.replay_capacity, self.num_envs), device=self.device),
            "is_first": torch.zeros((self.replay_capacity, self.num_envs), device=self.device),
            # aux supervision targets for WM pose/tac heads
            "pose_target": torch.zeros((self.replay_capacity, self.num_envs, 9), device=self.device),
            "tac_contact_target": torch.zeros(
                (self.replay_capacity, self.num_envs, max(1, self.tac_pred_n_links)), device=self.device
            ),
            # aux supervision targets for WM value/object heads
            "value_target": torch.zeros((self.replay_capacity, self.num_envs), device=self.device),
            "obj_shape_target": torch.zeros(
                (self.replay_capacity, self.num_envs, max(1, self.obj_pred_dim)), device=self.device
            ),
        }
        self._write_idx = 0
        self._last_append_idx: Optional[int] = None
        self._filled = 0  # total contiguous rows written (clamped by capacity)
        self._total_appended = 0

        resume_body_from = str(wm_cfg.get("resume_body_from", "") or "").strip()
        if resume_body_from:
            self.load_body_from_checkpoint(resume_body_from)

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def step(self, prop: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """Run one obs_step. Returns deter feature ``h`` (num_envs, dyn_deter)."""
        prop = prop.to(self.device, non_blocking=True)
        image = image.to(self.device, non_blocking=True)
        wm_obs = {
            "prop": prop,
            "image": image,
            "is_first": self._wm_is_first,
        }
        embed = self.world_model.encoder(wm_obs)
        self._wm_latent, _ = self.world_model.dynamics.obs_step(
            self._wm_latent, self._wm_prev_action, embed, self._wm_is_first
        )
        self._wm_feature = self.world_model.dynamics.get_deter_feat(self._wm_latent)
        # After the first obs_step of an episode for an env, future steps are
        # not "first" anymore. RSSM's obs_step uses this flag to reset state.
        self._wm_is_first.zero_()
        return self._wm_feature

    @torch.no_grad()
    def decode_posterior_image(self) -> Optional[torch.Tensor]:
        """Decode the latest posterior RSSM state into an image prediction."""
        if self._wm_latent is None:
            return None
        feat = self.world_model.dynamics.get_feat(self._wm_latent)
        squeeze_time = False
        if feat.ndim == 2:
            feat = feat.unsqueeze(1)
            squeeze_time = True
        pred = self.world_model.heads["decoder"](feat).get("image", None)
        if pred is None:
            return None
        image = pred.mode()
        if squeeze_time:
            image = image.squeeze(1)
        return image.detach()

    def cache_prev_action(self, actions: torch.Tensor) -> None:
        self._wm_prev_action = actions.detach().to(self.device, non_blocking=True)

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        if env_ids is None:
            return
        if isinstance(env_ids, np.ndarray):
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        else:
            env_ids_t = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        if env_ids_t.numel() == 0:
            return
        self._wm_is_first[env_ids_t] = 1.0
        self._wm_prev_action[env_ids_t] = 0.0
        # On reset, RSSM.obs_step rebuilds the state from `initial()` for
        # rows where is_first=1. We still zero our cached deter feature so
        # the actor reads zeros until the next forward.
        self._wm_feature[env_ids_t] = 0.0
        if self._wm_latent is not None:
            for k, v in self._wm_latent.items():
                v[env_ids_t] = 0.0

    @property
    def feature(self) -> torch.Tensor:
        return self._wm_feature

    # --------------------------------------------------------------- ring replay
    def append(
        self,
        prop: torch.Tensor,
        image: torch.Tensor,
        image_clean: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        is_first: torch.Tensor,
        pose_target: Optional[torch.Tensor] = None,
        tac_contact_target: Optional[torch.Tensor] = None,
        value_target: Optional[torch.Tensor] = None,
        obj_shape_target: Optional[torch.Tensor] = None,
    ) -> None:
        idx = self._write_idx
        self.ring["prop"][idx] = prop.detach().to(self.device, non_blocking=True)
        img_device = "cpu" if self.image_on_cpu else self.device
        self.ring["image"][idx] = image.detach().to(img_device, non_blocking=True)
        self.ring["image_clean"][idx] = image_clean.detach().to(img_device, non_blocking=True)
        self.ring["action"][idx] = action.detach().to(self.device, non_blocking=True)
        self.ring["reward"][idx] = reward.detach().to(self.device, non_blocking=True).view(-1)
        self.ring["is_first"][idx] = is_first.detach().to(self.device, non_blocking=True).float().view(-1)
        if pose_target is not None:
            self.ring["pose_target"][idx] = pose_target.detach().to(self.device, non_blocking=True)
        if tac_contact_target is not None and self.tac_pred_n_links > 0:
            self.ring["tac_contact_target"][idx] = tac_contact_target.detach().to(self.device, non_blocking=True)
        if value_target is not None:
            self.ring["value_target"][idx] = value_target.detach().to(self.device, non_blocking=True).view(-1)
        if obj_shape_target is not None and self.obj_pred_dim > 0:
            self.ring["obj_shape_target"][idx] = (
                obj_shape_target.detach().to(self.device, non_blocking=True).view(self.num_envs, self.obj_pred_dim)
            )
        self._last_append_idx = idx
        self._write_idx = (idx + 1) % self.replay_capacity
        self._filled = min(self._filled + 1, self.replay_capacity)
        self._total_appended += self.num_envs

    def set_last_value_target(self, value_target: torch.Tensor) -> None:
        if self._filled == 0 or self._last_append_idx is None:
            return
        self.ring["value_target"][self._last_append_idx] = (
            value_target.detach().to(self.device, non_blocking=True).view(-1)
        )

    def filled_steps(self) -> int:
        """Total transitions in the ring (rows * num_envs)."""
        return self._filled * self.num_envs

    def total_appended(self) -> int:
        return self._total_appended

    def can_train(self) -> bool:
        return (
            self._filled >= self.batch_length + 1
            and self._total_appended >= self.train_after_warmup_steps
        )

    def sample_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a contiguous (batch_size, batch_length) window per env.

        Returns a dict ready to feed ``WorldModel._train``. None if the ring
        hasn't accumulated enough contiguous frames yet.
        """
        if self._filled < self.batch_length + 1:
            return None
        B = self.batch_size
        T = self.batch_length
        # Per-batch-element pick an env id and a window start. We sample only
        # from the "filled" prefix to avoid crossing the ring write pointer
        # (which would mix old & new transitions). This drops a small fraction
        # of valid windows but keeps the implementation simple and bug-free.
        max_start = self._filled - T
        if max_start <= 0:
            return None
        # Window start indices are relative to the oldest valid row, which is
        # at `_write_idx` once filled wraps. Sample in the "logical" order
        # then translate to physical ring indices.
        env_ids = torch.randint(0, self.num_envs, (B,), device=self.device)
        starts = torch.randint(0, max_start, (B,), device=self.device)
        # logical[i] = (write_idx - filled + i) mod capacity for i in 0..filled-1
        # so logical idx s corresponds to physical (write_idx - filled + s) mod cap.
        oldest = (self._write_idx - self._filled) % self.replay_capacity
        time_offsets = torch.arange(T, device=self.device)  # (T,)
        logical = starts.unsqueeze(1) + time_offsets.unsqueeze(0)  # (B, T)
        physical = (oldest + logical) % self.replay_capacity  # (B, T)
        # gather: ring[name][physical[b,t], env_ids[b]] for each (b,t)
        prop = self.ring["prop"][physical, env_ids.unsqueeze(1).expand(-1, T)]
        action = self.ring["action"][physical, env_ids.unsqueeze(1).expand(-1, T)]
        reward = self.ring["reward"][physical, env_ids.unsqueeze(1).expand(-1, T)]
        is_first = self.ring["is_first"][physical, env_ids.unsqueeze(1).expand(-1, T)]
        # Force is_first=1 at t=0 of every batch element so RSSM.observe
        # reinitializes per-batch state cleanly.
        is_first = is_first.clone()
        is_first[:, 0] = 1.0
        env_expand = env_ids.unsqueeze(1).expand(-1, T)
        if self.image_on_cpu:
            physical_cpu = physical.to("cpu")
            env_cpu = env_expand.to("cpu")
            image = self.ring["image"][physical_cpu, env_cpu].to(self.device, non_blocking=True)
            image_clean = self.ring["image_clean"][physical_cpu, env_cpu].to(self.device, non_blocking=True)
        else:
            image = self.ring["image"][physical, env_expand]
            image_clean = self.ring["image_clean"][physical, env_expand]
        pose_target = self.ring["pose_target"][physical, env_expand]
        tac_contact_target = self.ring["tac_contact_target"][physical, env_expand] if self.tac_pred_n_links > 0 else None
        value_target = self.ring["value_target"][physical, env_expand]
        obj_shape_target = self.ring["obj_shape_target"][physical, env_expand] if self.obj_pred_dim > 0 else None
        batch = {
            "prop": prop,
            "image": image,
            "image_clean": image_clean,
            "action": action,
            "reward": reward,
            "is_first": is_first,
            "pose_target": pose_target,
            "value_target": value_target,
        }
        if tac_contact_target is not None:
            batch["tac_contact_target"] = tac_contact_target
        if obj_shape_target is not None:
            batch["obj_shape_target"] = obj_shape_target
        return batch

    # ---------------------------------------------------------------- checkpoint
    def state_dict(self) -> Dict[str, Any]:
        return {
            "world_model": self.world_model.state_dict(),
            "wm_optimizer": self.world_model._model_opt._opt.state_dict(),
        }

    def _resolve_checkpoint_path(self, path: str) -> pathlib.Path:
        ckpt_path = pathlib.Path(path).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = self.repo_root / ckpt_path
        return ckpt_path

    @staticmethod
    def _load_torch_checkpoint(path: pathlib.Path) -> Dict[str, Any]:
        try:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(path), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise ValueError(f"checkpoint is not a dict: {path}")
        return checkpoint

    @staticmethod
    def _extract_world_model_state(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "world_model" in checkpoint:
            wm_block = checkpoint["world_model"]
            if isinstance(wm_block, dict) and "world_model" in wm_block:
                wm_state = wm_block["world_model"]
            else:
                wm_state = wm_block
        else:
            wm_state = checkpoint
        if not isinstance(wm_state, dict):
            raise ValueError("checkpoint does not contain a world_model state dict")
        return wm_state

    def load_body_from_checkpoint(self, path: str) -> None:
        """Initialize only the reusable WM body from an agent checkpoint."""
        ckpt_path = self._resolve_checkpoint_path(path)
        checkpoint = self._load_torch_checkpoint(ckpt_path)
        source_state = self._extract_world_model_state(checkpoint)
        target_state = self.world_model.state_dict()

        filtered: Dict[str, torch.Tensor] = {}
        skipped_prefix = 0
        skipped_missing = 0
        skipped_shape = 0
        for key, value in source_state.items():
            if not key.startswith(_WM_BODY_RESUME_PREFIXES):
                skipped_prefix += 1
                continue
            if key not in target_state:
                skipped_missing += 1
                continue
            if tuple(value.shape) != tuple(target_state[key].shape):
                skipped_shape += 1
                continue
            filtered[key] = value

        if not filtered:
            print(
                f"[world_model] body resume loaded 0 keys from {ckpt_path}; "
                f"skipped_prefix={skipped_prefix}, skipped_missing={skipped_missing}, "
                f"skipped_shape={skipped_shape}"
            )
            return

        result = self.world_model.load_state_dict(filtered, strict=False)
        print(
            f"[world_model] body resume from {ckpt_path}: loaded={len(filtered)}, "
            f"skipped_prefix={skipped_prefix}, skipped_missing={skipped_missing}, "
            f"skipped_shape={skipped_shape}, missing_after_partial={len(result.missing_keys)}, "
            f"unexpected_after_partial={len(result.unexpected_keys)}"
        )

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True) -> None:
        if "world_model" in state:
            self.world_model.load_state_dict(state["world_model"], strict=False)
        if load_optimizer and "wm_optimizer" in state:
            try:
                self.world_model._model_opt._opt.load_state_dict(state["wm_optimizer"])
            except Exception as exc:  # pragma: no cover - best effort restore
                print(f"[world_model] optimizer state restore skipped: {exc}")
