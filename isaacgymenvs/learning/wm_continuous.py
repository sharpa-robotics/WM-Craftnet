# Copyright (c) 2026 The WM-Craftnet Authors
# SPDX-License-Identifier: Apache-2.0

"""WM-Craftnet PPO agent with a per-epoch World Synesthesia Model hook.

Wraps the stock ``rl_games.algos_torch.a2c_continuous.A2CAgent`` to add three
things when ``config['world_model_enabled']`` is set:
  1. After each PPO epoch, sample ``train_steps_per_epoch`` batches from the
     task's ring replay and run ``WorldModel._train``.
  2. Persist WM weights and its optimizer state in the agent checkpoint.
  3. Log WM losses to tensorboard.

The WM forward used by the actor lives in the env (see
``WorldModelAdapter.step``) and is *not* a parameter of this PPO optimizer.
That separation matches WMP and keeps the gradient graphs disjoint.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from rl_games.algos_torch import a2c_continuous, torch_ext


class WMA2CAgent(a2c_continuous.A2CAgent):
    """A2CAgent that also trains the WM-Craftnet World Synesthesia Model."""

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self._wm_enabled = bool(self.config.get("world_model_enabled", False))
        self._wm_adapter = self._resolve_wm_adapter() if self._wm_enabled else None
        if self._wm_enabled and self._wm_adapter is None:
            raise RuntimeError(
                "world_model_enabled=True but the task did not expose a wm_adapter."
            )

    def _resolve_wm_adapter(self):
        """Walk through the vec_env wrappers to find the task's wm_adapter."""
        env = getattr(self, "vec_env", None)
        if env is None:
            return None
        # Try common attribute paths used by RLGPUEnv / isaacgymenvs vec_task.
        for getter in (
            lambda e: getattr(e, "env", None),
            lambda e: getattr(getattr(e, "env", None), "env", None),
            lambda e: getattr(getattr(e, "env", None), "task", None),
        ):
            inner = getter(env)
            if inner is not None and hasattr(inner, "wm_adapter"):
                wm = getattr(inner, "wm_adapter", None)
                if wm is not None:
                    return wm
        # Last-ditch: scan attributes shallowly.
        for attr in dir(env):
            if attr.startswith("_"):
                continue
            try:
                cand = getattr(env, attr)
            except Exception:
                continue
            if hasattr(cand, "wm_adapter") and getattr(cand, "wm_adapter") is not None:
                return cand.wm_adapter
        return None

    def get_action_values(self, obs):
        res_dict = super().get_action_values(obs)
        if self._wm_enabled and self._wm_adapter is not None and "values" in res_dict:
            try:
                self._wm_adapter.set_last_value_target(res_dict["values"])
            except Exception as exc:  # pragma: no cover - best effort alignment
                print(f"[world_model] value target backfill skipped: {exc}")
        return res_dict

    # ---------------------------------------------------------------- per epoch
    def train_epoch(self):
        result = super().train_epoch()
        if self._wm_enabled and self._wm_adapter is not None:
            self._train_world_model_once()
        return result

    @torch.no_grad()
    def _to_torch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # WSM preprocessing uses torch.Tensor(v); make sure everything is a
        # contiguous float tensor on the WM device beforehand to avoid a
        # double-allocation through CPU.
        out = {}
        device = self._wm_adapter.device
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            v = v.to(device=device, dtype=torch.float32, non_blocking=True).contiguous()
            out[k] = v
        return out

    def _train_world_model_once(self) -> None:
        adapter = self._wm_adapter
        if not adapter.can_train():
            return
        metrics_acc: Dict[str, float] = {}
        steps = 0
        for _ in range(adapter.train_steps_per_epoch):
            batch = adapter.sample_batch()
            if batch is None:
                continue
            batch = self._to_torch(batch)
            try:
                _, _, metrics = adapter.world_model._train(batch)
            except Exception as exc:  # pragma: no cover - log and skip on bad batch
                print(f"[world_model] train step skipped: {exc}")
                continue
            steps += 1
            for k, v in metrics.items():
                val = float(np.asarray(v).mean()) if not np.isscalar(v) else float(v)
                metrics_acc[k] = metrics_acc.get(k, 0.0) + val
        if self.writer is not None and steps > 0:
            for k, v in metrics_acc.items():
                self.writer.add_scalar(f"world_model/{k}", v / steps, self.epoch_num)
            self.writer.add_scalar("world_model/train_steps", steps, self.epoch_num)
            self.writer.add_scalar(
                "world_model/replay_filled_steps",
                adapter.filled_steps(),
                self.epoch_num,
            )

    # ---------------------------------------------------------------- checkpoint
    def get_full_state_weights(self) -> Dict[str, Any]:
        state = super().get_full_state_weights()
        if self._wm_enabled and self._wm_adapter is not None:
            state["world_model"] = self._wm_adapter.state_dict()
        return state

    def set_full_state_weights(self, weights: Dict[str, Any], mode: str = "student") -> None:
        super().set_full_state_weights(weights, mode)
        if self._wm_enabled and self._wm_adapter is not None and "world_model" in weights:
            self._wm_adapter.load_state_dict(weights["world_model"], load_optimizer=True)
