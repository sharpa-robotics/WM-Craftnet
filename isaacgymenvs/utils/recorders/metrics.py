"""Rolling per-episode metrics for WM-Craftnet training and evaluation.

Step-level accumulation is intentionally kept in the task as vectorized torch
buffers. This class only receives finished episodes, filters initial fly-away
episodes, and exposes a rolling summary suitable for TensorBoard logging.
"""

import csv
import math
import os
from collections import deque
from builtins import open as builtin_open


class EvalMetricsAggregator:
    def __init__(self, env_ids, invalid_episode_max_steps=10, window_size=1024):
        self.env_ids = [int(e) for e in env_ids]
        self.invalid_episode_max_steps = int(invalid_episode_max_steps)
        self.window_size = max(1, int(window_size))
        self.valid_episodes = deque(maxlen=self.window_size)
        # True = valid, False = filtered invalid. This tracks the denominator
        # for recent invalid-rate style metrics without storing invalid rows.
        self.recent_episode_valid = deque(maxlen=self.window_size)
        self.n_valid_total = 0
        self.n_invalid_total = 0
        self.n_total_completed = 0
        self._last_summary = None

    def reset_env(self, env_id):
        """Compatibility hook; step buffers live in the task now."""
        return None

    def on_step(self, env_id, spin_reward, spin_delta_axis, obj_angvel_norm,
                obj_linvel_norm, torque_mean_abs, torque_norm):
        """Compatibility hook; step buffers live in the task now."""
        return None

    def on_episode_end(
        self,
        env_id,
        steps,
        terminated_by_fall,
        rotr,
        rotations_rad,
        angvel_var,
        objvel_mean,
        torque_mean_abs,
        torque_norm,
        offaxis_mean=None,
        episode_return=None,
    ):
        """Record a completed episode and update the rolling summary."""
        env_id = int(env_id)
        steps = int(steps)
        if steps <= 0:
            return False

        self.n_total_completed += 1
        if terminated_by_fall and steps < self.invalid_episode_max_steps:
            self.n_invalid_total += 1
            self.recent_episode_valid.append(False)
        else:
            self.n_valid_total += 1
            self.recent_episode_valid.append(True)
            episode = {
                "env_id": env_id,
                "steps": steps,
                "terminated_by_fall": bool(terminated_by_fall),
                "RotR": float(rotr),
                "Rotations_rad": float(rotations_rad),
                "AngVel_var": max(float(angvel_var), 0.0),
                "ObjVel_mean": float(objvel_mean),
                "Torque_mean_abs": float(torque_mean_abs),
                "Torque_norm": float(torque_norm),
            }
            if offaxis_mean is not None:
                episode["OffAxis_mean"] = float(offaxis_mean)
            if episode_return is not None:
                episode["Return"] = float(episode_return)
            self.valid_episodes.append(episode)

        self._last_summary = self.summary()
        return True

    def summary(self):
        n_valid = len(self.valid_episodes)
        n_recent = len(self.recent_episode_valid)
        n_invalid_recent = sum(1 for is_valid in self.recent_episode_valid if not is_valid)

        def mean_std_ci(values):
            values = [float(v) for v in values]
            n = len(values)
            if n == 0:
                return math.nan, math.nan, math.nan, 0
            avg = sum(values) / n
            if n <= 1:
                return avg, 0.0, 0.0, n
            var = sum((v - avg) ** 2 for v in values) / (n - 1)
            std = math.sqrt(max(var, 0.0))
            ci95 = 1.96 * std / math.sqrt(n)
            return avg, std, ci95, n

        def episode_values(key):
            return [e[key] for e in self.valid_episodes if key in e]

        if n_valid > 0:
            fall_rate, fall_std, fall_ci95, fall_n = mean_std_ci(
                1.0 if e["terminated_by_fall"] else 0.0 for e in self.valid_episodes
            )
            avg_rotr, rotr_std, rotr_ci95, rotr_n = mean_std_ci(episode_values("RotR"))
            avg_rotations, rotations_std, rotations_ci95, rotations_n = mean_std_ci(episode_values("Rotations_rad"))
            avg_angvel_var, angvel_var_std, angvel_var_ci95, angvel_var_n = mean_std_ci(episode_values("AngVel_var"))
            avg_objvel, objvel_std, objvel_ci95, objvel_n = mean_std_ci(episode_values("ObjVel_mean"))
            avg_tq_mean, tq_mean_std, tq_mean_ci95, tq_mean_n = mean_std_ci(episode_values("Torque_mean_abs"))
            avg_tq_norm, tq_norm_std, tq_norm_ci95, tq_norm_n = mean_std_ci(episode_values("Torque_norm"))
            avg_steps, steps_std, steps_ci95, steps_n = mean_std_ci(e["steps"] for e in self.valid_episodes)
            avg_offaxis, offaxis_std, offaxis_ci95, offaxis_n = mean_std_ci(episode_values("OffAxis_mean"))
            avg_return, return_std, return_ci95, return_n = mean_std_ci(episode_values("Return"))
        else:
            fall_rate = math.nan
            fall_std = math.nan
            fall_ci95 = math.nan
            fall_n = 0
            avg_rotr = math.nan
            rotr_std = math.nan
            rotr_ci95 = math.nan
            rotr_n = 0
            avg_rotations = math.nan
            rotations_std = math.nan
            rotations_ci95 = math.nan
            rotations_n = 0
            avg_angvel_var = math.nan
            angvel_var_std = math.nan
            angvel_var_ci95 = math.nan
            angvel_var_n = 0
            avg_objvel = math.nan
            objvel_std = math.nan
            objvel_ci95 = math.nan
            objvel_n = 0
            avg_tq_mean = math.nan
            tq_mean_std = math.nan
            tq_mean_ci95 = math.nan
            tq_mean_n = 0
            avg_tq_norm = math.nan
            tq_norm_std = math.nan
            tq_norm_ci95 = math.nan
            tq_norm_n = 0
            avg_steps = math.nan
            steps_std = math.nan
            steps_ci95 = math.nan
            steps_n = 0
            avg_offaxis = math.nan
            offaxis_std = math.nan
            offaxis_ci95 = math.nan
            offaxis_n = 0
            avg_return = math.nan
            return_std = math.nan
            return_ci95 = math.nan
            return_n = 0

        return {
            ("n_episodes_recent"): n_recent,
            ("n_episodes_invalid_recent"): n_invalid_recent,
            ("n_episodes_valid"): n_valid,
            ("n_episodes_total_completed"): self.n_total_completed,
            ("n_episodes_invalid_total"): self.n_invalid_total,
            ("n_episodes_valid_total"): self.n_valid_total,
            ("fall_rate"): fall_rate,
            ("fall_rate_std"): fall_std,
            ("fall_rate_ci95"): fall_ci95,
            ("fall_rate_n"): fall_n,
            ("RotR"): avg_rotr,
            ("RotR_std"): rotr_std,
            ("RotR_ci95"): rotr_ci95,
            ("RotR_n"): rotr_n,
            ("Rotations_rad"): avg_rotations,
            ("Rotations_rad_std"): rotations_std,
            ("Rotations_rad_ci95"): rotations_ci95,
            ("Rotations_rad_n"): rotations_n,
            ("AngVel_var"): avg_angvel_var,
            ("AngVel_var_std"): angvel_var_std,
            ("AngVel_var_ci95"): angvel_var_ci95,
            ("AngVel_var_n"): angvel_var_n,
            ("ObjVel_mean"): avg_objvel,
            ("ObjVel_mean_std"): objvel_std,
            ("ObjVel_mean_ci95"): objvel_ci95,
            ("ObjVel_mean_n"): objvel_n,
            ("Torque_mean_abs"): avg_tq_mean,
            ("Torque_mean_abs_std"): tq_mean_std,
            ("Torque_mean_abs_ci95"): tq_mean_ci95,
            ("Torque_mean_abs_n"): tq_mean_n,
            ("Torque_norm"): avg_tq_norm,
            ("Torque_norm_std"): tq_norm_std,
            ("Torque_norm_ci95"): tq_norm_ci95,
            ("Torque_norm_n"): tq_norm_n,
            ("EpisodeLen"): avg_steps,
            ("EpisodeLen_std"): steps_std,
            ("EpisodeLen_ci95"): steps_ci95,
            ("EpisodeLen_n"): steps_n,
            ("OffAxis_mean"): avg_offaxis,
            ("OffAxis_mean_std"): offaxis_std,
            ("OffAxis_mean_ci95"): offaxis_ci95,
            ("OffAxis_mean_n"): offaxis_n,
            ("Return"): avg_return,
            ("Return_std"): return_std,
            ("Return_ci95"): return_ci95,
            ("Return_n"): return_n,
        }

    def last_summary(self):
        if self._last_summary is not None:
            return dict(self._last_summary)
        return self.summary()

    def flush(self, output_path):
        """Optional CSV export for standalone tools; training logs TB scalars."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary = self.last_summary()
        with builtin_open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for metric_name, metric_value in summary.items():
                writer.writerow([metric_name, metric_value])
        out = dict(summary)
        out["output_path"] = output_path
        return out
