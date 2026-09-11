#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Minimal remote ZMQ policy server.

Server responsibilities:
1) Receive qpos_isaac (raw joint states from real hand).
2) Maintain all target state internally (init + run).
3) Return target_isaac command only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any

import gym
import numpy as np
from omegaconf import OmegaConf

try:
    import zmq
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("pyzmq is required. Install it with: pip install pyzmq") from exc


ISAAC_JOINT_NAMES: list[str] = [
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
]


def scale_to_minus1_plus1(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    offset = 0.5 * (lower + upper)
    return 2.0 * (x - offset) / (upper - lower)


def axis_to_vec(axis: str) -> np.ndarray:
    m = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    if axis not in m:
        raise ValueError(f"Unsupported axis: {axis}")
    return m[axis]


def _setup_import_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _resolve_checkpoint_cfg_path(checkpoint_path: str) -> Path:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    cfg_path = ckpt_path.with_suffix(".yaml")
    if cfg_path.is_file():
        return cfg_path
    raise RuntimeError(
        "Configuration for checkpoint not found. Place a sibling YAML next to "
        f"the checkpoint (example_ckpt/<name>.pth + example_ckpt/<name>.yaml). "
        f"Tried: {cfg_path}"
    )


def _load_run_cfg_from_checkpoint(checkpoint_path: str):
    cfg_path = _resolve_checkpoint_cfg_path(checkpoint_path)
    cfg_obj = OmegaConf.load(str(cfg_path))
    if "train" not in cfg_obj or "params" not in cfg_obj["train"]:
        raise RuntimeError(f"Configuration is missing train.params: {cfg_path}")
    return cfg_obj, str(cfg_path)


def _resolve_urdf_path(repo_root: Path, urdf_file_name: str) -> Path | None:
    if not urdf_file_name:
        return None
    urdf_rel = Path(urdf_file_name)
    if urdf_rel.is_absolute():
        candidate_paths = [urdf_rel]
    else:
        candidate_paths = [
            repo_root / "assets" / urdf_rel,
            repo_root / "isaacgymenvs" / "assets" / urdf_rel,
        ]
    for p in candidate_paths:
        p_abs = p.expanduser().resolve()
        if p_abs.is_file():
            return p_abs
    return None


def _load_dof_limits_from_asset_urdf(
    repo_root: Path,
    robot_asset_file: str,
    fallback_asset_file: str,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_files = [robot_asset_file, fallback_asset_file]
    tried: list[str] = []
    for f in candidate_files:
        if not f:
            continue
        urdf_path = _resolve_urdf_path(repo_root, f)
        if urdf_path is None:
            tried.append(f"{f} (not found)")
            continue

        root = ET.parse(urdf_path).getroot()
        limit_by_name: dict[str, tuple[float, float]] = {}
        for joint in root.findall("joint"):
            name = joint.get("name", "").strip()
            if not name:
                continue
            jtype = joint.get("type", "").strip()
            if jtype not in ("revolute", "continuous", "prismatic"):
                continue
            limit = joint.find("limit")
            if limit is None:
                continue
            lower_raw = limit.get("lower")
            upper_raw = limit.get("upper")
            if lower_raw is None or upper_raw is None:
                continue
            try:
                lower = float(lower_raw)
                upper = float(upper_raw)
            except ValueError:
                continue
            limit_by_name[name] = (lower, upper)

        missing = [name for name in ISAAC_JOINT_NAMES if name not in limit_by_name]
        if missing:
            tried.append(f"{urdf_path} (missing joints: {missing})")
            continue

        lower = np.array([limit_by_name[name][0] for name in ISAAC_JOINT_NAMES], dtype=np.float32)
        upper = np.array([limit_by_name[name][1] for name in ISAAC_JOINT_NAMES], dtype=np.float32)
        print(f"[INFO] Loaded DOF limits from URDF: {urdf_path}")
        return lower, upper

    raise RuntimeError(
        "Could not load complete hand joint limits from the simulation asset URDF. tried=["
        + "; ".join(tried) + "]"
    )


def _apply_train_limit_tightening(
    lower: np.ndarray,
    upper: np.ndarray,
    train_limit_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    out_lower = np.asarray(lower, dtype=np.float32).copy()
    out_upper = np.asarray(upper, dtype=np.float32).copy()
    if not isinstance(train_limit_cfg, dict):
        return out_lower, out_upper

    global_scale = float(train_limit_cfg.get("global_scale", 1.0))
    unilateral_cfg = train_limit_cfg.get("unilateral_override_deg", {})
    if not isinstance(unilateral_cfg, dict):
        unilateral_cfg = {}

    if global_scale != 1.0:
        out_lower *= global_scale
        out_upper *= global_scale

    for joint_name, joint_cfg in unilateral_cfg.items():
        if joint_name not in ISAAC_JOINT_NAMES:
            continue
        if not isinstance(joint_cfg, dict):
            continue
        idx = ISAAC_JOINT_NAMES.index(joint_name)
        lo = float(out_lower[idx])
        hi = float(out_upper[idx])
        if "lower" in joint_cfg:
            lo = float(np.deg2rad(float(joint_cfg["lower"])))
        if "upper" in joint_cfg:
            hi = float(np.deg2rad(float(joint_cfg["upper"])))
        if lo > hi:
            raise ValueError(f"Invalid trainLimit override for {joint_name}: lower({lo}) > upper({hi})")
        out_lower[idx] = lo
        out_upper[idx] = hi
    return out_lower, out_upper


def get_sim_init_pose_isaac(hand_init_pose_by_name_rad: dict[str, float]) -> np.ndarray:
    pose = np.zeros((22,), dtype=np.float32)
    for i, name in enumerate(ISAAC_JOINT_NAMES):
        pose[i] = float(hand_init_pose_by_name_rad.get(name, 0.0))
    return pose


def _extract_infer_runtime_cfg(cfg_obj: Any) -> dict[str, Any]:
    train_params = cfg_obj["train"]["params"]
    env_cfg_raw = cfg_obj["task"]["env"]
    sim_cfg_raw = cfg_obj["task"]["sim"]

    train_params_dict = OmegaConf.to_container(train_params, resolve=False)
    env_cfg = OmegaConf.to_container(env_cfg_raw, resolve=False)
    sim_cfg = OmegaConf.to_container(sim_cfg_raw, resolve=False)
    if not isinstance(train_params_dict, dict) or not isinstance(env_cfg, dict) or not isinstance(sim_cfg, dict):
        raise RuntimeError("config parse failed for train/task env/sim.")

    asset_cfg = env_cfg.get("asset", {})
    if not isinstance(asset_cfg, dict):
        asset_cfg = {}

    def _as_bool(v: Any, default: bool) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off", ""):
                return False
        return bool(default)

    def _as_int(v: Any, default: int) -> int:
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, (int, np.integer)):
            return int(v)
        if isinstance(v, float):
            return int(v)
        if isinstance(v, str) and v.strip():
            try:
                return int(float(v.strip()))
            except ValueError:
                return int(default)
        return int(default)

    def _as_float(v: Any, default: float) -> float:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        if isinstance(v, str) and v.strip():
            try:
                return float(v.strip())
            except ValueError:
                return float(default)
        return float(default)

    control_dt = _as_float(sim_cfg.get("dt", 1.0 / 60.0), 1.0 / 60.0) * _as_float(
        env_cfg.get("controlFrequencyInv", 1), 1.0
    )
    control_hz = 1.0 / max(control_dt, 1e-6)
    train_limit_cfg = env_cfg.get("trainLimit", {})
    if not isinstance(train_limit_cfg, dict):
        train_limit_cfg = {}
    unilateral_cfg = train_limit_cfg.get("unilateral_override_deg", {})
    if not isinstance(unilateral_cfg, dict):
        unilateral_cfg = {}
    train_limit_cfg = {
        "global_scale": _as_float(train_limit_cfg.get("global_scale", 1.0), 1.0),
        "unilateral_override_deg": unilateral_cfg,
    }

    hand_init_type = str(env_cfg.get("handInit", "default"))
    hand_init_poses_cfg = env_cfg.get("handInitPoses", {})
    if not isinstance(hand_init_poses_cfg, dict):
        raise RuntimeError("task.env.handInitPoses parse failed")
    selected_hand_init_deg = hand_init_poses_cfg.get(hand_init_type, None)
    if not isinstance(selected_hand_init_deg, dict):
        raise RuntimeError(f"task.env.handInit '{hand_init_type}' not found in handInitPoses")
    hand_init_pose_rad: dict[str, float] = {}
    for joint_name in ISAAC_JOINT_NAMES:
        if joint_name not in selected_hand_init_deg:
            raise RuntimeError(f"handInitPoses['{hand_init_type}'] missing joint: {joint_name}")
        hand_init_pose_rad[joint_name] = float(np.deg2rad(float(selected_hand_init_deg[joint_name])))

    return {
        "train_params": train_params_dict,
        "rel_scale": _as_float(env_cfg.get("relScale", 0.2), 0.2),
        "actions_moving_average": _as_float(env_cfg.get("actionsMovingAverage", 0.8), 0.8),
        "clip_observations": _as_float(env_cfg.get("clipObservations", np.inf), np.inf),
        "obs_stack": _as_int(env_cfg.get("obs_stack", 4), 4),
        "axis": str(env_cfg.get("axis", "z")),
        "disable_finger_tactile_obs_reward": _as_bool(env_cfg.get("disableTacObs", False), False),
        "control_hz": control_hz,
        "dof_limit_scale": _as_float(env_cfg.get("dofLimitScale", env_cfg.get("dof_limit_scale", 1.0)), 1.0),
        "train_limit": train_limit_cfg,
        "robot_asset_file": str(env_cfg.get("robotAssetFile", "")),
        "asset_file_name": str(asset_cfg.get("assetFileName", "")),
        "hand_init_type": hand_init_type,
        "hand_init_pose_rad": hand_init_pose_rad,
    }


def build_player(
    checkpoint: str,
    train_params: dict[str, Any],
    obs_dim: int,
    device_name: str,
):
    from rl_games.algos_torch.players import PpoPlayerContinuous

    params: dict[str, Any] = copy.deepcopy(train_params)
    params["config"]["device_name"] = device_name
    params["config"]["env_name"] = "offline_real_zmq_server"
    params["config"]["player"] = {"deterministic": True, "games_num": 1, "print_stats": False}

    obs_spaces: dict[str, gym.Space] = {
        "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
    }
    params["config"]["env_info"] = {
        "action_space": gym.spaces.Box(low=-1.0, high=1.0, shape=(22,), dtype=np.float32),
        "observation_space": gym.spaces.Dict(obs_spaces),
        "agents": 1,
        "value_size": 1,
    }
    params["config"]["vec_env"] = None
    player = PpoPlayerContinuous(params)
    player.restore(checkpoint)
    player.reset()
    player.has_batch_dimension = True
    return player


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Minimal remote ZMQ policy server")
    parser.add_argument(
        "--checkpoint",
        default=str(repo_root / "example_ckpt/wm_craftnet_set_z.pth"),
    )
    parser.add_argument("--bind-endpoint", default="tcp://*:5555")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--init-step-rad", type=float, default=0.08)
    parser.add_argument("--init-tol-rad", type=float, default=0.08)
    parser.add_argument("--diag-log-interval", type=int, default=20)
    return parser.parse_args()


def _json_error(step_id: int, msg: str) -> dict[str, Any]:
    return {
        "ok": False,
        "step_id": int(step_id),
        "error": str(msg),
        "server_ts": float(time.time()),
    }


def main() -> None:
    _setup_import_paths()
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    run_cfg_obj, run_cfg_path = _load_run_cfg_from_checkpoint(args.checkpoint)
    infer_cfg = _extract_infer_runtime_cfg(run_cfg_obj)
    print(f"[INFO] Loaded run config: {run_cfg_path}")

    rel_scale = float(infer_cfg["rel_scale"])
    actions_moving_average = float(infer_cfg["actions_moving_average"])
    clip_observations = float(infer_cfg["clip_observations"])
    n_stack = int(infer_cfg["obs_stack"])
    axis = str(infer_cfg["axis"])
    dof_limit_scale = float(infer_cfg["dof_limit_scale"])
    finger_tactile_obs_dim = 0 if bool(infer_cfg["disable_finger_tactile_obs_reward"]) else 5
    n_base_obs = 22 * 2 + finger_tactile_obs_dim + 24
    obs_dim = n_base_obs * n_stack
    print(
        f"[INFO] obs_dim={obs_dim} | obs_stack={n_stack} | clipObs={clip_observations} "
        f"| finger_tactile_obs_dim={finger_tactile_obs_dim}"
    )

    urdf_lower, urdf_upper = _load_dof_limits_from_asset_urdf(
        repo_root=repo_root,
        robot_asset_file=str(infer_cfg.get("robot_asset_file", "")),
        fallback_asset_file=str(infer_cfg.get("asset_file_name", "")),
    )
    urdf_lower, urdf_upper = _apply_train_limit_tightening(
        lower=urdf_lower,
        upper=urdf_upper,
        train_limit_cfg=infer_cfg.get("train_limit", {}),
    )
    low = urdf_lower * dof_limit_scale
    high = urdf_upper * dof_limit_scale

    player = build_player(
        checkpoint=os.path.abspath(args.checkpoint),
        train_params=infer_cfg["train_params"],
        obs_dim=obs_dim,
        device_name=args.device,
    )

    sim_init = get_sim_init_pose_isaac(infer_cfg.get("hand_init_pose_rad", {})).astype(np.float32)
    sim_init = np.clip(sim_init, low, high)
    spin_axis = np.tile(axis_to_vec(axis), 8).astype(np.float32)

    prev_target_isaac: np.ndarray | None = None
    last_action = np.zeros((22,), dtype=np.float32)
    obs_hist: deque[np.ndarray] = deque(maxlen=n_stack)
    server_state = "init"
    diag_interval = max(1, int(args.diag_log_interval))
    handled = 0

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.bind_endpoint)
    print(f"[INFO] ZMQ server bound at {args.bind_endpoint}")

    while True:
        raw = sock.recv_string()
        step_id = -1
        try:
            req = json.loads(raw)
            step_id = int(req.get("step_id", -1))
            qpos_isaac = np.asarray(req.get("qpos_isaac", []), dtype=np.float32).reshape(-1)
            if qpos_isaac.shape[0] != 22:
                sock.send_string(json.dumps(_json_error(step_id, f"qpos_isaac dim must be 22, got {qpos_isaac.shape}")))
                continue
            if not np.all(np.isfinite(qpos_isaac)):
                sock.send_string(json.dumps(_json_error(step_id, "qpos_isaac contains NaN/Inf")))
                continue

            if prev_target_isaac is None:
                prev_target_isaac = qpos_isaac.copy()
                for _ in range(n_stack):
                    qpos_obs = scale_to_minus1_plus1(qpos_isaac, low, high)
                    target_obs = scale_to_minus1_plus1(prev_target_isaac, low, high)
                    if finger_tactile_obs_dim > 0:
                        contact_obs = np.zeros((5,), dtype=np.float32)
                        base = np.concatenate([qpos_obs, target_obs, contact_obs, spin_axis], axis=0)
                    else:
                        base = np.concatenate([qpos_obs, target_obs, spin_axis], axis=0)
                    obs_hist.appendleft(base.astype(np.float32))

            if server_state == "init":
                init_delta = np.clip(
                    sim_init - prev_target_isaac,
                    -float(args.init_step_rad),
                    float(args.init_step_rad),
                )
                target = np.clip(prev_target_isaac + init_delta, low, high)
                prev_target_isaac = target.copy()
                max_init_err = float(np.max(np.abs(sim_init - qpos_isaac)))
                if max_init_err <= float(args.init_tol_rad):
                    server_state = "run"
                    print(f"[INFO] init->run transition at step_id={step_id}, max_init_err={max_init_err:.4f}rad")
            else:
                qpos_obs = scale_to_minus1_plus1(qpos_isaac, low, high)
                target_obs = scale_to_minus1_plus1(prev_target_isaac, low, high)
                if finger_tactile_obs_dim > 0:
                    contact_obs = np.zeros((5,), dtype=np.float32)
                    base = np.concatenate([qpos_obs, target_obs, contact_obs, spin_axis], axis=0)
                else:
                    base = np.concatenate([qpos_obs, target_obs, spin_axis], axis=0)
                obs_hist.appendleft(base.astype(np.float32))
                obs_vec = np.concatenate(list(obs_hist), axis=0).astype(np.float32)
                obs_vec = np.clip(obs_vec, -clip_observations, clip_observations)
                obs_dict: dict[str, np.ndarray] = {"obs": obs_vec[None, :]}
                action = player.get_action(player.obs_to_torch({"obs": obs_dict}), is_deterministic=True)
                action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
                smooth = action_np * actions_moving_average + last_action * (1.0 - actions_moving_average)
                target_raw = prev_target_isaac + rel_scale * smooth
                target = np.clip(target_raw, low, high)
                prev_target_isaac = target.copy()
                last_action = smooth.copy()

            handled += 1
            if handled % diag_interval == 0:
                print(
                    f"[INFO][diag] handled={handled} step_id={step_id} "
                    f"| state={server_state} | target_norm={float(np.linalg.norm(prev_target_isaac)):.4f}"
                )

            resp = {
                "ok": True,
                "step_id": int(step_id),
                "target_isaac": prev_target_isaac.astype(np.float32).tolist(),
                "server_state": server_state,
                "server_ts": float(time.time()),
            }
            sock.send_string(json.dumps(resp, ensure_ascii=True))
        except Exception as exc:  # noqa: BLE001
            sock.send_string(json.dumps(_json_error(step_id, f"server exception: {exc}"), ensure_ascii=True))


if __name__ == "__main__":
    main()
