#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Real-hand online checkpoint inference in Python3.10.

Features:
- Load rl_games checkpoint and build policy network.
- Read D405 depth stream as camera observation.
- Build stacked observation with sim-like semantics.
- Apply sim-consistent action postprocess (relative control + smoothing + clamp).
- Handle Isaac<->Sharpa joint order mapping.
- Optional pre-move to sim default init hand pose before inference.
"""

from __future__ import annotations

import copy
import csv
import ctypes
import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import gym
import numpy as np
import torch
from omegaconf import OmegaConf

try:
    import pyrealsense2 as rs
except Exception as exc:  # noqa: BLE001
    raise RuntimeError("pyrealsense2 is required. Install it before running this script.") from exc


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

SHARPA_JOINT_NAMES: list[str] = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]

ISAACLAB2SHARPA_IDX: list[int] = [ISAAC_JOINT_NAMES.index(name) for name in SHARPA_JOINT_NAMES]

_OLD_LIMIT_ORDER: list[str] = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA", "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP", "right_pinky_DIP",
]
_OLD_DOF_LOWER = np.array(
    [
        -0.1745, -0.1745, 0.0000, -0.1745, -0.1745, -0.3491, -0.3491, -0.1745, -0.3491, -0.3491,
        0.0000, 0.0000, -0.3491, 0.0000, -0.5236, 0.0000, 0.0000, 0.0000, 0.0000, -0.3491, 0.0000,
        0.0000,
    ],
    dtype=np.float32,
)
_OLD_DOF_UPPER = np.array(
    [
        1.5708, 1.5708, 0.2618, 1.5708, 1.9199, 0.3491, 0.3491, 1.5708, 0.3491, 0.3491,
        1.7453, 1.7453, 0.3491, 1.7453, 1.3963, 1.3963, 1.3963, 1.7453, 1.3963, 0.3491, 1.3963,
        1.7453,
    ],
    dtype=np.float32,
)
_LOWER_BY_NAME = {n: v for n, v in zip(_OLD_LIMIT_ORDER, _OLD_DOF_LOWER)}
_UPPER_BY_NAME = {n: v for n, v in zip(_OLD_LIMIT_ORDER, _OLD_DOF_UPPER)}
DOF_LOWER = np.array([_LOWER_BY_NAME[name] for name in ISAAC_JOINT_NAMES], dtype=np.float32)
DOF_UPPER = np.array([_UPPER_BY_NAME[name] for name in ISAAC_JOINT_NAMES], dtype=np.float32)

def sharpa_to_isaac(sharpa_angles: np.ndarray) -> np.ndarray:
    isaac = np.zeros_like(sharpa_angles, dtype=np.float32)
    for sharpa_i, isaac_i in enumerate(ISAACLAB2SHARPA_IDX):
        isaac[isaac_i] = float(sharpa_angles[sharpa_i])
    return isaac


def isaac_to_sharpa(isaac_angles: np.ndarray) -> np.ndarray:
    return isaac_angles[np.asarray(ISAACLAB2SHARPA_IDX, dtype=np.int64)]


def scale_to_minus1_plus1(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    offset = 0.5 * (lower + upper)
    return 2.0 * (x - offset) / (upper - lower)


def get_sim_init_pose_isaac(hand_init_pose_by_name_rad: dict[str, float]) -> np.ndarray:
    pose = np.zeros((22,), dtype=np.float32)
    for i, name in enumerate(ISAAC_JOINT_NAMES):
        pose[i] = float(hand_init_pose_by_name_rad.get(name, 0.0))
    return pose


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
            print(f"[WARN] trainLimit joint not found in ISAAC_JOINT_NAMES: {joint_name}, skip.")
            continue
        if not isinstance(joint_cfg, dict):
            print(f"[WARN] trainLimit.unilateral_override_deg[{joint_name}] must be dict, skip.")
            continue
        idx = ISAAC_JOINT_NAMES.index(joint_name)
        lo = float(out_lower[idx])
        hi = float(out_upper[idx])
        if "lower" in joint_cfg:
            lo = float(np.deg2rad(float(joint_cfg["lower"])))
        if "upper" in joint_cfg:
            hi = float(np.deg2rad(float(joint_cfg["upper"])))
        if lo > hi:
            raise ValueError(
                f"Invalid trainLimit override for {joint_name}: lower({lo}) > upper({hi})"
            )
        out_lower[idx] = lo
        out_upper[idx] = hi

    if global_scale != 1.0 or len(unilateral_cfg) > 0:
        print(f"[INFO][trainLimit][infer] global_scale={global_scale}")
        for joint_name in unilateral_cfg.keys():
            if joint_name not in ISAAC_JOINT_NAMES:
                continue
            jidx = ISAAC_JOINT_NAMES.index(joint_name)
            lo_deg = float(np.rad2deg(out_lower[jidx]))
            hi_deg = float(np.rad2deg(out_upper[jidx]))
            print(f"[INFO][trainLimit][infer] {joint_name}: [{lo_deg:.2f}, {hi_deg:.2f}] deg")

    return out_lower, out_upper


def _ok(err, action: str) -> None:
    if hasattr(err, "code") and err.code != 0:
        msg = getattr(err, "message", "")
        raise RuntimeError(f"{action} failed, code={err.code}, message={msg}")


def _err_code_msg(err) -> tuple[int, str]:
    if hasattr(err, "code"):
        code = int(getattr(err, "code", 0))
        msg = str(getattr(err, "message", ""))
        return code, msg
    return 0, ""


def _move_to_pose_interp_and_wait(
    hand,
    target_isaac: np.ndarray,
    enable_actuation: bool,
    tol_rad: float = 0.08,
    poll_dt_s: float = 0.05,
    step_rad: float = 0.08,
) -> None:
    state = hand.get_states()
    qpos_sharpa = np.asarray(state.angles, dtype=np.float32)
    if qpos_sharpa.shape[0] != 22:
        raise RuntimeError(f"Unexpected dof count from SDK: {qpos_sharpa.shape[0]}")
    start_isaac = sharpa_to_isaac(qpos_sharpa)
    max_delta = float(np.max(np.abs(target_isaac - start_isaac)))
    n_steps = max(1, int(np.ceil(max_delta / max(step_rad, 1e-6))))
    print(
        "[INFO][init_interp] start move-to-init: "
        f"max_delta={max_delta:.4f}rad, n_steps={n_steps}, "
        f"step_rad={step_rad:.4f}, tol={tol_rad:.4f}rad, actuation={int(enable_actuation)}"
    )
    if enable_actuation:
        for k in range(1, n_steps + 1):
            alpha = float(k) / float(n_steps)
            interp = start_isaac + alpha * (target_isaac - start_isaac)
            cmd_ret = hand.set_joint_position(isaac_to_sharpa(interp).tolist())
            code, msg = _err_code_msg(cmd_ret)
            if k == 1 or k == n_steps or k % max(1, n_steps // 5) == 0:
                cur_err = float(np.max(np.abs(interp - start_isaac)))
                print(
                    "[INFO][init_interp] "
                    f"step={k}/{n_steps}, progress={alpha:.3f}, interp_delta={cur_err:.4f}rad, "
                    f"cmd_code={code}, cmd_msg='{msg}'"
                )
            _ok(cmd_ret, "set_joint_position(interp_init)")
            time.sleep(max(1e-3, float(poll_dt_s)))
    else:
        print("[INFO][init_interp] no-actuation: skipping interpolated init commands and target wait.")
        return

    last_err = None
    log_every_s = 0.5
    next_log_t = time.time()
    while True:
        state = hand.get_states()
        qpos_sharpa = np.asarray(state.angles, dtype=np.float32)
        if qpos_sharpa.shape[0] != 22:
            raise RuntimeError(f"Unexpected dof count from SDK: {qpos_sharpa.shape[0]}")
        qpos_isaac = sharpa_to_isaac(qpos_sharpa)
        err = np.abs(target_isaac - qpos_isaac)
        last_err = float(np.max(err))
        if last_err <= float(tol_rad):
            print(f"[INFO][init_interp] target reached, max_err={last_err:.4f}rad")
            return
        now_t = time.time()
        if now_t >= next_log_t:
            top_idx = np.argsort(err)[-3:][::-1]
            top_msg = ", ".join([f"{ISAAC_JOINT_NAMES[int(i)]}={float(err[int(i)]):.3f}" for i in top_idx])
            print(f"[INFO][init_interp] wait max_err={last_err:.4f}rad | top={top_msg}")
            next_log_t = now_t + log_every_s
        if enable_actuation:
            cmd_ret = hand.set_joint_position(isaac_to_sharpa(target_isaac).tolist())
            _ok(cmd_ret, "set_joint_position(init_wait)")
        time.sleep(max(1e-3, float(poll_dt_s)))


def sanitize_colon_path_var(name: str, skip_pattern: str) -> None:
    raw = os.environ.get(name, "")
    if not raw:
        return
    kept = []
    for item in raw.split(":"):
        if not item:
            continue
        if skip_pattern in item:
            continue
        kept.append(item)
    os.environ[name] = ":".join(kept)


def _setup_import_paths() -> None:
    deploy_root = Path(__file__).resolve().parents[1]
    repo_root = deploy_root.parent
    sdk_py_dir = deploy_root / "sharpa_sdk" / "python"
    sdk_lib_dir = deploy_root / "sharpa_sdk" / "lib"
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if sdk_py_dir.exists() and str(sdk_py_dir) not in sys.path:
        sys.path.insert(0, str(sdk_py_dir))
    if sdk_lib_dir.exists():
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = f"{sdk_lib_dir}:{old}" if old else str(sdk_lib_dir)


def _load_lib_if_exists(lib_path: Path) -> bool:
    if not lib_path.exists():
        return False
    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
    return True


def _preload_sdk_libs() -> None:
    deploy_root = Path(__file__).resolve().parents[1]
    candidate_dirs = [deploy_root / "sharpa_sdk" / "lib"]

    deduped_dirs: list[Path] = []
    seen = set()
    for d in candidate_dirs:
        try:
            exists = d.exists()
        except PermissionError:
            exists = False
        key = str(d.resolve()) if exists else str(d)
        if key in seen:
            continue
        seen.add(key)
        deduped_dirs.append(d)

    required = [
        "libsharpa-wave-sdk.so",
        "libSharpaWaveSDKWrapper.so",
    ]

    loaded: list[str] = []
    errors: list[str] = []
    searchable_dirs: list[Path] = []
    for d in deduped_dirs:
        try:
            if d.is_dir() and os.access(d, os.R_OK | os.X_OK):
                searchable_dirs.append(d)
        except PermissionError:
            continue

    for libname in required:
        for d in searchable_dirs:
            p = d / libname
            try:
                if _load_lib_if_exists(p):
                    loaded.append(str(p))
                    break
            except OSError as err:
                errors.append(f"{p}: {err}")

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(str(d) for d in searchable_dirs)
    if prefix:
        os.environ["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix

    core_names = ("libsharpa-wave-sdk.so",)
    if not all(any(name in item for item in loaded) for name in core_names):
        looked = ", ".join(str(d) for d in searchable_dirs) or "(no readable directories)"
        msg = (
            "Cannot locate/load required Sharpa SDK libs. "
            f"Looked in: {looked}. "
            "The repository deploy/sharpa_sdk/lib directory must contain "
            "the public libsharpa-wave-sdk.so runtime."
        )
        if errors:
            msg += f" First loader error: {errors[0]}"
        raise RuntimeError(msg)


def _is_glove_sn(sn: str) -> bool:
    return sn.strip().upper().startswith("GLOVE")


def _select_target_sn(devices: list[str], requested_sn: str, allow_glove: bool) -> str:
    if requested_sn:
        if requested_sn not in devices:
            raise RuntimeError(f"Requested SN '{requested_sn}' not found. Available devices: {devices}")
        if _is_glove_sn(requested_sn) and not allow_glove:
            raise RuntimeError(
                "The requested SN appears to be a glove device. "
                "Select a hand device or set allow_glove to True."
            )
        return requested_sn
    non_glove = [sn for sn in devices if not _is_glove_sn(sn)]
    if non_glove:
        return non_glove[0]
    if allow_glove:
        return devices[0]
    raise RuntimeError("Only glove devices found. Connect a Wave hand or set allow_glove to True.")


def _discover_devices(manager, timeout_s: float, requested_sn: str = "") -> list[str]:
    t0 = time.time()
    seen: dict[str, None] = {}
    while time.time() - t0 < timeout_s:
        for sn in (manager.get_all_device_sn() or []):
            seen[sn] = None
        if requested_sn and requested_sn in seen:
            break
        time.sleep(0.2)
    return list(seen.keys())


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


def _extract_infer_runtime_cfg(cfg_obj: Any) -> dict[str, Any]:
    train_params = cfg_obj["train"]["params"]
    env_cfg_raw = cfg_obj["task"]["env"]
    sim_cfg_raw = cfg_obj["task"]["sim"]

    train_params_dict = OmegaConf.to_container(train_params, resolve=False)
    env_cfg = OmegaConf.to_container(env_cfg_raw, resolve=False)
    sim_cfg = OmegaConf.to_container(sim_cfg_raw, resolve=False)
    if not isinstance(train_params_dict, dict):
        raise RuntimeError("Failed to parse train.params: expected a dictionary.")
    if not isinstance(env_cfg, dict):
        raise RuntimeError("Failed to parse task.env: expected a dictionary.")
    if not isinstance(sim_cfg, dict):
        raise RuntimeError("Failed to parse task.sim: expected a dictionary.")

    camera_policy_cfg = env_cfg.get("cameraPolicy", {})
    if not isinstance(camera_policy_cfg, dict):
        camera_policy_cfg = {}
    asset_cfg = env_cfg.get("asset", {})
    network_cfg = train_params_dict.setdefault("network", {})
    if not isinstance(asset_cfg, dict):
        asset_cfg = {}
    if not isinstance(network_cfg, dict):
        network_cfg = {}
        train_params_dict["network"] = network_cfg

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
        if isinstance(v, str):
            s = v.strip()
            if s:
                try:
                    return int(float(s))
                except ValueError:
                    return int(default)
        return int(default)

    def _as_float(v: Any, default: float) -> float:
        if isinstance(v, bool):
            return float(int(v))
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if s:
                try:
                    return float(s)
                except ValueError:
                    return float(default)
        return float(default)

    wm_sub = camera_policy_cfg.get("worldModel", {})
    if not isinstance(wm_sub, dict):
        wm_sub = {}
    wm_enabled = _as_bool(network_cfg.get("world_model_enabled", wm_sub.get("enabled", False)), False)
    wm_latent_dim = _as_int(network_cfg.get("wm_latent_dim", wm_sub.get("wm_latent_dim", 16)), 16)
    wm_feature_hidden_raw = network_cfg.get("wm_feature_hidden", wm_sub.get("wm_feature_hidden", [64, 32]))
    if not isinstance(wm_feature_hidden_raw, (list, tuple)):
        wm_feature_hidden_raw = [64, 32]
    wm_feature_hidden = [_as_int(x, 0) for x in wm_feature_hidden_raw]
    wm_feature_hidden = [x for x in wm_feature_hidden if x > 0]
    wm_dyn_deter = _as_int(wm_sub.get("dyn_deter", 512), 512)
    network_cfg["world_model_enabled"] = wm_enabled
    network_cfg["wm_latent_dim"] = wm_latent_dim
    network_cfg["wm_feature_hidden"] = wm_feature_hidden

    control_dt = _as_float(sim_cfg.get("dt", 1.0 / 60.0), 1.0 / 60.0) * _as_float(
        env_cfg.get("controlFrequencyInv", 1), 1.0
    )
    control_hz = 1.0 / max(control_dt, 1e-6)
    # Keep real infer aligned with training camera preprocessing:
    # resize to 96x72 first, then apply pixel crop.
    depth_pre_base_width = 96
    depth_pre_base_height = 72
    depth_pre_cfg_raw = camera_policy_cfg.get("depth_preprocess", {})
    if not isinstance(depth_pre_cfg_raw, dict):
        depth_pre_cfg_raw = {}
    # Crop config lives under depth_preprocess.crop
    depth_crop_cfg_raw = depth_pre_cfg_raw.get("crop", {})
    if not isinstance(depth_crop_cfg_raw, dict):
        depth_crop_cfg_raw = {}
    depth_pre_cfg = {
        "enabled": _as_bool(depth_crop_cfg_raw.get("enabled", False), False),
        "crop_top_px": max(0, _as_int(depth_crop_cfg_raw.get("top_px", 0), 0)),
        "crop_bottom_px": max(0, _as_int(depth_crop_cfg_raw.get("bottom_px", 0), 0)),
        "crop_left_px": max(0, _as_int(depth_crop_cfg_raw.get("left_px", 0), 0)),
        "crop_right_px": max(0, _as_int(depth_crop_cfg_raw.get("right_px", 0), 0)),
        "pre_resize_width": depth_pre_base_width,
        "pre_resize_height": depth_pre_base_height,
    }
    policy_width = depth_pre_base_width
    policy_height = depth_pre_base_height
    if depth_pre_cfg["enabled"]:
        policy_width = depth_pre_base_width - int(depth_pre_cfg["crop_left_px"]) - int(depth_pre_cfg["crop_right_px"])
        policy_height = depth_pre_base_height - int(depth_pre_cfg["crop_top_px"]) - int(depth_pre_cfg["crop_bottom_px"])
        if policy_width <= 0 or policy_height <= 0:
            raise ValueError(
                "Invalid depth_preprocess.crop pixels in run config: "
                f"base=({depth_pre_base_width}, {depth_pre_base_height}), "
                f"left/right=({depth_pre_cfg['crop_left_px']}, {depth_pre_cfg['crop_right_px']}), "
                f"top/bottom=({depth_pre_cfg['crop_top_px']}, {depth_pre_cfg['crop_bottom_px']})."
            )
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

    disable_finger_tactile_obs_reward = _as_bool(
        env_cfg.get("disableTacObs", False),
        False,
    )
    include_target_in_obs = _as_bool(
        env_cfg.get("includeTargetInObs", True),
        True,
    )
    clip_observations = _as_float(env_cfg.get("clipObservations", np.inf), np.inf)
    hand_init_type = str(env_cfg.get("handInit", "default"))
    hand_init_poses_cfg = env_cfg.get("handInitPoses", {})
    if not isinstance(hand_init_poses_cfg, dict):
        raise RuntimeError("Failed to parse task.env.handInitPoses: expected a dictionary.")
    selected_hand_init_deg = hand_init_poses_cfg.get(hand_init_type, None)
    if not isinstance(selected_hand_init_deg, dict):
        raise RuntimeError(
            f"task.env.handInit='{hand_init_type}' is missing from handInitPoses or has an invalid format."
        )
    hand_init_pose_rad: dict[str, float] = {}
    for joint_name in ISAAC_JOINT_NAMES:
        if joint_name not in selected_hand_init_deg:
            raise RuntimeError(f"handInitPoses['{hand_init_type}'] is missing joint: {joint_name}")
        hand_init_pose_rad[joint_name] = float(np.deg2rad(_as_float(selected_hand_init_deg[joint_name], 0.0)))

    return {
        "train_params": train_params_dict,
        "seed": _as_int(cfg_obj.get("seed", 42), 42),
        "world_model_enabled": wm_enabled,
        "world_model": wm_sub,
        "wm_latent_dim": wm_latent_dim,
        "wm_feature_hidden": wm_feature_hidden,
        "wm_dyn_deter": wm_dyn_deter,
        "obs_stack": _as_int(env_cfg.get("obs_stack", 4), 4),
        "clip_observations": clip_observations,
        "disable_finger_tactile_obs_reward": disable_finger_tactile_obs_reward,
        "include_target_in_obs": include_target_in_obs,
        "rel_scale": _as_float(env_cfg.get("relScale", 0.2), 0.2),
        "actions_moving_average": _as_float(env_cfg.get("actionsMovingAverage", 0.8), 0.8),
        "hand_init_type": hand_init_type,
        "hand_init_pose_rad": hand_init_pose_rad,
        "axis": str(env_cfg.get("axis", "z")),
        "max_depth": _as_float(camera_policy_cfg.get("max_depth", 5.0), 5.0),
        "policy_width": policy_width,
        "policy_height": policy_height,
        "camera_raw_width": depth_pre_base_width,
        "camera_raw_height": depth_pre_base_height,
        "depth_preprocess": depth_pre_cfg,
        "train_limit": train_limit_cfg,
        "control_hz": control_hz,
        "dof_limit_scale": _as_float(
            env_cfg.get(
                "dofLimitScale",
                env_cfg.get("dof_limit_scale", 1.0),
            ),
            1.0,
        ),
        "robot_asset_file": str(env_cfg.get("robotAssetFile", "")),
        "asset_file_name": str(asset_cfg.get("assetFileName", "")),
    }


class RealDepthSource:
    def __init__(self, width: int, height: int, fps: int, max_depth: float):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.max_depth = float(max_depth)
        self.pipeline = rs.pipeline()
        # Try requested mode first, then common depth modes.
        # D405 often cannot serve 96x72 directly, so we fallback and resize.
        request_modes = [
            (self.width, self.height, self.fps),
            (640, 480, self.fps),
            (640, 480, 30),
            (640, 360, 30),
            (848, 480, 30),
            (480, 270, 30),
            (424, 240, 30),
            (424, 240, 15),
        ]
        profile = None
        last_exc = None
        used_mode = None
        for w, h, f in request_modes:
            try:
                cfg = rs.config()
                cfg.enable_stream(rs.stream.depth, int(w), int(h), rs.format.z16, int(f))
                profile = self.pipeline.start(cfg)
                used_mode = (int(w), int(h), int(f))
                break
            except RuntimeError as exc:
                last_exc = exc
                continue

        if profile is None:
            dev_info = []
            try:
                for dev in rs.context().query_devices():
                    dev_info.append(
                        {
                            "name": dev.get_info(rs.camera_info.name),
                            "serial": dev.get_info(rs.camera_info.serial_number),
                        }
                    )
            except Exception:
                pass
            raise RuntimeError(
                "Could not start the RealSense depth stream. Original error: "
                f"{last_exc}. Detected devices: {dev_info if dev_info else '[]'}"
            ) from last_exc

        assert used_mode is not None
        self.native_width, self.native_height, self.native_fps = used_mode
        if (self.native_width, self.native_height, self.native_fps) != (self.width, self.height, self.fps):
            print(
                "[WARN] Requested depth mode "
                f"{self.width}x{self.height}@{self.fps} not available, "
                f"fallback to {self.native_width}x{self.native_height}@{self.native_fps} and resize."
            )
        self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

        # RealSense SDK built-in filters (spatial + temporal), matching WMP paper preprocessing.
        # spatial_filter:  edge-preserving smoothing, fills holes within each frame
        # temporal_filter: exponential moving average across frames, suppresses flickering noise
        self._spatial = rs.spatial_filter()
        self._spatial.set_option(rs.option.filter_magnitude, 2)   # smoothing iterations
        self._spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self._spatial.set_option(rs.option.filter_smooth_delta, 20)
        self._temporal = rs.temporal_filter()
        self._temporal.set_option(rs.option.filter_smooth_alpha, 0.4)  # EMA weight for current frame
        self._temporal.set_option(rs.option.filter_smooth_delta, 20)
        print(
            "[INFO] RealSense spatial+temporal filters enabled "
            "(magnitude=2, spatial_alpha=0.5, temporal_alpha=0.4)"
        )

    def get_depth_norm(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames(timeout_ms=2000)
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            raise RuntimeError("D405 did not return a depth frame")
        # Apply SDK-level spatial then temporal filter before converting to numpy.
        depth_frame = self._spatial.process(depth_frame)
        depth_frame = self._temporal.process(depth_frame)
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale
        # RealSense invalid depth pixels are commonly reported as 0.
        # Treat invalid values as "far" instead of "near" to avoid black holes.
        invalid_mask = (~np.isfinite(depth)) | (depth <= 0.0)
        if np.any(invalid_mask):
            depth = depth.copy()
            depth[invalid_mask] = self.max_depth
        depth = np.clip(depth, 0.0, self.max_depth) / self.max_depth
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth_t = torch.from_numpy(depth[None, None, ...])  # [1,1,H,W]
            depth_t = torch.nn.functional.interpolate(
                depth_t,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            depth = depth_t[0, 0].cpu().numpy()
        return depth[..., None]

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


class RealtimeTactileReader:
    """Read tactile force from SDK callback (5 channels)."""

    def __init__(
        self,
        hand_side: int = 1,
        force_scale: float = (1.0 / 1.5),
        contact_threshold: float = 0.05,
        disable_tactile_ids: list[int] | None = None,
    ):
        self.hand_side = int(hand_side)
        self.force_scale = float(force_scale)
        self.contact_threshold = float(contact_threshold)
        self.disable_tactile_ids = set(disable_tactile_ids or [])
        self._lock = threading.Lock()
        # Mirror WaveRealEnv channel convention:
        # right hand (hand_side=1): channels 0..4
        # left hand  (hand_side=0): channels 5..9
        self._base_ch = 5 * (1 - self.hand_side)
        self._forces_by_channel = np.zeros((5,), dtype=np.float32)
        self._initialized = False
        self._last_update_s = 0.0
        self._callback_count = 0

    def callback(self, frame: dict[str, Any]) -> None:
        try:
            ch_raw = int(frame.get("channel", -1))
        except Exception:
            return
        local_ch = ch_raw - self._base_ch
        if local_ch < 0 or local_ch >= 5:
            return

        content = frame.get("content", {})
        f6 = content.get("F6")
        if f6 is None:
            return
        try:
            f6_arr = np.asarray(f6, dtype=np.float32).reshape(-1)
            if f6_arr.shape[0] < 3:
                return
            f_norm = float(np.linalg.norm(f6_arr[:3]))
        except Exception:
            return

        with self._lock:
            self._forces_by_channel[local_ch] = f_norm
            self._initialized = True
            self._last_update_s = time.time()
            self._callback_count += 1

    def get_contact_obs(self) -> np.ndarray:
        # Match rl_isaaclab real obs order: reverse channel order.
        with self._lock:
            force = self._forces_by_channel[::-1].copy()
            initialized = self._initialized
            last_update_s = self._last_update_s

        force *= self.force_scale
        force = np.where(force >= self.contact_threshold, 1.0, 0.0).astype(np.float32, copy=False)
        for idx in self.disable_tactile_ids:
            if 0 <= idx < 5:
                force[idx] = 0.0

        # If callback stream has not started, return zeros gracefully.
        if not initialized or (time.time() - last_update_s > 1.0):
            return np.zeros((5,), dtype=np.float32)
        return force.astype(np.float32, copy=False)

    def get_stats(self) -> tuple[float, int]:
        with self._lock:
            return self._last_update_s, self._callback_count


def build_player(
    checkpoint: str,
    train_params: dict[str, Any],
    obs_dim: int,
    device_name: str,
    world_model_enabled: bool = False,
    wm_latent_dim: int = 16,
    wm_feature_hidden: Optional[list[int]] = None,
    wm_dyn_deter: int = 512,
):
    from rl_games.algos_torch.players import PpoPlayerContinuous

    params: dict[str, Any] = copy.deepcopy(train_params)
    params["config"]["device_name"] = device_name
    params["config"]["env_name"] = "offline_real"
    params["network"]["world_model_enabled"] = bool(world_model_enabled)
    params["network"]["wm_latent_dim"] = int(wm_latent_dim)
    params["network"]["wm_feature_hidden"] = list(wm_feature_hidden or [64, 32])
    params["config"]["player"] = {"deterministic": True, "games_num": 1, "print_stats": False}

    obs_spaces: dict[str, gym.Space] = {
        "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
    }
    if world_model_enabled:
        obs_spaces["wm_feature"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(wm_dyn_deter,), dtype=np.float32,
        )

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


def axis_to_vec(axis: str) -> np.ndarray:
    m = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "-x": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "-y": np.array([0.0, -1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "-z": np.array([0.0, 0.0, -1.0], dtype=np.float32),
    }
    return m[axis]


def _resize_depth_to_hw(depth_hw1: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize depth [H,W,1] to target size with bilinear interpolation."""
    if depth_hw1.shape[0] == out_h and depth_hw1.shape[1] == out_w:
        return depth_hw1
    depth_t = torch.from_numpy(depth_hw1[None, ...]).permute(0, 3, 1, 2).float()
    depth_t = torch.nn.functional.interpolate(
        depth_t,
        size=(int(out_h), int(out_w)),
        mode="bilinear",
        align_corners=False,
    )
    return depth_t.permute(0, 2, 3, 1)[0].cpu().numpy().astype(np.float32)


def _safe_crop_bounds(
    height: int,
    width: int,
    top_px: int,
    bottom_px: int,
    left_px: int,
    right_px: int,
) -> tuple[int, int, int, int]:
    top = max(0, int(top_px))
    bottom = max(0, int(bottom_px))
    left = max(0, int(left_px))
    right = max(0, int(right_px))
    y0 = min(max(0, top), max(0, height - 1))
    y1 = max(y0 + 1, min(height, height - max(0, bottom)))
    x0 = min(max(0, left), max(0, width - 1))
    x1 = max(x0 + 1, min(width, width - max(0, right)))
    return y0, y1, x0, x1


def _apply_depth_preprocess_np(depth_hw1: np.ndarray, depth_pre_cfg: dict[str, Any]) -> np.ndarray:
    if not bool(depth_pre_cfg.get("enabled", False)):
        return depth_hw1.astype(np.float32, copy=False)
    if depth_hw1.ndim != 3 or depth_hw1.shape[-1] != 1:
        raise RuntimeError(f"Unexpected depth shape for preprocess: {depth_hw1.shape}")

    h, w = int(depth_hw1.shape[0]), int(depth_hw1.shape[1])
    top_px = int(depth_pre_cfg.get("crop_top_px", 0))
    bottom_px = int(depth_pre_cfg.get("crop_bottom_px", 0))
    left_px = int(depth_pre_cfg.get("crop_left_px", 0))
    right_px = int(depth_pre_cfg.get("crop_right_px", 0))
    y0, y1, x0, x1 = _safe_crop_bounds(h, w, top_px, bottom_px, left_px, right_px)
    depth_out = depth_hw1[y0:y1, x0:x1, :].astype(np.float32, copy=False)
    return depth_out


def _depth_to_u8(depth_hw1: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_hw1[..., 0], dtype=np.float32)
    depth = np.clip(depth, 0.0, 1.0)
    return (depth * 255.0).astype(np.uint8)


def _build_finger_joint_groups() -> list[tuple[str, list[int]]]:
    finger_keys = [
        ("thumb", "right_thumb"),
        ("index", "right_index"),
        ("middle", "right_middle"),
        ("ring", "right_ring"),
        ("pinky", "right_pinky"),
    ]
    groups: list[tuple[str, list[int]]] = []
    for label, token in finger_keys:
        idxs = [i for i, name in enumerate(ISAAC_JOINT_NAMES) if token in name]
        groups.append((label, idxs))
    return groups


def _save_real_infer_tracking_csv(
    csv_path: Path,
    records: list[dict[str, np.ndarray | float | int]],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    touch_names = ["thumb_touch", "index_touch", "middle_touch", "ring_touch", "pinky_touch"]
    header = ["sample_idx", "step", "time_s"] + touch_names
    for jn in ISAAC_JOINT_NAMES:
        header.extend([f"target_{jn}_deg", f"state_{jn}_deg", f"err_{jn}_deg"])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, rec in enumerate(records):
            target = np.rad2deg(np.asarray(rec["target_isaac"], dtype=np.float32).reshape(-1))
            state = np.rad2deg(np.asarray(rec["state_isaac"], dtype=np.float32).reshape(-1))
            err = np.abs(target - state)
            touch = np.asarray(rec["contact_obs"], dtype=np.float32).reshape(-1)
            row = [int(i), int(rec["step"]), float(rec["time_s"])]
            for k in range(5):
                row.append(float(touch[k]) if k < touch.shape[0] else 0.0)
            for j in range(22):
                row.extend([float(target[j]), float(state[j]), float(err[j])])
            writer.writerow(row)


def _save_real_infer_tracking_plot_png(
    png_path: Path,
    records: list[dict[str, np.ndarray | float | int]],
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    png_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray([float(r["time_s"]) for r in records], dtype=np.float32)
    target_deg = np.rad2deg(np.stack([np.asarray(r["target_isaac"], dtype=np.float32) for r in records], axis=0))
    state_deg = np.rad2deg(np.stack([np.asarray(r["state_isaac"], dtype=np.float32) for r in records], axis=0))
    touch = np.stack([np.asarray(r["contact_obs"], dtype=np.float32) for r in records], axis=0)

    groups = _build_finger_joint_groups()
    max_joint_rows = max(len(idxs) for _, idxs in groups)
    max_rows = 1 + max_joint_rows
    fig, axes = plt.subplots(
        max_rows,
        len(groups),
        figsize=(5.2 * len(groups), 2.8 * max_rows),
        squeeze=False,
    )
    touch_names = ["thumb_touch", "index_touch", "middle_touch", "ring_touch", "pinky_touch"]

    for col, (finger, idxs) in enumerate(groups):
        ax_touch = axes[0][col]
        ax_touch.plot(t, touch[:, col], color="tab:blue", linewidth=1.4)
        ax_touch.set_title(finger)
        ax_touch.set_ylabel(touch_names[col])
        ax_touch.grid(True, alpha=0.25)

        for row in range(1, max_rows):
            ax = axes[row][col]
            ji_local = row - 1
            if ji_local >= len(idxs):
                ax.axis("off")
                continue
            ji = idxs[ji_local]
            jn = ISAAC_JOINT_NAMES[ji]
            ax.plot(t, target_deg[:, ji], label="target", linewidth=1.6, color="tab:orange", linestyle="--")
            ax.plot(t, state_deg[:, ji], label="state", linewidth=1.2, color="tab:blue", linestyle="-")
            mae = float(np.mean(np.abs(target_deg[:, ji] - state_deg[:, ji])))
            ax.set_ylabel(f"{jn.split('right_')[-1]} (deg)", fontsize=8)
            ax.set_title(f"mae={mae:.2f}deg", fontsize=8)
            ax.grid(True, alpha=0.25)
            if row == max_rows - 1:
                ax.set_xlabel("time (s)")
            if row == 1 and col == 0:
                ax.legend(loc="upper right", fontsize=8)

    overall_mae = float(np.mean(np.abs(target_deg - state_deg)))
    fig.suptitle(f"Real Infer Tracking (overall MAE={overall_mae:.3f} deg)", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def build_hardcoded_config() -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[2]
    # Edit parameters here directly; they are no longer supplied through the CLI.
    config = {
        "checkpoint": str(repo_root / "example_ckpt/wm_craftnet_set_z.pth"),
        "cam_width": 640,
        "cam_height": 480,
        "cam_fps": 30,
        "show_depth": True,
        "depth_view_scale": 6,
        "device": "cuda:0",
        "sn": "",
        "allow_glove": False,
        "discover_timeout": 10.0,
        "speed_coef": 0.3,
        "current_coef": 0.5,
        "enable_tactile": True,
        "hand_side": 1,  # 0=left, 1=right
        "force_scale": (1.0 / 1.5),
        "contact_threshold": 0.05,
        "disable_tactile_ids": [],
        "tactile_config_path": "deploy/configs/tactile.json",
        "init_tol": 0.08,
        "init_poll_dt": 0.05,
        "diag_log_interval": 10,
        "save_tracking_plot": True,
        "tracking_output_prefix": "",
        "no_actuation": False,
        "max_steps": 0,  # 0 means run indefinitely
    }
    return SimpleNamespace(**config)


def main() -> None:
    sanitize_colon_path_var("PYTHONPATH", "/_isaac_sim")
    sanitize_colon_path_var("LD_LIBRARY_PATH", "/_isaac_sim")
    _setup_import_paths()
    args = build_hardcoded_config()
    repo_root = Path(__file__).resolve().parents[2]
    _preload_sdk_libs()
    run_cfg_obj, run_cfg_path = _load_run_cfg_from_checkpoint(args.checkpoint)
    infer_cfg = _extract_infer_runtime_cfg(run_cfg_obj)
    print(f"[INFO] Loaded run config: {run_cfg_path}")
    real_infer_out_dir = repo_root / "deploy" / "logs" / "real_infer"

    world_model_enabled = bool(infer_cfg.get("world_model_enabled", False))
    if not world_model_enabled:
        raise RuntimeError("Checkpoint config must enable cameraPolicy.worldModel for WM-Craftnet inference.")
    wm_latent_dim = int(infer_cfg.get("wm_latent_dim", 16))
    wm_feature_hidden = list(infer_cfg.get("wm_feature_hidden", [64, 32]))
    wm_dyn_deter = int(infer_cfg.get("wm_dyn_deter", 512))
    policy_height = int(infer_cfg["policy_height"])
    policy_width = int(infer_cfg["policy_width"])
    camera_raw_height = int(infer_cfg.get("camera_raw_height", policy_height))
    camera_raw_width = int(infer_cfg.get("camera_raw_width", policy_width))
    depth_preprocess_cfg = infer_cfg.get("depth_preprocess", {})
    if not isinstance(depth_preprocess_cfg, dict):
        depth_preprocess_cfg = {}
    rel_scale = float(infer_cfg["rel_scale"])
    actions_moving_average = float(infer_cfg["actions_moving_average"])
    clip_observations = float(infer_cfg.get("clip_observations", np.inf))
    control_hz = float(infer_cfg["control_hz"])
    axis = str(infer_cfg["axis"])
    max_depth = float(infer_cfg["max_depth"])
    dof_limit_scale = float(infer_cfg.get("dof_limit_scale", 1.0))
    seed = int(infer_cfg["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)

    from sharpa import (  # pylint: disable=import-error
        ControlMode,
        ControlSource,
        SharpaWaveConfig,
        SharpaWaveManager,
    )

    urdf_lower, urdf_upper = _load_dof_limits_from_asset_urdf(
        repo_root=repo_root,
        robot_asset_file=str(infer_cfg.get("robot_asset_file", "")),
        fallback_asset_file=str(infer_cfg.get("asset_file_name", "")),
    )
    train_limit_cfg = infer_cfg.get("train_limit", {})
    urdf_lower, urdf_upper = _apply_train_limit_tightening(
        lower=urdf_lower,
        upper=urdf_upper,
        train_limit_cfg=train_limit_cfg if isinstance(train_limit_cfg, dict) else {},
    )
    low = urdf_lower * dof_limit_scale
    high = urdf_upper * dof_limit_scale
    disable_finger_tactile_obs_reward = bool(infer_cfg.get("disable_finger_tactile_obs_reward", False))
    include_target_in_obs = bool(infer_cfg.get("include_target_in_obs", True))
    finger_tactile_obs_dim = 0 if disable_finger_tactile_obs_reward else 5
    target_obs_dim = 22 if include_target_in_obs else 0
    n_base_obs = 22 + target_obs_dim + finger_tactile_obs_dim + 24
    n_stack = int(infer_cfg["obs_stack"])
    obs_dim = n_base_obs * n_stack
    print(
        f"[INFO] finger_tactile_obs_dim={finger_tactile_obs_dim} "
        f"| includeTargetInObs={include_target_in_obs} "
        f"(disableTacObs={disable_finger_tactile_obs_reward}) "
        f"| n_base_obs={n_base_obs} obs_stack={n_stack} obs_dim={obs_dim}"
    )
    print(
        f"[INFO] clipObservations={clip_observations} | "
        f"handInit='{infer_cfg.get('hand_init_type', 'default')}' (from config.yaml)"
    )

    depth_source = RealDepthSource(args.cam_width, args.cam_height, args.cam_fps, max_depth)
    player = build_player(
        checkpoint=os.path.abspath(args.checkpoint),
        train_params=infer_cfg["train_params"],
        obs_dim=obs_dim,
        device_name=args.device,
        world_model_enabled=world_model_enabled,
        wm_latent_dim=wm_latent_dim,
        wm_feature_hidden=wm_feature_hidden,
        wm_dyn_deter=wm_dyn_deter,
    )
    # WorldModel adapter for real-time inference (runs on device, no training)
    wm_adapter_rt = None
    if world_model_enabled:
        from isaacgymenvs.utils.world_model import WorldModelAdapter
        wm_adapter_rt = WorldModelAdapter(
            repo_root=repo_root,
            num_envs=1,
            num_actions=22,
            prop_dim=n_base_obs,
            image_shape=(policy_height, policy_width, 1),
            wm_cfg=infer_cfg["world_model"],
            device=args.device,
        )
        ckpt_data = torch.load(os.path.abspath(args.checkpoint), map_location=args.device, weights_only=False)
        if "world_model" in ckpt_data:
            wm_adapter_rt.load_state_dict(ckpt_data["world_model"], load_optimizer=False)
            print(f"[INFO] WorldModel loaded from checkpoint ({wm_adapter_rt.wm_feature_dim}-dim deter feature)")
        else:
            raise RuntimeError("WM-Craftnet checkpoint does not contain world_model state.")

    manager = SharpaWaveManager.get_instance()
    devices = _discover_devices(manager, timeout_s=args.discover_timeout, requested_sn=args.sn)
    if not devices:
        raise RuntimeError("No Sharpa device found.")
    sn = _select_target_sn(devices, args.sn, args.allow_glove)
    sdk_config = SharpaWaveConfig()
    sdk_config.disable_tactile = not args.enable_tactile
    if args.enable_tactile and args.tactile_config_path:
        tactile_config_path = Path(args.tactile_config_path).expanduser()
        if not tactile_config_path.is_absolute():
            tactile_config_path = repo_root / tactile_config_path
        if not tactile_config_path.is_file():
            raise RuntimeError(f"Tactile config not found: {tactile_config_path}")
        sdk_config.tactile_config_file = str(tactile_config_path.resolve())
        print(f"[INFO] Using public SDK tactile config: {sdk_config.tactile_config_file}")
    hand = manager.connect(sn, sdk_config)

    obs_hist: deque[np.ndarray] = deque(maxlen=n_stack)
    last_action = np.zeros((22,), dtype=np.float32)
    tactile_reader: RealtimeTactileReader | None = None
    tracking_records: list[dict[str, np.ndarray | float | int]] = []
    tracking_t0 = time.perf_counter()
    cv2_mod = None
    depth_window_name = f"policy_depth_{policy_width}x{policy_height}"
    if args.show_depth:
        try:
            import cv2 as cv2_mod_import  # type: ignore

            cv2_mod = cv2_mod_import
            print(
                f"[INFO] show-depth enabled: window='{depth_window_name}', "
                f"source={policy_width}x{policy_height}, scale={max(1, int(args.depth_view_scale))}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] show_depth is enabled, but cv2 import failed; skipping display: {exc}")
            cv2_mod = None

    try:
        if args.no_actuation:
            print("[WARN] no-actuation enabled: skipping all set_joint_position commands.")
        hand.set_control_mode(ControlMode.POSITION)
        hand.set_control_source(ControlSource.SDK)
        hand.set_speed_coeff(args.speed_coef)
        hand.set_current_coeff(args.current_coef)
        if args.enable_tactile:
            tactile_reader = RealtimeTactileReader(
                hand_side=args.hand_side,
                force_scale=args.force_scale,
                contact_threshold=args.contact_threshold,
                disable_tactile_ids=args.disable_tactile_ids,
            )
            hand.set_tactile_callback(tactile_reader.callback)
            hand.start()
        if not args.no_actuation:
            zero_target_isaac = np.zeros((22,), dtype=np.float32)
            zero_cmd_ret = hand.set_joint_position(isaac_to_sharpa(zero_target_isaac).tolist())
            zero_code, zero_msg = _err_code_msg(zero_cmd_ret)
            print(f"[INFO] zero-pose command sent | cmd_code={zero_code} | cmd_msg='{zero_msg}'")
            _ok(zero_cmd_ret, "set_joint_position(zero_pose_once)")
            time.sleep(0.2)
        else:
            print("[INFO] no-actuation: skipping the startup zero-pose command.")

        sim_init_isaac = get_sim_init_pose_isaac(
            infer_cfg.get("hand_init_pose_rad", {})
        ).astype(np.float32, copy=False)
        sim_init_isaac = np.clip(sim_init_isaac, low, high)
        _move_to_pose_interp_and_wait(
            hand=hand,
            target_isaac=sim_init_isaac,
            enable_actuation=not args.no_actuation,
            tol_rad=float(args.init_tol),
            poll_dt_s=float(args.init_poll_dt),
        )
        print("[INFO] Initial pose reached. Starting inference after a 3-second pause.")
        time.sleep(3.0)
        init_qpos_sharpa = np.asarray(hand.get_states().angles, dtype=np.float32)
        init_qpos_isaac = sharpa_to_isaac(init_qpos_sharpa)
        init_qpos_isaac = np.clip(init_qpos_isaac, low, high)
        prev_target_isaac = init_qpos_isaac.copy()
        print(
            "[INFO] prev_target init from measured qpos | "
            f"max|qpos-sim_init|={float(np.max(np.abs(init_qpos_isaac - sim_init_isaac))):.4f}rad"
        )
        spin_axis = np.tile(axis_to_vec(axis), 8)
        for _ in range(n_stack):
            qpos_obs = scale_to_minus1_plus1(prev_target_isaac, low, high)
            target_obs = scale_to_minus1_plus1(prev_target_isaac, low, high)
            if include_target_in_obs:
                policy_target_obs = target_obs
            contact_obs = (
                tactile_reader.get_contact_obs()
                if tactile_reader is not None
                else np.zeros((5,), dtype=np.float32)
            )
            if finger_tactile_obs_dim > 0:
                if include_target_in_obs:
                    base = np.concatenate([qpos_obs, policy_target_obs, contact_obs, spin_axis], axis=0)
                else:
                    base = np.concatenate([qpos_obs, contact_obs, spin_axis], axis=0)
            else:
                if include_target_in_obs:
                    base = np.concatenate([qpos_obs, policy_target_obs, spin_axis], axis=0)
                else:
                    base = np.concatenate([qpos_obs, spin_axis], axis=0)
            obs_hist.appendleft(base.astype(np.float32))

        # Log a strict "post-reset, pre-first-action" step=0 snapshot for sim/real alignment.
        init_state_sharpa = np.asarray(hand.get_states().angles, dtype=np.float32)
        init_state_isaac = np.clip(sharpa_to_isaac(init_state_sharpa), low, high)
        init_contact_obs = (
            tactile_reader.get_contact_obs()
            if tactile_reader is not None
            else np.zeros((5,), dtype=np.float32)
        )
        tracking_records.append(
            {
                "step": int(0),
                "time_s": float(time.perf_counter() - tracking_t0),
                "contact_obs": init_contact_obs.copy(),
                "target_isaac": prev_target_isaac.copy(),
                "state_isaac": init_state_isaac.copy(),
            }
        )

        dt = 1.0 / max(control_hz, 1e-6)
        period_timeout_threshold_s = 0.1
        period_timeout_count = 0
        diag_log_interval = max(1, int(args.diag_log_interval))
        step = 1
        tactile_stat_t = time.time()
        tactile_stat_step = 1
        while True:
            # step=0 is the pre-action snapshot already logged above.
            # Keep max_steps semantics as "number of control actions".
            if args.max_steps > 0 and (step - 1) >= args.max_steps:
                break
            loop_t0 = time.perf_counter()

            qpos_sharpa = np.asarray(hand.get_states().angles, dtype=np.float32)
            qpos_isaac = sharpa_to_isaac(qpos_sharpa)
            depth_norm = depth_source.get_depth_norm()
            depth_norm = _resize_depth_to_hw(depth_norm, camera_raw_height, camera_raw_width)
            depth_norm = _apply_depth_preprocess_np(depth_norm, depth_preprocess_cfg)
            if depth_norm.shape[0] != policy_height or depth_norm.shape[1] != policy_width:
                raise RuntimeError(
                    "Depth preprocess output shape mismatch in real infer: "
                    f"got ({depth_norm.shape[1]}x{depth_norm.shape[0]}), "
                    f"expected ({policy_width}x{policy_height}). "
                    "Check the cameraPolicy.depth_preprocess.crop_*_px settings."
                )
            # Keep visualization semantics unchanged (near=0, far=1),
            # but flip only model input so near objects get higher response.
            depth_for_model = 1.0 - depth_norm
            if cv2_mod is not None:
                depth_u8 = _depth_to_u8(depth_norm)
                view_scale = max(1, int(args.depth_view_scale))
                if view_scale > 1:
                    depth_u8 = cv2_mod.resize(
                        depth_u8,
                        (policy_width * view_scale, policy_height * view_scale),
                        interpolation=cv2_mod.INTER_NEAREST,
                    )
                cv2_mod.imshow(depth_window_name, depth_u8)
                key = cv2_mod.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("[INFO] depth window requested exit (ESC/q).")
                    break
            qpos_obs = scale_to_minus1_plus1(qpos_isaac, low, high)
            target_obs = scale_to_minus1_plus1(prev_target_isaac, low, high)
            if include_target_in_obs:
                policy_target_obs = target_obs
            contact_obs = (
                tactile_reader.get_contact_obs()
                if tactile_reader is not None
                else np.zeros((5,), dtype=np.float32)
            )
            if finger_tactile_obs_dim > 0:
                if include_target_in_obs:
                    base = np.concatenate([qpos_obs, policy_target_obs, contact_obs, spin_axis], axis=0)
                else:
                    base = np.concatenate([qpos_obs, contact_obs, spin_axis], axis=0)
            else:
                if include_target_in_obs:
                    base = np.concatenate([qpos_obs, policy_target_obs, spin_axis], axis=0)
                else:
                    base = np.concatenate([qpos_obs, spin_axis], axis=0)
            obs_hist.appendleft(base.astype(np.float32))
            obs_vec = np.concatenate(list(obs_hist), axis=0).astype(np.float32)
            obs_vec = np.clip(obs_vec, -clip_observations, clip_observations)
            obs_dict: dict[str, np.ndarray] = {"obs": obs_vec[None, :]}

            if wm_adapter_rt is not None:
                prop_t = torch.from_numpy(base[None, :].astype(np.float32)).to(args.device)
                img_t = torch.from_numpy(depth_for_model[None, ...].astype(np.float32)).to(args.device)
                wm_h = wm_adapter_rt.step(prop_t, img_t)
                obs_dict["wm_feature"] = wm_h.detach().cpu().numpy()

            # rl_games get_action expects a torch.Tensor (it calls .size()).
            # Use the player's standard tensor conversion to avoid passing numpy directly.
            infer_t0 = time.perf_counter()
            obs_torch = player.obs_to_torch({"obs": obs_dict})
            action = player.get_action(obs_torch, is_deterministic=True)
            action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
            infer_s = time.perf_counter() - infer_t0
            action_abs = np.abs(action_np)
            action_abs_max = float(np.max(action_abs)) if action_abs.size > 0 else 0.0
            action_abs_mean = float(np.mean(action_abs)) if action_abs.size > 0 else 0.0
            action_over_1_count = int(np.sum(action_abs > (0.95 + 1e-6)))
            if action_over_1_count > 0:
                print(
                    "[WARN] action exceeds [-1,1], apply safety clip: "
                    f"count={action_over_1_count}, max_abs={action_abs_max:.6f}"
                )
                action_np = np.clip(action_np, -1.0, 1.0)

            smooth = action_np * actions_moving_average + last_action * (1.0 - actions_moving_average)
            target_raw = prev_target_isaac + rel_scale * smooth
            target = target_raw.copy()
            target = np.clip(target, low, high)
            clip_delta = np.abs(target_raw - target)
            clipped_joint_count = int(np.sum(clip_delta > 1e-7))
            max_clip = float(np.max(clip_delta)) if clip_delta.size > 0 else 0.0
            pre_send_err = float(np.max(np.abs(target - qpos_isaac)))

            cmd_code = 0
            cmd_msg = ""
            if not args.no_actuation:
                cmd_ret = hand.set_joint_position(isaac_to_sharpa(target).tolist())
                cmd_code, cmd_msg = _err_code_msg(cmd_ret)
                _ok(cmd_ret, "set_joint_position(loop)")
            tracking_records.append(
                {
                    "step": int(step),
                    "time_s": float(time.perf_counter() - tracking_t0),
                    "contact_obs": contact_obs.copy(),
                    "target_isaac": target.copy(),
                    "state_isaac": qpos_isaac.copy(),
                }
            )
            prev_target_isaac = target.copy()
            last_action = smooth.copy()
            if wm_adapter_rt is not None:
                wm_adapter_rt.cache_prev_action(
                    torch.from_numpy(action_np[None, :].astype(np.float32)).to(args.device)
                )

            loop_exec_s = time.perf_counter() - loop_t0
            sleep_t = dt - loop_exec_s
            if sleep_t < 0.0:
                sleep_t = 0.0
            cycle_total_s = loop_exec_s + sleep_t
            period_overrun_s = max(0.0, cycle_total_s - period_timeout_threshold_s)
            period_is_timeout = cycle_total_s > period_timeout_threshold_s
            if period_is_timeout:
                period_timeout_count += 1

            step += 1
            if step % diag_log_interval == 0:
                contact_max = float(np.max(contact_obs))
                tactile_01 = ",".join(str(int(value)) for value in contact_obs)
                tactile_obs_fps = 0.0
                tactile_stale = False
                now_s = time.time()
                dt_stats = max(1e-6, now_s - tactile_stat_t)
                tactile_obs_fps = float(step - tactile_stat_step) / dt_stats
                if tactile_reader is not None:
                    last_update_s, _ = tactile_reader.get_stats()
                    tactile_stale = (now_s - last_update_s) > 1.0
                tactile_stat_t = now_s
                tactile_stat_step = step
                print(
                    f"[INFO][diag] step={step} | action_abs_mean={action_abs_mean:.4f} "
                    f"| action_abs_max={action_abs_max:.4f} | action_over_1_count={action_over_1_count} "
                    f"| smooth_norm={float(np.linalg.norm(smooth)):.4f} "
                    f"| pre_send_err_max={pre_send_err:.4f}rad "
                    f"| clipped_joints={clipped_joint_count}/22 | max_clip={max_clip:.4f}rad "
                    f"| cmd_code={cmd_code} | cmd_msg='{cmd_msg}' "
                    f"| tactile_01[thumb,index,middle,ring,pinky]=[{tactile_01}] "
                    f"| tactile_max={contact_max:.4f} | tactile_obs_fps={tactile_obs_fps:.2f} "
                    f"| tactile_stale={int(tactile_stale)} "
                    f"| infer_ms={infer_s * 1000.0:.2f} "
                    f"| cycle_exec_ms={loop_exec_s * 1000.0:.2f} "
                    f"| cycle_total_ms={cycle_total_s * 1000.0:.2f} "
                    f"| over_100ms={int(period_is_timeout)} "
                    f"| over_100ms_ms={period_overrun_s * 1000.0:.2f} "
                    f"| over_100ms_count={period_timeout_count}/{step}"
                )

            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        if cv2_mod is not None:
            try:
                cv2_mod.destroyWindow(depth_window_name)
            except Exception:
                pass
        if args.save_tracking_plot and tracking_records:
            stem = (
                args.tracking_output_prefix.strip()
                if args.tracking_output_prefix.strip()
                else f"real_infer_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            csv_path = real_infer_out_dir / f"{stem}.csv"
            png_path = real_infer_out_dir / f"{stem}.png"
            try:
                _save_real_infer_tracking_csv(csv_path, tracking_records)
                print(f"[INFO] Real infer tracking CSV saved: {csv_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save real infer tracking CSV: {exc}")
            try:
                _save_real_infer_tracking_plot_png(png_path, tracking_records)
                print(f"[INFO] Real infer tracking PNG saved: {png_path}")
            except Exception as exc:
                print(f"[WARN] Failed to save real infer tracking PNG (CSV retained if possible): {exc}")
        depth_source.close()
        try:
            hand.stop()
        except Exception:
            pass
        try:
            manager.disconnect_all()
        except Exception:
            pass


if __name__ == "__main__":
    main()
