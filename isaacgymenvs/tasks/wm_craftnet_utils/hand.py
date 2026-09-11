import math
import os
from functools import lru_cache

import yaml


_DEFAULT_HAND_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "cfg", "task", "common", "hand.yaml")
)


@lru_cache(maxsize=4)
def load_hand_config(path=None):
    config_path = os.path.abspath(path or _DEFAULT_HAND_CONFIG)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def hand_init_poses_rad(config):
    poses = config.get("handInitPoses", {})
    parsed = {}
    for pose_name, joint_map in poses.items():
        parsed[str(pose_name)] = {
            str(joint_name): math.radians(float(joint_value))
            for joint_name, joint_value in joint_map.items()
        }
    return parsed


def get_hand_init_pose(config, pose_name):
    poses = hand_init_poses_rad(config)
    if pose_name not in poses:
        raise KeyError(f"Unknown handInit '{pose_name}'. Available: {sorted(poses.keys())}")
    return poses[pose_name]


def build_pd_gain_lists(config, joint_names, num_actions):
    gain_table = config.get("pdGains", {})
    p_gains = []
    d_gains = []
    for joint_name in joint_names[:num_actions]:
        if joint_name not in gain_table:
            raise KeyError(f"Missing PD gain entry for joint: {joint_name}")
        gain_cfg = gain_table[joint_name]
        p_gains.append(float(gain_cfg["p"]))
        d_gains.append(float(gain_cfg["d"]))
    return p_gains, d_gains


def apply_dof_runtime_params(robot_dof_props, joint_names, arm_dof_num, config):
    runtime_params = config.get("dofRuntime", {})
    for i, joint_name in enumerate(joint_names):
        if i < arm_dof_num:
            robot_dof_props["velocity"][i] = 1.0
            robot_dof_props["effort"][i] = 20.0
            robot_dof_props["friction"][i] = 0.1
            robot_dof_props["stiffness"][i] = 0.0
            robot_dof_props["armature"][i] = 0.1
            robot_dof_props["damping"][i] = 100.0
            continue

        joint_params = runtime_params.get(joint_name)
        if joint_params is None:
            robot_dof_props["velocity"][i] = 3.0
            robot_dof_props["effort"][i] = 0.5
            robot_dof_props["friction"][i] = 0.1
            robot_dof_props["stiffness"][i] = 0.0
            robot_dof_props["armature"][i] = 0.1
            robot_dof_props["damping"][i] = 0.0
            continue

        for prop_name in ("velocity", "effort", "friction", "stiffness", "armature", "damping"):
            robot_dof_props[prop_name][i] = float(joint_params[prop_name])
    return robot_dof_props
