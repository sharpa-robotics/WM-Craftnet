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
import pickle

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.recorders.evaluation import flush_spin_trace_episode
import pytorch3d.transforms as transform
import torch.nn.functional as F
import json
import xml.etree.ElementTree as ET

import random

import math
import time
import trimesh
from collections.abc import Sequence
from isaacgymenvs.utils.torch_jit_utils import conditional_jit
from isaacgymenvs.utils.spin_reward_utils import (
    compute_spin_deltas_from_rot_mats,
)
from isaacgymenvs.tasks.wm_craftnet_utils.hand import (
    apply_dof_runtime_params,
    build_pd_gain_lists,
    get_hand_init_pose,
    load_hand_config,
)

def degrees_to_radians(degrees_list):
    return [math.radians(deg) for deg in degrees_list]

def read_dict_from_json(file_path):
	# Opening JSON file
	with open(file_path) as json_file:
		data = json.load(json_file)
	return data


def _sample_unit_ball_points(num_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    pts = []
    while len(pts) < num_points:
        cand = rng.uniform(-1.0, 1.0, size=(num_points * 2, 3))
        mask = np.sum(cand * cand, axis=-1) <= 1.0
        if np.any(mask):
            pts.append(cand[mask])
        merged = np.concatenate(pts, axis=0) if len(pts) > 0 else np.zeros((0, 3), dtype=np.float32)
        if merged.shape[0] >= num_points:
            return merged[:num_points].astype(np.float32)
    return np.zeros((num_points, 3), dtype=np.float32)


def xyzw_to_wxyz(quat):
    # holy****, isaacgym uses xyzw format. pytorch3d uses wxyz format.
    new_quat = quat.clone()
    new_quat[:, :1] = quat[:, -1:]
    new_quat[:, 1:] = quat[:, :-1]
    return new_quat


# Debug script: python ./isaacgymenvs/train.py test=False task=AllegroArmLeftContinuous pipeline=cpu
class RealmanSharpaHa4Rotation(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.time = time.time()
        
        self.training = True
        self.cfg = cfg
        self.cfg["env"]["legacy_obs"] = False
        self.use_default_ground_plane = bool(self.cfg["env"].get("useDefaultGroundPlane", False))
        self.white_ground_enabled = bool(self.cfg["env"].get("whiteGroundEnabled", True))
        self.white_ground_size = float(self.cfg["env"].get("whiteGroundSize", 20.0))
        self.white_ground_thickness = float(self.cfg["env"].get("whiteGroundThickness", 0.004))
        self.white_ground_z = float(self.cfg["env"].get("whiteGroundZ", 0.0))

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        pd_randomization_cfg = self.cfg["env"].get("pdRandomization", {})
        if not isinstance(pd_randomization_cfg, dict):
            pd_randomization_cfg = {}
        self.pd_dr_enabled = bool(pd_randomization_cfg.get("enabled", False))
        pd_scale_range = pd_randomization_cfg.get("scale_range", [0.8, 1.2])
        if (
            not isinstance(pd_scale_range, Sequence)
            or isinstance(pd_scale_range, (str, bytes))
            or len(pd_scale_range) != 2
        ):
            raise ValueError(
                f"pdRandomization.scale_range must be a length-2 list/tuple, got: {pd_scale_range}"
            )
        self.pd_dr_scale_lower = float(pd_scale_range[0])
        self.pd_dr_scale_upper = float(pd_scale_range[1])
        if self.pd_dr_scale_lower > self.pd_dr_scale_upper:
            raise ValueError(
                f"Invalid pdRandomization scale range: {self.pd_dr_scale_lower} > {self.pd_dr_scale_upper}"
            )
        self.pd_dr_min_gain = float(pd_randomization_cfg.get("min_gain", 1e-6))

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.control_penalty_scale = self.cfg["env"]["controlPenaltyScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.hand_pose_coef = self.cfg["env"].get("handPoseCoef", 0.0)
        # Penalty for object drifting from episode initial position: -coef * ||p - p0||^2 (per env).
        self.obj_init_pos_dev_coef = float(self.cfg["env"].get("objInitPosDeviationCoef", 0.0))
        # Reset if object does not spin around the target axis for too many consecutive steps.
        self.no_spin_angvel_thresh = float(self.cfg["env"].get("noSpinAngVelThresh", 0.5))
        self.no_spin_max_steps = int(self.cfg["env"].get("noSpinMaxSteps", 60))
        self.no_spin_reset_penalty = float(self.cfg["env"].get("noSpinResetPenalty", -10.0))
        gravity_curriculum_cfg = self.cfg["env"].get("gravityCurriculum", {})
        self.gravity_curriculum_enabled = bool(gravity_curriculum_cfg.get("enabled", False))
        self.gravity_curriculum_start_max_angle_deg = float(gravity_curriculum_cfg.get("startMaxAngleDeg", 2.0))
        self.gravity_curriculum_angle_step_deg = float(gravity_curriculum_cfg.get("angleStepDeg", 1.0))
        self.gravity_curriculum_max_angle_deg = float(gravity_curriculum_cfg.get("maxAngleDeg", 15.0))
        self.gravity_curriculum_ratio_threshold = float(gravity_curriculum_cfg.get("fullRunRatioThreshold", 0.85))
        self.gravity_curriculum_window_episodes = max(1, int(gravity_curriculum_cfg.get("windowEpisodes", 512)))
        self.action_conservative_scale_start = float(gravity_curriculum_cfg.get("actionScaleStart", 0.1))
        self.action_conservative_scale_step = float(gravity_curriculum_cfg.get("actionScaleStep", 0.1))
        self.action_conservative_scale_max = float(gravity_curriculum_cfg.get("actionScaleMax", 10.0))
        self.action_conservative_scale = min(
            self.action_conservative_scale_start, self.action_conservative_scale_max
        )
        self.gravity_curriculum_current_max_angle_deg = min(
            self.gravity_curriculum_start_max_angle_deg, self.gravity_curriculum_max_angle_deg
        )
        self.gravity_curriculum_upgrade_count = 0
        self.gravity_curriculum_full_run_ratio = 0.0
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.m_lower = self.cfg["env"].get("m_low", 0.03)
        self.m_upper = self.cfg["env"].get("m_up", 0.3)

        self.relative_scale = self.cfg["env"].get("relScale", 0.5)
        self.absolute_scale = float(self.cfg["env"].get("absScale", 0.8))
        spin_trace_cfg = self.cfg["env"].get("spinTrace", {})
        # spinTrace is a test/eval diagnostic only.
        # Training phase must keep it disabled even if config says True.
        self.spin_trace_cfg_enabled = bool(spin_trace_cfg.get("enabled", False))
        self.spin_trace_enabled = self.spin_trace_cfg_enabled and bool(self.cfg["env"].get("isTestRun", False))
        # Single-env legacy fields kept so existing test/replay paths still work
        # via spin_trace_env_id / spin_trace_episode_idx; multi-env writers use
        # spin_trace_env_ids and the per-env _spin_trace_* dicts below.
        self.spin_trace_env_id = int(spin_trace_cfg.get("envId", 0))
        self.spin_trace_env_ids = [self.spin_trace_env_id]
        self.spin_trace_output_dir = str(spin_trace_cfg.get("outputDir", "outputs/spin_trace"))
        self.spin_trace_episode_idx = int(spin_trace_cfg.get("startEpisodeIdx", 0))
        # Per-env row buffers and episode counters (default mirrors the
        # single-env legacy behaviour for envId).
        self._spin_trace_rows_per_env = {self.spin_trace_env_id: []}
        self._spin_trace_episode_idx_per_env = {self.spin_trace_env_id: self.spin_trace_episode_idx}
        # Optional per-env override of where each (env, episode) artifact lands.
        # Subclasses (e.g. camera task) populate this so that spin_trace.csv/png
        # land next to the matching demo video.
        self._spin_trace_output_path_resolver = None
        # Legacy single-env row buffer kept for backward-compatible paths;
        # always points at the first env in spin_trace_env_ids.
        self._spin_trace_rows = self._spin_trace_rows_per_env[self.spin_trace_env_id]
        self._spin_trace_finger_layout = None

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        # resetPositionNoise: per-axis half-width [ax, ay, az]; final ~ shift + U(-a, +a).
        # Stored as a plain tuple here; converted to a device tensor after super().__init__().
        _reset_pos_noise_cfg = self.cfg["env"]["resetPositionNoise"]
        assert len(_reset_pos_noise_cfg) == 3, \
            f"resetPositionNoise must be a length-3 list [ax, ay, az], got {_reset_pos_noise_cfg}"
        self._reset_position_noise_cfg = tuple(float(v) for v in _reset_pos_noise_cfg)
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.random_force_prob_scalar = self.cfg["env"].get("forceProbScalar", 0.25)
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        perturb_cfg = self.cfg["env"].get("inProcessPerturbation", {})
        if not isinstance(perturb_cfg, dict):
            perturb_cfg = {}
        self.in_process_perturb_enabled = bool(perturb_cfg.get("enabled", False))
        self.perturb_interval_control_steps = int(perturb_cfg.get("intervalControlSteps", 200))
        self.perturb_distance = float(perturb_cfg.get("distance", 0.03))
        perturb_direction = perturb_cfg.get("direction", [1.0, 0.0, 0.0])
        if (
            not isinstance(perturb_direction, Sequence)
            or isinstance(perturb_direction, (str, bytes))
            or len(perturb_direction) != 3
        ):
            raise ValueError(
                f"inProcessPerturbation.direction must be a length-3 list, got {perturb_direction}"
            )
        self.perturb_direction_cfg = tuple(float(v) for v in perturb_direction)
        self.rotation_axis = self.cfg["env"]["axis"]

        friction_randomization_cfg = self.cfg["env"].get("frictionRandomization", {})
        if not isinstance(friction_randomization_cfg, dict):
            friction_randomization_cfg = {}
        friction_scale_range = friction_randomization_cfg.get("scale_range", [0.5, 2.0])
        if len(friction_scale_range) != 2:
            raise ValueError(
                f"frictionRandomization.scale_range must have 2 values, got: {friction_scale_range}"
            )
        self.friction_scale_lower = float(friction_scale_range[0])
        self.friction_scale_upper = float(friction_scale_range[1])
        if self.friction_scale_lower > self.friction_scale_upper:
            raise ValueError(
                f"Invalid friction scale range: {self.friction_scale_lower} > {self.friction_scale_upper}"
            )
        self.object_base_friction = float(friction_randomization_cfg.get("object_base", 0.5))
        self.hand_other_base_friction = float(friction_randomization_cfg.get("hand_metal_base", 0.1))
        self.hand_elastomer_base_friction = float(friction_randomization_cfg.get("hand_elastomer_base", 0.8))
        default_elastomer_links = [
            "right_thumb_elastomer",
            "right_index_elastomer",
            "right_middle_elastomer",
            "right_ring_elastomer",
            "right_pinky_elastomer",
        ]
        elastomer_links = friction_randomization_cfg.get("elastomer_links", default_elastomer_links)
        if isinstance(elastomer_links, str):
            elastomer_links = [elastomer_links]
        elif not isinstance(elastomer_links, (list, tuple, set)):
            elastomer_links = default_elastomer_links
        self.hand_elastomer_links = set(elastomer_links)
        self.hand_elastomer_shape_ids = set()
        self.hand_other_shape_ids = set()
        self._printed_hand_friction_shape_groups = False
        self.randomize_mass_lower = self.m_lower
        self.randomize_mass_upper = self.m_upper

        self.tac_thresh = float(self.cfg["env"].get("TacThresh", 1.0))
        self.tac_thresh_rand = float(self.cfg["env"].get("tacThreshRand", 0.8))
        self.sensor_noise = self.cfg["env"].get("sensorNoise", 0.2)

        if self.rotation_axis == "x":
            self.rotation_id = 0
        elif self.rotation_axis == "y":
            self.rotation_id = 1
        else:
            self.rotation_id = 2

        train_limit_cfg = self.cfg["env"].get("trainLimit", {})
        if not isinstance(train_limit_cfg, dict):
            train_limit_cfg = {}
        self.train_limit_global_scale = float(train_limit_cfg.get("global_scale", 1.0))
        unilateral_cfg = train_limit_cfg.get("unilateral_override_deg", {})
        if not isinstance(unilateral_cfg, dict):
            unilateral_cfg = {}
        self.train_limit_unilateral_override_deg = unilateral_cfg
        self.control_mode = str(self.cfg["env"].get("controlMode", "relative")).lower()
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError(f"Unsupported controlMode: {self.control_mode}, expected relative/absolute")
        self.use_relative_control = (self.control_mode == "relative")
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.spin_coef = self.cfg["env"].get("spin_coef", 1.0)
        self.reward_max_spin_rate = float(self.cfg["env"].get("reward_max_spin_rate", 2.0))
        self.contact_coef = self.cfg["env"].get("contact_coef", 1.0)
        self.vel_coef = self.cfg["env"].get("vel_coef", -0.3)
        self.torque_coef = self.cfg["env"].get("torque_coef", -0.01)
        self.work_coef = self.cfg['env'].get("work_coef", -0.0002)
        self.finger_coef = self.cfg['env'].get('finger_coef', 0.1)
        self.distance_reward_deadzone = float(self.cfg["env"].get("distanceRewardDeadzone", 0.025))
        self.distance_reward_penalty_width = float(self.cfg["env"].get("distanceRewardPenaltyWidth", 0.03))
        self.disable_tac_obs = bool(self.cfg["env"].get("disableTacObs", False))
        self.finger_tactile_obs_dim = 0 if self.disable_tac_obs else 5
        self.palm_root_contact_coef = float(self.cfg["env"].get("palm_root_contact_coef", 0.0))
        self.palm_root_contact_force_thresh = 0.5
        self.thumb_non_tip_contact_penalty = float(
            self.cfg["env"].get("thumbNonTipContactPenalty", 0.0)
        )
        self.thumb_non_tip_contact_force_thresh = float(
            self.cfg["env"].get("thumbNonTipContactForceThresh", 1.0)
        )
        self.thumb_non_tip_contact_link_names = list(
            self.cfg["env"].get(
                "thumbNonTipContactLinks",
                [
                    "right_thumb_CMC_VL",
                    "right_thumb_MC",
                    "right_thumb_MCP_VL",
                    "right_thumb_PP",
                    "right_thumb_DP",
                ],
            )
        )
        self.tip_force_penalty_coef = float(self.cfg["env"].get("tipForcePenaltyCoef", 0.3))
        self.tip_force_penalty_thresh = float(self.cfg["env"].get("tipForcePenaltyThresh", 5.0))
        self.tip_force_penalty_exp_alpha = float(self.cfg["env"].get("tipForcePenaltyExpAlpha", 0.14))
        self.axis_dev_penalty_coef = float(self.cfg["env"].get("axisDevPenaltyCoef", 0.0))
        self.latency = self.cfg['env'].get("latency", 0.25)

        self.torque_control = self.cfg["env"].get("torqueControl", True)
        self.robot_asset_files_dict = {
            "thick": "urdf/right_sharpa_wave/right_sharpa_wave.urdf" #
        }

        self.asset_files_dict = {
            "set_obj12_cylinder_corner_y_axis": "urdf/objects/set_obj12_cylinder_corner_y_axis.urdf",
            "set_obj14_irregular_block_cross": "urdf/objects/set_obj14_irregular_block_cross.urdf",
            "set_obj16_cylinder_axis": "urdf/objects/set_obj16_cylinder_axis.urdf",
            "set_obj1_regular_block": "urdf/objects/set_obj1_regular_block.urdf",
            "set_obj2_block": "urdf/objects/set_obj2_block.urdf",
            "set_obj6_block_corner": "urdf/objects/set_obj6_block_corner.urdf",
            "bottle": "urdf/objects/contactdb/water_bottle/water_bottle.urdf",
            "bulb": "urdf/objects/ycb/bulb/bulb.urdf",
            "duck":"urdf/objects/ycb/duck/duck.urdf",
            "strawberry": "urdf/objects/ycb/strawberry/strawberry.urdf",
            "lego": "urdf/objects/ycb/lego/lego.urdf",
            "apple": "urdf/objects/contactdb/apple/apple.urdf",
            "flashlight_y_axis": "urdf/objects/contactdb/flashlight/flashlight_y_axis.urdf",
            "piggy_bank": "urdf/objects/contactdb/piggy_bank/piggy_bank.urdf",
            "toothpaste": "urdf/objects/contactdb/toothpaste/toothpaste.urdf",
            "coca_can": "urdf/objects/daily/coca_can/coca_can.urdf",
            "coca_can_y_axis": "urdf/objects/daily/coca_can/coca_can_y_axis.urdf",
            "dextool_short_screwdriver": "urdf/objects/dextoolbench/screwdriver/short_screwdriver/short_screwdriver.urdf",
            "toy_trash_can": "urdf/objects/trashcan/trashcan.urdf",
        }

        self.object_sets = {
            "set_z":
                ['set_obj6_block_corner','duck','apple','piggy_bank','strawberry',
                'set_obj14_irregular_block_cross','coca_can','bulb','set_obj16_cylinder_axis'],

            "set_x4":['set_obj1_regular_block','set_obj6_block_corner','set_obj2_block','lego'],

            "set_y":
                ['toy_trash_can','set_obj6_block_corner','set_obj1_regular_block','set_obj12_cylinder_corner_y_axis','toothpaste',
                'coca_can_y_axis','dextool_short_screwdriver','flashlight_y_axis','bottle'],
        }
        self.object_set_id = self.cfg["env"].get("objSet", "0")
        if str(self.object_set_id) not in self.object_sets:
            raise ValueError(
                f"Unknown objSet={self.object_set_id!r}. "
                f"Available sets: {', '.join(sorted(self.object_sets))}"
            )
        self.used_training_objects = self.object_sets[str(self.object_set_id)]

        self.num_training_objects = len(self.used_training_objects)
        cam_policy_cfg = self.cfg["env"].get("cameraPolicy", {})
        if not isinstance(cam_policy_cfg, dict):
            cam_policy_cfg = {}
        wm_cfg = cam_policy_cfg.get("worldModel", {})
        if not isinstance(wm_cfg, dict):
            wm_cfg = {}
        self.wm_obj_bps_num_points = max(1, int(wm_cfg.get("wm_obj_bps_num_points", 512)))
        self.wm_obj_bps_seed = int(wm_cfg.get("wm_obj_bps_seed", 2026))
        self.wm_obj_bps_radius_m = float(wm_cfg.get("wm_obj_bps_radius_m", 0.08))
        self.wm_obj_bps_target_scale = float(wm_cfg.get("wm_obj_bps_target_scale", 100.0))
        self.wm_obj_bps_target_dim = self.wm_obj_bps_num_points * 3
        self._wm_bps_basis_points = (
            _sample_unit_ball_points(self.wm_obj_bps_num_points, self.wm_obj_bps_seed)
            * self.wm_obj_bps_radius_m
        )
        self._object_bps_by_name = {}
        self.object_bps_vector_table = None
        self.object_bps_vector_per_env = None

        # Allow numEnvs='auto' to default to one env per object in the (possibly filtered) set.
        num_envs_cfg = self.cfg["env"].get("numEnvs", None)
        if isinstance(num_envs_cfg, str) and num_envs_cfg.strip().lower() == "auto":
            self.cfg["env"]["numEnvs"] = self.num_training_objects
            print(
                f"[rotation] numEnvs=auto -> {self.num_training_objects} envs "
                f"(objSet='{self.object_set_id}')."
            )

        # objInitPosShift: world-frame xyz offset added on top of arm_hand_start_pose.
        cfg_obj_init_pos_shift = self.cfg["env"]["objInitPosShift"]
        assert len(cfg_obj_init_pos_shift) == 3, \
            f"objInitPosShift must be a length-3 list [x, y, z], got {cfg_obj_init_pos_shift}"
        self.obj_init_pos_shift = tuple(float(v) for v in cfg_obj_init_pos_shift)

        self.hand_config = load_hand_config(self.cfg["env"].get("handConfigFile", None))
        self.hand_init_type = self.cfg["env"].get("handInit", "default")
        self.hand_qpos_init_override = get_hand_init_pose(self.hand_config, self.hand_init_type)

        print("Obs type: partial_stack (fixed)")

        self.palm_name = "palm"
        # self.contact_sensor_names = ["link_1.0_fsr", "link_2.0_fsr", "link_3.0_tip_fsr", "link_5.0_fsr",
        #                              "link_6.0_fsr", "link_7.0_tip_fsr", "link_9.0_fsr", "link_10.0_fsr",
        #                              "link_11.0_tip_fsr", "link_14.0_fsr", "link_15.0_fsr", "link_15.0_tip_fsr",
        #                              "link_0.0_fsr", "link_4.0_fsr", "link_8.0_fsr", "link_13.0_fsr"] #16
        self.contact_sensor_names = ["right_thumb_elastomer","right_index_elastomer","right_middle_elastomer","right_ring_elastomer","right_pinky_elastomer"] #5

        # self.tip_sensor_names = ["link_3.0_tip_fsr",  "link_7.0_tip_fsr",
        #                         "link_11.0_tip_fsr", "link_15.0_tip_fsr"]#4
        # self.tip_sensor_names = ["thumb_fsr","index_fsr","middle_fsr","ring_fsr","pinky_fsr"]#5  for diy
        self.tip_sensor_names = ["right_thumb_fingertip","right_index_fingertip","right_middle_fingertip","right_ring_fingertip","right_pinky_fingertip"] #5  
        
        self.arm_sensor_names = ["arm_link1_r", "arm_link2_r", "arm_link3_r", "arm_link4_r", "arm_link5_r", "arm_link6_r","arm_link7_r"]#7
        
        # self.arm_sensor_names = ["link1", "link2", "link3", "link4", "link5", "link6"]#6

        self.n_stack = self.cfg['env'].get('obs_stack', 4)#4
        self.arm_dof_num = self.cfg["env"].get("armDofNum", 0)
        self.include_target_in_obs = bool(self.cfg["env"].get("includeTargetInObs", True))
        self.obs_target_dim = self.cfg["env"].get("numActions", 22) if self.include_target_in_obs else 0
        self.n_obs_dim = (
            self.cfg["env"].get("numActions", 22) + self.obs_target_dim + self.finger_tactile_obs_dim + 24
        )

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0

        if self.asymmetric_obs: #True
            num_states = 212

        self.cfg["env"]["numObservations"] = self.n_obs_dim * self.n_stack
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = self.cfg["env"].get("numActions", 22)
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        # Per-axis position-noise half-widths, broadcast against rand_floats[:, 0:3] in reset_object_pose.
        self.reset_position_noise = torch.tensor(
            self._reset_position_noise_cfg, device=self.device, dtype=torch.float
        )
        self.object_class_indices_tensor = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.last_obs_buf = torch.zeros((self.num_envs, self.n_obs_dim), device=self.device, dtype=torch.float)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        self.control_dt = self.dt * float(control_freq_inv)
        self.hand_pose_non_thumb_scale = float(self.cfg["env"].get("handPoseNonThumbScale", 1.0))
        self.hand_pose_joint_weight = torch.ones(
            (self.num_arm_hand_dofs,), dtype=torch.float, device=self.device
        )
        for dof_idx, joint_name in enumerate(self.arm_hand_dof_names):
            if dof_idx < self.arm_dof_num:
                continue
            if not joint_name.startswith("right_thumb_"):
                self.hand_pose_joint_weight[dof_idx] = self.hand_pose_non_thumb_scale
        # spin_trace_output_dir is created lazily by flush_spin_trace_episode
        # right before the first csv/png write so we don't leave an empty
        # fallback dir when an output_path resolver routes every artifact to
        # a per-episode subfolder elsewhere.
        if self.reset_time > 0.0: #-1, not used
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))

        if not self.headless:
            # if self.viewer != None:
            #     cam_pos = gymapi.Vec3(5.4, 4.05, 0.57)
            #     cam_target = gymapi.Vec3(4.1, 5.35, 0.20)
            #     self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
            if self.viewer is not None:
                cam_pos = gymapi.Vec3(2.0, 0.0, 1.3)
                cam_target = gymapi.Vec3(-1.0, 0.0, 1.0)
                middle_env = self.envs[self.num_envs // 2 + int(math.sqrt(self.num_envs)) // 2]
                self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

            if self.viewer:
                self.debug_contacts = np.zeros((16, 49), dtype=np.float32)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        contact_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.asymmetric_obs:
             dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
             self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_arm_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Contact.
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.spin_axis = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.disable_sets = {
            'A': [0, 3, 6, 12, 13, 14, 9],
            'B': [1, 2, 4, 5, 7, 8, 10, 11]
        } # disable part of fingers.
        self.disable_mode = self.cfg["env"].get("disableSet", '0')
        self.use_disable = False

        if self.disable_mode in self.disable_sets:
            self.use_disable = True
            self.disable_sensor_idxes = torch.tensor(self.disable_sets[self.disable_mode],
                                                     dtype=torch.long, device=self.device)

        if self.rotation_axis == "x":
            self.all_spin_choices = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

        elif self.rotation_axis == "y":
            self.all_spin_choices = torch.tensor([[0.0, -1.0, 0.0]], device=self.device)

        elif self.rotation_axis == "z":
            self.all_spin_choices = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        elif self.rotation_axis == "all":
            self.all_spin_choices = torch.tensor(
                [
                    [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                ],
                device=self.device,
            )

        else:
            assert False, "wrong spin axis"

        # Contact.
        self.contact_tensor = gymtorch.wrap_tensor(contact_tensor).view(self.num_envs, -1) #16 111

        print("Contact Tensor Dimension", self.contact_tensor.shape)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.object_init_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # Prior object reference position from cfg (objInitPosShift + arm base pose).
        # Used by position-deviation penalty; reset randomization should not move this reference.
        self.object_prior_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.object_init_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.relative_scale_tensor = torch.full((self.num_envs, 1), self.relative_scale, device=self.device)

        self.p_gain_defaults = torch.tensor(self.hand_p_gain_list, device=self.device, dtype=torch.float)
        self.d_gain_defaults = torch.tensor(self.hand_d_gain_list, device=self.device, dtype=torch.float)
        self.p_gain = self.p_gain_defaults.unsqueeze(0).repeat(self.num_envs, 1)
        self.d_gain = self.d_gain_defaults.unsqueeze(0).repeat(self.num_envs, 1)

        self.reset_goal_buf = self.reset_buf.clone()
        self.init_stack_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.no_spin_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.gravity_rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.gravity_body_masses = torch.zeros((self.num_envs, self.num_bodies), dtype=torch.float, device=self.device)
        self.gravity_env_vector = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.gravity_tilt_theta = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.gravity_tilt_phi = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
        self.gravity_curriculum_outcomes = torch.zeros(
            (self.gravity_curriculum_window_episodes,), dtype=torch.float, device=self.device
        )
        self.gravity_curriculum_outcome_write_idx = 0
        self.gravity_curriculum_outcome_count = 0
        self.gravity_hand_body_start = 0
        self.gravity_object_body_start = 0
        self.gravity_hand_body_count = 0
        self.gravity_object_body_count = 0
        self.base_gravity = torch.tensor(
            [self.sim_params.gravity.x, self.sim_params.gravity.y, self.sim_params.gravity.z],
            dtype=torch.float,
            device=self.device,
        )
        self.base_gravity_mag = torch.norm(self.base_gravity)
        if self.base_gravity_mag < 1e-6:
            self.base_gravity = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float, device=self.device)
            self.base_gravity_mag = torch.norm(self.base_gravity)
        self.base_gravity_dir = self.base_gravity / torch.clamp(self.base_gravity_mag, min=1e-6)
        self.gravity_basis_u, self.gravity_basis_v = self._build_gravity_cone_basis(self.base_gravity_dir)
        self.gravity_env_vector[:] = self.base_gravity.unsqueeze(0)
        self._refresh_gravity_mass_cache()
        self.last_contacts = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)#16
        self.contact_thresh = torch.zeros((self.num_envs, 5), dtype=torch.float, device=self.device)#16

        self.post_init()

    def get_internal_state(self):
        return self.root_state_tensor[self.object_indices, 3:7]

    def _build_gravity_cone_basis(self, base_dir):
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device)
        if torch.abs(torch.dot(base_dir, ref)) > 0.95:
            ref = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float, device=self.device)
        basis_u = torch.cross(base_dir, ref, dim=0)
        basis_u = basis_u / torch.clamp(torch.norm(basis_u), min=1e-6)
        basis_v = torch.cross(base_dir, basis_u, dim=0)
        basis_v = basis_v / torch.clamp(torch.norm(basis_v), min=1e-6)
        return basis_u, basis_v

    def _refresh_gravity_mass_cache(self):
        for env_idx, env_ptr in enumerate(self.envs):
            hand_handle = self.arm_hands[env_idx]
            object_handle = self.objects[env_idx]
            hand_props = self.gym.get_actor_rigid_body_properties(env_ptr, hand_handle)
            object_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            if env_idx == 0:
                self.gravity_hand_body_count = len(hand_props)
                self.gravity_object_body_count = len(object_props)
                self.gravity_hand_body_start = max(
                    0, self.num_bodies - self.gravity_hand_body_count - self.gravity_object_body_count
                )
                self.gravity_object_body_start = self.gravity_hand_body_start + self.gravity_hand_body_count
            hand_masses = torch.tensor([p.mass for p in hand_props], dtype=torch.float, device=self.device)
            object_masses = torch.tensor([p.mass for p in object_props], dtype=torch.float, device=self.device)
            hand_end = min(self.gravity_hand_body_start + hand_masses.shape[0], self.num_bodies)
            object_end = min(self.gravity_object_body_start + object_masses.shape[0], self.num_bodies)
            self.gravity_body_masses[env_idx, self.gravity_hand_body_start:hand_end] = hand_masses[: hand_end - self.gravity_hand_body_start]
            self.gravity_body_masses[env_idx, self.gravity_object_body_start:object_end] = object_masses[: object_end - self.gravity_object_body_start]

    def _update_gravity_curriculum(self, full_run_outcomes):
        if not self.gravity_curriculum_enabled or full_run_outcomes.numel() == 0:
            return
        values = full_run_outcomes.float().view(-1)
        for value in values:
            self.gravity_curriculum_outcomes[self.gravity_curriculum_outcome_write_idx] = value
            self.gravity_curriculum_outcome_write_idx = (
                self.gravity_curriculum_outcome_write_idx + 1
            ) % self.gravity_curriculum_window_episodes
            self.gravity_curriculum_outcome_count = min(
                self.gravity_curriculum_outcome_count + 1, self.gravity_curriculum_window_episodes
            )
        valid_count = max(1, self.gravity_curriculum_outcome_count)
        self.gravity_curriculum_full_run_ratio = float(
            self.gravity_curriculum_outcomes[:valid_count].mean().item()
        )
        if self.gravity_curriculum_outcome_count < self.gravity_curriculum_window_episodes:
            return
        if (
            self.gravity_curriculum_full_run_ratio >= self.gravity_curriculum_ratio_threshold
            and (
                self.gravity_curriculum_current_max_angle_deg < self.gravity_curriculum_max_angle_deg
                or self.action_conservative_scale < self.action_conservative_scale_max
            )
        ):
            self.gravity_curriculum_current_max_angle_deg = min(
                self.gravity_curriculum_current_max_angle_deg + self.gravity_curriculum_angle_step_deg,
                self.gravity_curriculum_max_angle_deg,
            )
            self.action_conservative_scale = min(
                self.action_conservative_scale + self.action_conservative_scale_step,
                self.action_conservative_scale_max,
            )
            self.gravity_curriculum_upgrade_count += 1
            self.gravity_curriculum_outcome_count = 0
            self.gravity_curriculum_outcome_write_idx = 0
            self.gravity_curriculum_outcomes.zero_()
            self.gravity_curriculum_full_run_ratio = 0.0

    def _sample_gravity_for_resets(self, env_ids):
        if len(env_ids) == 0:
            return
        if not self.gravity_curriculum_enabled:
            self.gravity_env_vector[env_ids] = self.base_gravity.unsqueeze(0).repeat(len(env_ids), 1)
            self.gravity_tilt_theta[env_ids] = 0.0
            self.gravity_tilt_phi[env_ids] = 0.0
            return
        max_angle_rad = math.radians(self.gravity_curriculum_current_max_angle_deg)
        theta = torch.rand((len(env_ids),), device=self.device) * max_angle_rad
        phi = torch.rand((len(env_ids),), device=self.device) * (2.0 * math.pi)
        cone_dir = (
            torch.cos(theta).unsqueeze(-1) * self.base_gravity_dir.unsqueeze(0)
            + torch.sin(theta).unsqueeze(-1)
            * (
                torch.cos(phi).unsqueeze(-1) * self.gravity_basis_u.unsqueeze(0)
                + torch.sin(phi).unsqueeze(-1) * self.gravity_basis_v.unsqueeze(0)
            )
        )
        cone_dir = cone_dir / torch.clamp(torch.norm(cone_dir, dim=-1, keepdim=True), min=1e-6)
        self.gravity_env_vector[env_ids] = cone_dir * self.base_gravity_mag
        self.gravity_tilt_theta[env_ids] = theta
        self.gravity_tilt_phi[env_ids] = phi

    def _apply_gravity_equivalent_force(self):
        delta_g = self.gravity_env_vector - self.base_gravity.unsqueeze(0)
        torch.mul(
            self.gravity_body_masses.unsqueeze(-1),
            delta_g.unsqueeze(1),
            out=self.gravity_rb_forces,
        )
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.gravity_rb_forces),
            None,
            gymapi.ENV_SPACE,
        )

    def get_internal_info(self, key):
        if key == 'target':
            # PD target before torque computation.
            return self.cur_targets
        elif key == 'qpos':
            # Full per-env DOF positions (radians for revolute joints). Used by HDF5 / sim2real tooling.
            return self.arm_hand_dof_pos
        elif key == 'joint_names':
            return getattr(self, 'arm_hand_dof_names', [])
        elif key == 'contact':
            return  self.sensed_contacts 
        elif key == 'obj':
            return torch.tensor(self.object_class_indices, dtype=torch.long, device=self.device).reshape(self.num_envs, -1)
        elif key == 'qinit':
            return self.object_init_quat

        return None

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.create_object_asset_dict(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))

        if self.use_default_ground_plane:
            self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    @staticmethod
    def _rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
        ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return (rz @ ry @ rx).astype(np.float32)

    def _load_object_mesh_from_urdf(self, asset_root: str, urdf_rel_path: str) -> trimesh.Trimesh:
        urdf_path = os.path.abspath(os.path.join(asset_root, urdf_rel_path))
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        # Some third-party URDFs have a leading blank line before the XML declaration.
        # ET.parse() rejects these with "XML declaration not at start of entity".
        with open(urdf_path, "rb") as f:
            xml_blob = f.read().lstrip()
        root = ET.fromstring(xml_blob)
        mesh_parts = []
        urdf_dir = os.path.dirname(urdf_path)
        for visual in root.findall(".//visual"):
            origin = visual.find("origin")
            xyz = np.zeros((3,), dtype=np.float32)
            rot_m = np.eye(3, dtype=np.float32)
            if origin is not None:
                xyz_text = origin.attrib.get("xyz", "").strip()
                if xyz_text:
                    vals = [float(v) for v in xyz_text.split()]
                    if len(vals) == 3:
                        xyz = np.asarray(vals, dtype=np.float32)
                rpy_text = origin.attrib.get("rpy", "").strip()
                if rpy_text:
                    vals = [float(v) for v in rpy_text.split()]
                    if len(vals) == 3:
                        rot_m = self._rpy_to_matrix(vals[0], vals[1], vals[2])
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            part_mesh = None
            mesh_tag = geometry.find("mesh")
            if mesh_tag is not None:
                filename = mesh_tag.attrib.get("filename", "").strip()
                if filename.startswith("package://"):
                    filename = filename[len("package://") :]
                mesh_path = filename if os.path.isabs(filename) else os.path.abspath(os.path.join(urdf_dir, filename))
                if not os.path.isfile(mesh_path):
                    alt_path = os.path.abspath(os.path.join(asset_root, filename))
                    if os.path.isfile(alt_path):
                        mesh_path = alt_path
                mesh = trimesh.load(mesh_path, force="mesh")
                if isinstance(mesh, trimesh.Scene):
                    dumped = mesh.dump(concatenate=True)
                    mesh = dumped if isinstance(dumped, trimesh.Trimesh) else trimesh.util.concatenate(tuple(dumped.values()))
                scale = np.ones((3,), dtype=np.float32)
                scale_text = mesh_tag.attrib.get("scale", "").strip()
                if scale_text:
                    scale_vals = [float(v) for v in scale_text.split()]
                    if len(scale_vals) == 3:
                        scale = np.asarray(scale_vals, dtype=np.float32)
                part_mesh = mesh.copy()
                part_mesh.vertices = part_mesh.vertices * scale[None, :]
            if part_mesh is None:
                box_tag = geometry.find("box")
                cyl_tag = geometry.find("cylinder")
                sph_tag = geometry.find("sphere")
                if box_tag is not None:
                    size_text = box_tag.attrib.get("size", "").strip()
                    vals = [float(v) for v in size_text.split()] if size_text else [1.0, 1.0, 1.0]
                    if len(vals) == 3:
                        part_mesh = trimesh.creation.box(extents=np.asarray(vals, dtype=np.float32))
                elif cyl_tag is not None:
                    radius = float(cyl_tag.attrib.get("radius", 0.05))
                    height = float(cyl_tag.attrib.get("length", 0.1))
                    part_mesh = trimesh.creation.cylinder(radius=radius, height=height)
                elif sph_tag is not None:
                    radius = float(sph_tag.attrib.get("radius", 0.05))
                    part_mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
            if part_mesh is None:
                continue
            transform = np.eye(4, dtype=np.float32)
            transform[:3, :3] = rot_m
            transform[:3, 3] = xyz
            part_mesh = part_mesh.copy()
            part_mesh.apply_transform(transform)
            mesh_parts.append(part_mesh)
        if len(mesh_parts) == 0:
            raise RuntimeError(f"No visual geometry mesh found from URDF: {urdf_path}")
        if len(mesh_parts) == 1:
            return mesh_parts[0]
        return trimesh.util.concatenate(mesh_parts)

    def _compute_object_bps_vector(self, mesh: trimesh.Trimesh) -> np.ndarray:
        mesh_abs = mesh.copy()
        verts = np.asarray(mesh_abs.vertices, dtype=np.float32).reshape(-1, 3)
        if verts.shape[0] == 0:
            raise RuntimeError("Mesh has no vertices.")
        center = verts.mean(axis=0, keepdims=True)
        verts = verts - center
        mesh_abs.vertices = verts
        basis = self._wm_bps_basis_points.astype(np.float32)  # (K, 3)
        try:
            nearest_pts, _, _ = trimesh.proximity.closest_point(mesh_abs, basis)
            nearest_pts = np.asarray(nearest_pts, dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                "Failed to compute BPS nearest points via trimesh.proximity.closest_point. "
                "Please install required proximity dependencies (e.g. rtree) "
                "or fix mesh assets."
            ) from exc
        disp = (nearest_pts - basis) * self.wm_obj_bps_target_scale  # meters -> configured target unit
        return disp.reshape(-1).astype(np.float32)

    def create_object_asset_dict(self, asset_root):
        self.object_asset_dict = {}
        print("ENTER ASSET CREATING!")
        for used_objects in self.used_training_objects:
            object_asset_file = self.asset_files_dict[used_objects]
            object_asset_options = gymapi.AssetOptions()
            # new for object
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 30000 #60000

            object_asset_options.vhacd_params.max_convex_hulls = 32 #64
            object_asset_options.vhacd_params.convex_hull_approximation = False

            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            object_asset_options.disable_gravity = True

            goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

            self.object_asset_dict[used_objects] = {'obj': object_asset, 'goal': goal_asset}
            try:
                object_mesh = self._load_object_mesh_from_urdf(
                    asset_root=asset_root,
                    urdf_rel_path=object_asset_file,
                )
                self._object_bps_by_name[used_objects] = self._compute_object_bps_vector(object_mesh)
            except Exception as exc:
                raise RuntimeError(
                    f"[wm-bps] failed to build BPS target for object '{used_objects}' "
                    f"(asset='{object_asset_file}')"
                ) from exc

            self.object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)

    def _resolve_hand_friction_shape_groups(self, env_ptr, hand_actor):
        hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
        shape_count = len(hand_props)
        elastomer_shape_ids = set()
        elastomer_body_indices = {}

        def add_shape_ids(shape_ids):
            if shape_ids is None:
                return
            if hasattr(shape_ids, "start") and hasattr(shape_ids, "count"):
                shape_ids = range(int(shape_ids.start), int(shape_ids.start) + int(shape_ids.count))
            elif isinstance(shape_ids, int):
                shape_ids = [shape_ids]
            for shape_id in shape_ids:
                shape_id = int(shape_id)
                if 0 <= shape_id < shape_count:
                    elastomer_shape_ids.add(shape_id)

        body_shape_indices = self.gym.get_actor_rigid_body_shape_indices(env_ptr, hand_actor)
        for link_name in self.hand_elastomer_links:
            body_index = self.gym.find_actor_rigid_body_index(
                env_ptr, hand_actor, link_name, gymapi.DOMAIN_ACTOR
            )
            elastomer_body_indices[link_name] = int(body_index)
            if 0 <= body_index < len(body_shape_indices):
                add_shape_ids(body_shape_indices[body_index])

        all_shape_ids = set(range(shape_count))
        other_shape_ids = all_shape_ids.difference(elastomer_shape_ids)
        if len(elastomer_shape_ids) == 0:
            raise RuntimeError(
                "[friction] Failed to resolve elastomer collision shapes. "
                "High hand_elastomer_base friction would not be applied. "
                f"elastomer_links={sorted(self.hand_elastomer_links)}, "
                f"body_indices={elastomer_body_indices}, "
                f"shape_count={shape_count}, body_count={self.num_arm_hand_bodies}."
            )
        if not self._printed_hand_friction_shape_groups:
            print(
                f"[friction] resolved elastomer collision shapes: "
                f"links={sorted(self.hand_elastomer_links)}, "
                f"shape_ids={sorted(elastomer_shape_ids)}, "
                f"other_shape_count={len(other_shape_ids)}"
            )
            self._printed_hand_friction_shape_groups = True
        return elastomer_shape_ids, other_shape_ids

    def _randomize_hand_object_friction(self, env_ptr, hand_actor, object_actor):
        friction_scale = np.random.uniform(self.friction_scale_lower, self.friction_scale_upper)
        hand_elastomer_friction = self.hand_elastomer_base_friction * friction_scale
        hand_other_friction = self.hand_other_base_friction * friction_scale
        object_friction = self.object_base_friction * friction_scale

        hand_props = self.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
        if len(self.hand_elastomer_shape_ids) + len(self.hand_other_shape_ids) != len(hand_props):
            self.hand_elastomer_shape_ids, self.hand_other_shape_ids = self._resolve_hand_friction_shape_groups(
                env_ptr, hand_actor
            )
        for shape_id, shape_prop in enumerate(hand_props):
            if shape_id in self.hand_elastomer_shape_ids:
                shape_prop.friction = hand_elastomer_friction
            else:
                shape_prop.friction = hand_other_friction
        self.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

        object_handles = object_actor if isinstance(object_actor, list) else [object_actor]
        for object_handle in object_handles:
            object_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            for shape_prop in object_props:
                shape_prop.friction = object_friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_props)

        return friction_scale, hand_elastomer_friction, hand_other_friction, object_friction

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        arm_hand_asset_file = self.robot_asset_files_dict[self.cfg["env"]["sensor"]]

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)

        # load arm and hand.
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        
        #new for realman luban
        # asset_options.vhacd_enabled = False
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 60000 #60000

        asset_options.vhacd_params.max_convex_hulls = 64
        asset_options.vhacd_params.convex_hull_approximation = False
        
        
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)
        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        print("Num dofs: ", self.num_arm_hand_dofs)
        self.num_arm_hand_actuators = self.num_arm_hand_dofs 
        self.arm_hand_dof_names = [self.gym.get_asset_dof_name(arm_hand_asset, i) for i in range(self.num_arm_hand_dofs)]
        print(f"arm_hand dof names ({len(self.arm_hand_dof_names)}): {self.arm_hand_dof_names}")
        self.hand_p_gain_list, self.hand_d_gain_list = build_pd_gain_lists(
            self.hand_config, self.arm_hand_dof_names, self.num_actions
        )
        self._build_spin_trace_finger_layout()
        arm_hand_body_names = [self.gym.get_asset_rigid_body_name(arm_hand_asset, i) for i in range(self.num_arm_hand_bodies)]
        print(f'arm_hand rigid body num: {self.num_arm_hand_bodies}, arm_hand_body_names: {arm_hand_body_names}')
        self.arm_hand_body_names = arm_hand_body_names
        
        
        # Set up each DOF.
        self.actuated_dof_indices = [i for i in range(self.num_arm_hand_dofs)]

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        # This part is very important (damping)
        for i in range(self.num_arm_hand_dofs):
        
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
        robot_dof_props = apply_dof_runtime_params(
            robot_dof_props, self.arm_hand_dof_names, self.arm_dof_num, self.hand_config
        )
        for i in range(self.num_arm_hand_dofs):
            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

        if self.train_limit_global_scale != 1.0:
            for i in range(self.arm_dof_num, self.num_arm_hand_dofs):
                robot_lower_qpos[i] = float(robot_lower_qpos[i]) * self.train_limit_global_scale
                robot_upper_qpos[i] = float(robot_upper_qpos[i]) * self.train_limit_global_scale

        for joint_name, joint_cfg in self.train_limit_unilateral_override_deg.items():
            if not isinstance(joint_cfg, dict):
                print(f"[WARN] trainLimit.unilateral_override_deg[{joint_name}] must be dict, skip.")
                continue
            if joint_name not in self.arm_hand_dof_names:
                print(f"[WARN] trainLimit joint not found: {joint_name}, skip.")
                continue
            idx = self.arm_hand_dof_names.index(joint_name)
            if idx < self.arm_dof_num:
                print(f"[WARN] trainLimit joint belongs to arm dof ({joint_name}), skip.")
                continue

            joint_lower = float(robot_lower_qpos[idx])
            joint_upper = float(robot_upper_qpos[idx])
            if "lower" in joint_cfg:
                joint_lower = math.radians(float(joint_cfg["lower"]))
            if "upper" in joint_cfg:
                joint_upper = math.radians(float(joint_cfg["upper"]))
            if joint_lower > joint_upper:
                raise ValueError(
                    f"Invalid trainLimit override for {joint_name}: lower({joint_lower}) > upper({joint_upper})"
                )
            robot_lower_qpos[idx] = joint_lower
            robot_upper_qpos[idx] = joint_upper

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_effort_limits = to_torch(robot_dof_props["effort"], device=self.device)

        print("DOF_LOWER_LIMITS", robot_lower_qpos)
        print("DOF_UPPER_LIMITS", robot_upper_qpos)
        if self.train_limit_global_scale != 1.0 or len(self.train_limit_unilateral_override_deg) > 0:
            print(f"[INFO][trainLimit] global_scale={self.train_limit_global_scale}")
            for joint_name in self.train_limit_unilateral_override_deg.keys():
                if joint_name not in self.arm_hand_dof_names:
                    continue
                idx = self.arm_hand_dof_names.index(joint_name)
                lo_deg = math.degrees(float(robot_lower_qpos[idx]))
                hi_deg = math.degrees(float(robot_upper_qpos[idx]))
                print(f"[INFO][trainLimit] {joint_name}: [{lo_deg:.2f}, {hi_deg:.2f}] deg")

        # Set up default arm position.
        # self.default_arm_pos = [0.00, 1.183, -1.541, 3.1416, 2.742, -1.569]
        # self.default_arm_pos = [0.0, 0, 0, 0, 0.0, 0.0,0.0]  
        # angles_degree=[-18.76, 75.2, -52.3, 79.0, 72.2, -32.2, -7.7]
        # self.default_arm_pos = degrees_to_radians(angles_degree)
        # self.default_arm_pos = [-0.3273, 1.31, -0.91, 1.37, 1.26, -0.56, -0.13]
        
        self.default_arm_pos = [-0.3273,  1.3117, -0.9124,  1.3785,  1.2601, -0.5624, -0.1342]
        
        #set the mean value of upper and lower limits as the default position:

        for i in range(self.arm_dof_num):
            self.default_arm_pos[i] = (self.arm_hand_dof_lower_limits[i] + self.arm_hand_dof_upper_limits[i]) / 2.0
            
        print("Default Arm Pos:", self.default_arm_pos)
        # exit(-1)
        
        # self.default_arm_pos = [0.2922342121601105, 1.4073221683502197, -0.8748521208763123, 1.7860136032104492, -0.8145279288291931, -2.2353851795196533, -0.8819681406021118]
          

        for i in range(self.num_arm_hand_dofs):
            if i < self.arm_dof_num:
                self.arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            else:
                self.arm_hand_dof_default_pos.append(0.0)
            self.arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # Put objects in the scene.
        arm_hand_start_pose = gymapi.Transform()
        # arm_hand_start_pose.p = gymapi.Vec3(0, 0.0, 0.0)
        # arm_hand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        arm_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 1.0) #gymapi.Vec3(1.0, -0.025, 1.33)
        arm_hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(-90.0))#xyzw


        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        pose_dx, pose_dy, pose_dz = self.obj_init_pos_shift
        object_start_pose.p.x = arm_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = arm_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = arm_hand_start_pose.p.z + pose_dz
        object_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.object_init_yaw_rad = 0.0
        if abs(self.object_init_yaw_rad) > 1e-8:
            z_yaw_quat = gymapi.Quat.from_axis_angle(gymapi.Vec3(0.0, 0.0, 1.0), self.object_init_yaw_rad)
            object_start_pose.r = z_yaw_quat * object_start_pose.r

        self.goal_displacement = gymapi.Vec3(0.2, 0.2, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 300
        max_agg_shapes = self.num_arm_hand_shapes + 300

        self.objects = []   # object handles
        self.arm_hands = [] # arm-hand handles
        self.envs = []      # environment pointers

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.object_indices = []

        self.object_class_indices = []
        white_ground_asset = None
        white_ground_pose = None
        if self.white_ground_enabled:
            ground_options = gymapi.AssetOptions()
            ground_options.fix_base_link = True
            ground_options.disable_gravity = True
            white_ground_asset = self.gym.create_box(
                self.sim,
                self.white_ground_size,
                self.white_ground_size,
                self.white_ground_thickness,
                ground_options,
            )
            white_ground_pose = gymapi.Transform()
            white_ground_pose.p = gymapi.Vec3(
                0.0,
                0.0,
                self.white_ground_z - 0.5 * self.white_ground_thickness,
            )

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.object_rb_handles = list(range(arm_hand_rb_count, arm_hand_rb_count + self.object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if white_ground_asset is not None:
                white_ground_handle = self.gym.create_actor(
                    env_ptr, white_ground_asset, white_ground_pose, "white_ground", i, 1, 0
                )
                self.gym.set_rigid_body_color(
                    env_ptr,
                    white_ground_handle,
                    0,
                    gymapi.MESH_VISUAL_AND_COLLISION,
                    gymapi.Vec3(1.0, 1.0, 1.0),
                )

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([arm_hand_start_pose.p.x,
                                           arm_hand_start_pose.p.y,
                                           arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x,
                                           arm_hand_start_pose.r.y,
                                           arm_hand_start_pose.r.z,
                                           arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, robot_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            for rb in range(arm_hand_rb_count):
                self.gym.set_rigid_body_segmentation_id(env_ptr, arm_hand_actor, rb, 2)
            self.hand_indices.append(hand_idx)

            # add object
            # Deterministic object mapping: env_id -> fixed index, and cycle when env count exceeds object count.
            obj_class_indice = i % len(self.used_training_objects)
            select_obj = self.used_training_objects[obj_class_indice]
            object_handle = self.gym.create_actor(env_ptr, self.object_asset_dict[select_obj]['obj'], object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                        object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_class_indices.append(obj_class_indice)

            friction_scale, hand_elastomer_friction, hand_other_friction, object_friction = (
                self._randomize_hand_object_friction(env_ptr, arm_hand_actor, object_handle)
            )
            if i == 0:
                print(
                    f"[friction] scale={friction_scale:.4f}, hand_elastomer={hand_elastomer_friction:.4f}, "
                    f"hand_other={hand_other_friction:.4f}, object={object_friction:.4f}, "
                    f"combine_mode=average(default)"
                )

            prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            for p in prop:
                p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
            self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)

            self.objects.append(object_handle)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)
            
        self.object_class_indices_tensor = torch.tensor(self.object_class_indices, dtype=torch.long, device=self.device)

        palm_handles = self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, self.palm_name)
        self.palm_indices = to_torch(palm_handles, dtype=torch.int64, device=self.device)
        virtual_hand_base_handle = self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, "virtual_hand_base")
        if virtual_hand_base_handle < 0:
            virtual_hand_base_handle = palm_handles
        self.virtual_hand_base_handle = int(virtual_hand_base_handle)
        self.virtual_hand_base_indices = to_torch(self.virtual_hand_base_handle, dtype=torch.int64, device=self.device)

        palm_root_contact_link_names = ["right_hand_C_MC", "right_pinky_MC"]
        palm_root_contact_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, link_name)
            for link_name in palm_root_contact_link_names
        ]
        self.palm_root_contact_valid_mask = to_torch(
            [1.0 if h >= 0 else 0.0 for h in palm_root_contact_handles],
            dtype=torch.float,
            device=self.device,
        )
        palm_root_contact_handles = [h if h >= 0 else 0 for h in palm_root_contact_handles]
        self.palm_root_contact_handle_indices = to_torch(
            palm_root_contact_handles, dtype=torch.int64, device=self.device
        )
        thumb_non_tip_contact_handles = [
            self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, link_name)
            for link_name in self.thumb_non_tip_contact_link_names
        ]
        self.thumb_non_tip_contact_valid_mask = to_torch(
            [1.0 if h >= 0 else 0.0 for h in thumb_non_tip_contact_handles],
            dtype=torch.float,
            device=self.device,
        )
        thumb_non_tip_contact_handles = [h if h >= 0 else 0 for h in thumb_non_tip_contact_handles]
        self.thumb_non_tip_contact_handle_indices = to_torch(
            thumb_non_tip_contact_handles, dtype=torch.int64, device=self.device
        )

        sensor_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                          for sensor_name in self.contact_sensor_names]
        self.sensor_valid_mask = to_torch([1.0 if h >= 0 else 0.0 for h in sensor_handles], dtype=torch.float, device=self.device)
        sensor_handles = [h if h >= 0 else 0 for h in sensor_handles]
        self.sensor_handle_indices = to_torch(sensor_handles, dtype=torch.int64, device=self.device)

        arm_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                          for sensor_name in self.arm_sensor_names]
        self.arm_handle_indices = to_torch(arm_handles, dtype=torch.int64, device=self.device)

        tip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, sensor_name)
                       for sensor_name in self.tip_sensor_names]
        tip_handles = [h if h >= 0 else 0 for h in tip_handles]
        self.fingertip_handles = to_torch(tip_handles, dtype=torch.int64, device=self.device)

        self.object_class_indices = to_torch(self.object_class_indices, dtype=torch.int64, device=self.device)
        self.object_one_hot_vector = F.one_hot(self.object_class_indices, num_classes=self.num_training_objects).float()
        bps_vectors = [self._object_bps_by_name[obj_name] for obj_name in self.used_training_objects]
        self.object_bps_vector_table = torch.tensor(
            np.stack(bps_vectors, axis=0), dtype=torch.float32, device=self.device
        )
        self.object_bps_vector_per_env = self.object_bps_vector_table[self.object_class_indices]

        # override!
        self.hand_override_info = []
        self.hand_override_missing = []
        for finger_name, qpos in self.hand_qpos_init_override.items():
            dof_handle = self.gym.find_actor_dof_handle(env_ptr, arm_hand_actor, finger_name)
            if 0 <= dof_handle < self.num_arm_hand_dofs:
                self.hand_override_info.append((dof_handle, qpos))
            else:
                self.hand_override_missing.append(finger_name)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.object_prior_pos = self.object_init_state[:, 0:3].clone()
        self.goal_states = self.object_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self._init_in_process_perturbation_state()

        self.contact_handles = [
            self.gym.find_asset_rigid_body_index(arm_hand_asset, name) for name in self.arm_hand_body_names
        ]
        print(f'contact_handles: {self.contact_handles}')
        self.contact_handles = to_torch(self.contact_handles, dtype=torch.long, device=self.device)

    def post_init(self):
        all_qpos = {}

        arm_hand_dof_default_pos = []
        arm_hand_dof_default_vel = []
        for (idx, qpos) in self.hand_override_info:
            print("Hand QPos Overriding: Idx:{} QPos: {}".format(idx, qpos))
            self.arm_hand_default_dof_pos[idx] = qpos
            all_qpos[idx] = qpos
        if len(self.hand_override_missing) > 0:
            print("Hand QPos Override Missing DOF Names:", self.hand_override_missing)
        print("Hand QPos Override Matched Count:", len(self.hand_override_info))

        # Resolve the default hand pose strictly from handInitPoses[handInit].
        missing_hand_joints = []
        for i in range(self.num_arm_hand_dofs):
            if i < self.arm_dof_num:
                arm_hand_dof_default_pos.append(self.default_arm_pos[i])
            else:
                if i in all_qpos:
                    arm_hand_dof_default_pos.append(all_qpos[i])
                else:
                    missing_hand_joints.append(self.arm_hand_dof_names[i])
                    arm_hand_dof_default_pos.append(0.0)
            arm_hand_dof_default_vel.append(0.0)
        if len(missing_hand_joints) > 0:
            raise ValueError(
                f"handInit '{self.hand_init_type}' is missing DOFs: {missing_hand_joints}"
            )

        self.arm_hand_dof_default_pos = to_torch(arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(arm_hand_dof_default_vel, device=self.device)

    def compute_reward(self, actions):
        self.control_error = torch.norm(self.cur_targets - self.arm_hand_dof_pos, dim=1)
        joint_delta = self.arm_hand_dof_pos - self.arm_hand_dof_default_pos.unsqueeze(0)
        weighted_joint_delta_sq = (joint_delta * joint_delta) * self.hand_pose_joint_weight.unsqueeze(0)
        hand_pose_delta_sq = torch.sum(weighted_joint_delta_sq, dim=-1)

        # Lets do some calculation

        prev_rot_mats = transform.quaternion_to_matrix(xyzw_to_wxyz(self.last_object_rot))
        curr_rot_mats = transform.quaternion_to_matrix(xyzw_to_wxyz(self.object_rot))

        spin_delta_axis, spin_delta_offaxis = compute_spin_deltas_from_rot_mats(
            prev_rot_mats, curr_rot_mats, self.spin_axis
        )

        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = (torch.abs(self.torques) * torch.abs(self.dof_vel_finite_diff)).sum(-1)
        conservative_scale = self.action_conservative_scale
        torque_coef_eff = self.torque_coef * conservative_scale
        work_coef_eff = self.work_coef * conservative_scale
        control_penalty_scale_eff = self.control_penalty_scale * conservative_scale
        action_penalty_scale_eff = self.action_penalty_scale * conservative_scale

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], \
        self.progress_buf[:], self.successes[:], self.consecutive_successes[:], \
        spin_reward, vel_reward, contact_reward, distance_reward, \
        torque_penalty_term, work_penalty_term, action_penalty_term, control_penalty_term, \
        hand_pose_reward_term, obj_init_pos_penalty_term, axis_dev_penalty_term, no_spin_penalty_term, reward_total, \
        self.no_spin_counter = compute_hand_reward_finger(
            torch.tensor(self.spin_coef).to(self.device),
            torch.tensor(self.vel_coef).to(self.device),
            torch.tensor(torque_coef_eff).to(self.device),
            torch.tensor(work_coef_eff).to(self.device),
            torch.tensor(self.contact_coef).to(self.device),
            torch.tensor(self.finger_coef).to(self.device),
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes,
            # self.max_episode_length, self.fingertip_pos, self.object_pos, self.object_rot, self.object_prior_pos,
            self.max_episode_length, self.fingertip_pos, self.object_pos, self.object_rot, self.object_init_pos,
            self.object_init_quat, self.object_linvel,
            self.object_angvel,
            self.goal_pos, self.goal_rot, self.finger_contacts, self.control_error,
            control_penalty_scale_eff, self.actions, action_penalty_scale_eff,
            torch.tensor(self.hand_pose_coef).to(self.device), hand_pose_delta_sq,
            torch.tensor(self.obj_init_pos_dev_coef).to(self.device),
            torch.tensor(self.distance_reward_deadzone).to(self.device),
            torch.tensor(self.distance_reward_penalty_width).to(self.device),
            torch.tensor(self.axis_dev_penalty_coef).to(self.device),
            torch.tensor(self.reward_max_spin_rate).to(self.device),
            self.no_spin_counter, torch.tensor(self.no_spin_angvel_thresh).to(self.device),
            self.no_spin_max_steps, torch.tensor(self.no_spin_reset_penalty).to(self.device),
            self.fall_dist, self.fall_penalty, self.spin_axis, spin_delta_axis, spin_delta_offaxis,
            torque_penalty, work_penalty, self.max_consecutive_successes, self.av_factor
        )

        self.extras['reward_terms/spin_reward'] = spin_reward.mean()
        self.extras['reward_terms/vel_reward'] = vel_reward.mean()
        self.extras['reward_terms/contact_reward'] = contact_reward.mean()
        self.extras['reward_terms/distance_reward'] = distance_reward.mean()
        self.extras['reward_terms/torque_penalty_term'] = torque_penalty_term.mean()
        self.extras['reward_terms/work_penalty_term'] = work_penalty_term.mean()
        self.extras['reward_terms/action_penalty_term'] = action_penalty_term.mean()
        self.extras['reward_terms/control_penalty_term'] = control_penalty_term.mean()
        self.extras['reward_terms/hand_pose_reward_term'] = hand_pose_reward_term.mean()
        self.extras['reward_terms/obj_init_pos_penalty_term'] = obj_init_pos_penalty_term.mean()
        self.extras['reward_terms/axis_dev_penalty_term'] = axis_dev_penalty_term.mean()
        self.extras['reward_terms/no_spin_penalty_term'] = no_spin_penalty_term.mean()
        self.extras['reward_terms/spin_delta_axis'] = spin_delta_axis.mean()
        self.extras['reward_terms/spin_delta_offaxis'] = spin_delta_offaxis.mean()
        self.last_spin_delta_axis = spin_delta_axis.detach()
        self.last_spin_delta_offaxis = spin_delta_offaxis.detach()
        tip_force_penalty_term = torch.zeros_like(self.rew_buf)
        tip_force_excess_mean = torch.zeros_like(self.rew_buf)
        tip_force_mean = torch.zeros_like(self.rew_buf)
        if self.tip_force_penalty_coef != 0.0:
            contacts = self.contact_tensor.view(self.num_envs, -1, 3)
            tip_contacts = contacts[:, self.sensor_handle_indices, :]
            tip_force = torch.norm(tip_contacts, dim=-1)
            tip_force = tip_force * self.sensor_valid_mask.unsqueeze(0)
            tip_force_mean = tip_force.mean(dim=-1)
            tip_force_excess = torch.relu(tip_force - self.tip_force_penalty_thresh)
            tip_force_excess_mean = tip_force_excess.mean(dim=-1)
            tip_force_penalty_term = -self.tip_force_penalty_coef * (
                torch.exp(self.tip_force_penalty_exp_alpha * tip_force_excess_mean) - 1.0
            )
            self.rew_buf += tip_force_penalty_term
            reward_total = reward_total + tip_force_penalty_term
        self.extras['reward_terms/tip_force_penalty_term'] = tip_force_penalty_term.mean()
        self.extras['reward_terms/tip_force_excess_mean'] = tip_force_excess_mean.mean()
        self.extras['reward_terms/tip_force_mean'] = tip_force_mean.mean()
        palm_root_contact_term = torch.zeros_like(self.rew_buf)
        if self.palm_root_contact_coef != 0.0:
            contacts = self.contact_tensor.view(self.num_envs, -1, 3)
            palm_root_contacts = contacts[:, self.palm_root_contact_handle_indices, :]
            palm_root_contact_force = torch.norm(palm_root_contacts, dim=-1)
            palm_root_contact_force = (
                palm_root_contact_force * self.palm_root_contact_valid_mask.unsqueeze(0)
            )
            has_palm_root_contact = torch.any(
                palm_root_contact_force > self.palm_root_contact_force_thresh, dim=1
            ).float()
            palm_root_contact_term = self.palm_root_contact_coef * has_palm_root_contact
            self.rew_buf += palm_root_contact_term
            reward_total = reward_total + palm_root_contact_term
        self.extras['reward_terms/palm_root_contact_term'] = palm_root_contact_term.mean()
        thumb_non_tip_contact_term = torch.zeros_like(self.rew_buf)
        thumb_non_tip_contact_hit = torch.zeros_like(self.rew_buf)
        thumb_non_tip_contact_force_max = torch.zeros_like(self.rew_buf)
        if self.thumb_non_tip_contact_penalty > 0.0:
            contacts = self.contact_tensor.view(self.num_envs, -1, 3)
            thumb_non_tip_contacts = contacts[:, self.thumb_non_tip_contact_handle_indices, :]
            thumb_non_tip_contact_force = torch.norm(thumb_non_tip_contacts, dim=-1)
            thumb_non_tip_contact_force = (
                thumb_non_tip_contact_force * self.thumb_non_tip_contact_valid_mask.unsqueeze(0)
            )
            thumb_non_tip_contact_force_max = torch.max(thumb_non_tip_contact_force, dim=1).values
            thumb_non_tip_contact_hit = torch.any(
                thumb_non_tip_contact_force > self.thumb_non_tip_contact_force_thresh, dim=1
            ).float()
            thumb_non_tip_contact_term = -self.thumb_non_tip_contact_penalty * thumb_non_tip_contact_hit
            self.rew_buf += thumb_non_tip_contact_term
            reward_total = reward_total + thumb_non_tip_contact_term
        self.extras['reward_terms/thumb_non_tip_contact_term'] = thumb_non_tip_contact_term.mean()
        self.extras['reward_terms/thumb_non_tip_contact_hit_rate'] = thumb_non_tip_contact_hit.mean()
        self.extras['reward_terms/thumb_non_tip_contact_force_max'] = thumb_non_tip_contact_force_max.mean()
        self.extras['reward_terms/reward_total'] = reward_total.mean()
        self.extras['consecutive_successes'] = self.consecutive_successes.mean()
        self.extras['gravity_curr/max_angle_deg'] = torch.tensor(
            self.gravity_curriculum_current_max_angle_deg, dtype=torch.float, device=self.device
        )
        self.extras['gravity_curr/full_run_ratio'] = torch.tensor(
            self.gravity_curriculum_full_run_ratio, dtype=torch.float, device=self.device
        )
        self.extras['gravity_curr/upgrade_count'] = torch.tensor(
            float(self.gravity_curriculum_upgrade_count), dtype=torch.float, device=self.device
        )
        self.extras['gravity_curr/mean_tilt_deg'] = torch.rad2deg(self.gravity_tilt_theta).mean()
        self.extras['gravity_curr/action_conservative_scale'] = torch.tensor(
            self.action_conservative_scale, dtype=torch.float, device=self.device
        )

        if self.print_success_stat:#not usefull
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]  # [num_env, 2, 7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]  # [num_env, 2, 3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]  # [num_env, 2, 4]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, ..., 0:7]
        self.goal_pos = self.goal_states[:, ..., 0:3]
        self.goal_rot = self.goal_states[:, ..., 3:7]

        # Keep index tensor aligned with runtime state tensor device (multi-GPU safe).
        fingertip_handles = self.fingertip_handles.to(self.rigid_body_states.device)
        self.fingertip_pos = self.rigid_body_states[:, fingertip_handles][:, :, 0:3]
        base_pose = self.rigid_body_states[:, self.virtual_hand_base_handle, 0:7]
        base_pos = base_pose[:, 0:3]
        base_quat = base_pose[:, 3:7]
        fingertip_delta_world = self.fingertip_pos - base_pos.unsqueeze(1)
        self.fingertip_pos_local = quat_rotate_inverse(
            base_quat.unsqueeze(1).expand(-1, fingertip_delta_world.shape[1], -1).reshape(-1, 4),
            fingertip_delta_world.reshape(-1, 3),
        ).reshape(self.num_envs, -1, 3)

        
        self.compute_contact_observations()

    def compute_contact_observations(self):
        arm_dof_num = self.arm_dof_num
        hand_start = arm_dof_num
        hand_end = self.num_arm_hand_dofs
        obs_action_start = self.num_arm_hand_dofs
        obs_contact_start = obs_action_start + self.obs_target_dim
        obs_spin_start = obs_contact_start + self.finger_tactile_obs_dim
        if self.asymmetric_obs:
            self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(
                self.arm_hand_dof_pos, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
            )
            self.states_buf[:, self.num_arm_hand_dofs:2 * self.num_arm_hand_dofs] = (
                self.vel_obs_scale * self.arm_hand_dof_vel
            )
            self.states_buf[:, 2 * self.num_arm_hand_dofs:3 * self.num_arm_hand_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )

            obj_obs_start = 3 * self.num_arm_hand_dofs
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            obs_end = 3 * self.num_arm_hand_dofs + 13
            self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
            self.states_buf[:, obs_end + self.num_actions: obs_end + self.num_actions + 24] = (
                self.spin_axis.repeat(1, 8)
            )
            all_contact = self.contact_tensor.view(self.num_envs, -1, 3).clone()
            all_contact = torch.norm(all_contact, dim=-1).float()
            all_contact = torch.where(all_contact >= 20.0, torch.ones_like(all_contact), all_contact / 20.0)
            contact_dim = all_contact.shape[1]
            self.states_buf[
                :, obs_end + self.num_actions + 24: obs_end + self.num_actions + 24 + contact_dim
            ] = all_contact
            end_pos = obs_end + self.num_actions + 24 + contact_dim
            self.states_buf[:, end_pos:end_pos + (hand_end - hand_start)] = self.prev_targets[:, hand_start:hand_end]

        self.last_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(
            self.arm_hand_dof_pos, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits
        )
        if arm_dof_num > 0:
            self.last_obs_buf[:, 0:arm_dof_num] = 0.0
        if self.obs_target_dim > 0:
            self.last_obs_buf[:, obs_action_start:obs_action_start + self.obs_target_dim] = 0

        contacts = self.contact_tensor.view(self.num_envs, -1, 3).clone()
        contacts = contacts[:, self.sensor_handle_indices, :]

        contacts = torch.norm(contacts, dim=-1)
        contacts = contacts * self.sensor_valid_mask.unsqueeze(0)
        gt_contacts = torch.where(contacts >= self.tac_thresh, 1.0, 0.0).clone()
        self.gt_contacts = gt_contacts
        contacts = torch.where(contacts >= self.contact_thresh, 1.0, 0.0)

        latency_samples = torch.rand_like(self.last_contacts)
        latency = torch.where(latency_samples < self.latency, 1, 0)
        self.last_contacts = self.last_contacts * latency + contacts * (1 - latency)

        mask = torch.rand_like(self.last_contacts)
        mask = torch.where(mask < self.sensor_noise, 0.0, 1.0)

        sensed_contacts = torch.where(self.last_contacts > 0.1, mask * self.last_contacts, self.last_contacts)
        if self.use_disable:
            sensed_contacts[:, self.disable_sensor_idxes] = 0
        if self.disable_tac_obs:
            self.sensed_contacts = torch.zeros_like(sensed_contacts)
        else:
            self.sensed_contacts = sensed_contacts
        if not self.headless:
            if self.viewer:
                self.debug_contacts = self.sensed_contacts.detach().cpu().numpy()
        if self.finger_tactile_obs_dim > 0:
            self.last_obs_buf[:, obs_contact_start:obs_contact_start + self.finger_tactile_obs_dim] = self.sensed_contacts
        self.last_obs_buf[:, obs_spin_start:obs_spin_start + 24] = self.spin_axis.repeat(1, 8)
        self.last_obs_buf[:, hand_start:hand_end] += (
            (torch.rand_like(self.last_obs_buf[:, hand_start:hand_end]) - 0.5) * 2 * 0.06
        )

        if arm_dof_num > 0:
            self.last_obs_buf[:, obs_action_start:obs_action_start + arm_dof_num] = 0
        if self.obs_target_dim > 0:
            self.last_obs_buf[
                :, obs_action_start + arm_dof_num:obs_action_start + self.obs_target_dim
            ] = unscale(self.prev_targets, self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)[
                :, hand_start:hand_end
            ]
        init_obs_ids = torch.where(self.init_stack_buf == 1)
        self.init_stack_buf[init_obs_ids] = 0
        self.obs_buf[init_obs_ids] = self.last_obs_buf[init_obs_ids].repeat(1, self.n_stack)
        self.obs_buf = torch.cat((self.last_obs_buf.clone(), self.obs_buf[:, :-self.n_obs_dim]), dim=-1)

        self.finger_contacts = gt_contacts

    def reset_spin_axis(self, env_ids, init_quat=None):
        env_ids_torch = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Reset the init_quat...
        if init_quat is None:
            self.object_init_quat[
                env_ids_torch] = self.root_state_tensor[self.object_indices[env_ids_torch], 3:7]
        else:
            self.object_init_quat[env_ids_torch] = init_quat

        self.object_init_pos[env_ids_torch] = self.root_state_tensor[self.object_indices[env_ids_torch], 0:3]
        # Reset the axis
        self.spin_axis[env_ids_torch] = self.all_spin_choices[torch.randint(0, int(self.all_spin_choices.size(0)),
                                                                            (len(env_ids), ))]
        return

    def _init_in_process_perturbation_state(self):
        direction = torch.tensor(
            self.perturb_direction_cfg, dtype=torch.float, device=self.device
        )
        direction_norm = torch.norm(direction)
        if direction_norm < 1e-8:
            raise ValueError("inProcessPerturbation.direction must be non-zero")
        self.perturb_direction = direction / direction_norm
        self.perturb_last_trigger_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )

    def _reset_in_process_perturbation(self, env_ids):
        if not hasattr(self, "perturb_last_trigger_step"):
            return
        env_ids_torch = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_torch.numel() == 0:
            return
        self.perturb_last_trigger_step[env_ids_torch] = -1

    def _set_object_root_states_for_envs(self, env_ids_torch):
        if env_ids_torch.numel() == 0:
            return
        obj_indices = self.object_indices[env_ids_torch].reshape(-1).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(obj_indices),
            obj_indices.numel(),
        )

    def _update_in_process_perturbation(self):
        if not self.in_process_perturb_enabled:
            return
        if self.perturb_interval_control_steps <= 0 or self.perturb_distance <= 0.0:
            return

        progress = self.progress_buf
        trigger_mask = (
            (progress > 0)
            & ((progress % self.perturb_interval_control_steps) == 0)
            & (self.perturb_last_trigger_step != progress)
            & (self.reset_buf == 0)
        )
        trigger_ids = torch.where(trigger_mask)[0]
        if trigger_ids.numel() == 0:
            return

        obj_indices = self.object_indices[trigger_ids]
        self.root_state_tensor[obj_indices, 0:3] += self.perturb_distance * self.perturb_direction.unsqueeze(0)
        self.perturb_last_trigger_step[trigger_ids] = progress[trigger_ids]
        self._set_object_root_states_for_envs(trigger_ids)

    def update_controller(self):
        previous_dof_pos = self.arm_hand_dof_pos.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._update_in_process_perturbation()

        if self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        if self.torque_control:
            dof_pos = self.arm_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            self.torques = torques.clone()
            effort_limits = self.arm_hand_dof_effort_limits.unsqueeze(0)
            self.torques = torch.clamp(self.torques, min=-effort_limits, max=effort_limits)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        return

    def refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

    def reset_idx(self, env_ids, goal_env_ids, is_test=False):
        if self.gravity_curriculum_enabled and len(env_ids) > 0:
            full_run_outcomes = (self.progress_buf[env_ids] >= (self.max_episode_length - 1)).float()
            self._update_gravity_curriculum(full_run_outcomes)
        if len(env_ids) > 0:
            self._sample_gravity_for_resets(env_ids)
        # generate random values
        if self.randomize:  #true
            self.apply_randomizations(self.randomization_params)

        for env_id in env_ids:
            env = self.envs[env_id]
            handle = self.gym.find_actor_handle(env, 'object')
            prop = self.gym.get_actor_rigid_body_properties(env, handle)
            if not is_test:
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
            else:
                for p in prop:
                    p.mass = np.random.uniform(self.randomize_mass_lower, self.randomize_mass_upper)
            self.gym.set_actor_rigid_body_properties(env, handle, prop)
            object_masses = torch.tensor([p.mass for p in prop], dtype=torch.float, device=self.device)
            object_end = min(self.gravity_object_body_start + object_masses.shape[0], self.num_bodies)
            self.gravity_body_masses[env_id, self.gravity_object_body_start:object_end] = (
                object_masses[: object_end - self.gravity_object_body_start]
            )

            self._randomize_hand_object_friction(
                self.envs[env_id], self.arm_hands[env_id], self.objects[env_id]
            )


        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)


        # reset contact
        self.contact_thresh[env_ids] = (
            (torch.rand_like(self.contact_thresh[env_ids]) * 2.0 - 1.0) * self.tac_thresh_rand + self.tac_thresh
        )
        self.last_contacts[env_ids] = 0.0

        self.obs_buf[env_ids] = 0
        self.init_stack_buf[env_ids] = 1

        base_p_gain = self.p_gain_defaults.unsqueeze(0).repeat(len(env_ids), 1)
        base_d_gain = self.d_gain_defaults.unsqueeze(0).repeat(len(env_ids), 1)
        if self.pd_dr_enabled and self.randomize and (not is_test):
            # Per-env, per-joint independent randomization on top of tuned PD gains.
            p_scale = torch_rand_float(
                self.pd_dr_scale_lower,
                self.pd_dr_scale_upper,
                (len(env_ids), self.num_actions),
                device=self.device,
            )
            d_scale = torch_rand_float(
                self.pd_dr_scale_lower,
                self.pd_dr_scale_upper,
                (len(env_ids), self.num_actions),
                device=self.device,
            )
            self.p_gain[env_ids] = torch.clamp(base_p_gain * p_scale, min=self.pd_dr_min_gain)
            self.d_gain[env_ids] = torch.clamp(base_d_gain * d_scale, min=self.pd_dr_min_gain)
        else:
            self.p_gain[env_ids] = base_p_gain
            self.d_gain[env_ids] = base_d_gain

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        # Per-axis position noise: reset_position_noise is shape (3,), broadcast against rand_floats[:, 0:3] in U(-1, 1).
        # Use as_tensor to tolerate overrides that set this attr to a plain list/tuple (e.g. from deploy scripts).
        reset_pos_noise = torch.as_tensor(
            self.reset_position_noise, device=self.device, dtype=torch.float
        )
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = self.object_init_state[env_ids, ..., 0:3] + \
            reset_pos_noise * rand_floats[:, 0:3]

        base_object_rot = self.object_init_state[env_ids, 3:7]

        # Controlled random-axis rotation in degrees:
        # angle_rad = resetRotationNoise * U(-1, 1) * pi / 180.
        # Set resetRotationNoise=0 to disable initial random rotation.
        rot_noise_scale_deg = float(self.reset_rotation_noise)
        rot_noise_angle_rad = torch.deg2rad(rot_noise_scale_deg * rand_floats[:, 4])
        random_axis = rand_floats[:, 1:4]
        random_axis = random_axis / torch.clamp(
            torch.norm(random_axis, dim=-1, keepdim=True),
            min=1e-8,
        )
        delta_object_rot = quat_from_angle_axis(rot_noise_angle_rad, random_axis)
        new_object_rot = quat_mul(base_object_rot, delta_object_rot)

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot.clone()
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids]]).to(torch.int32))

        # reset spinning axis
        self.reset_spin_axis(env_ids, init_quat=new_object_rot)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        self.arm_hand_dof_pos[env_ids, :] = self.arm_hand_dof_default_pos
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel 

        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        for env_id in env_ids:
            for (idx, qpos) in self.hand_override_info:
                self.dof_state[env_id * self.num_arm_hand_dofs + idx, 0] = qpos

        # Always initialize targets from current state tensor after all reset overrides.
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_pos[env_ids, :]
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = self.arm_hand_dof_pos[env_ids, :]

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.no_spin_counter[env_ids] = 0
        self._reset_in_process_perturbation(env_ids)

        for env_id in env_ids:
            self.object_init_pos[env_id] = self.root_state_tensor[self.object_indices[env_id], 0:3]
            self.object_init_quat[env_id] = self.root_state_tensor[self.object_indices[env_id], 3:7]
        if self.spin_trace_enabled:
            for env_id in env_ids:
                self._record_spin_trace_reset_step(int(env_id))

        # Weakly-coupled hook so subclasses (e.g. camera task) can rotate per-episode video files
        # without the base task needing to know about recording state.
        on_reset = getattr(self, "on_episode_reset", None)
        if callable(on_reset):
            on_reset(env_ids)

    def pre_physics_step(self, actions):
        # print("time:",time.time()-self.time)
        self.time = time.time()

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)
        self.last_object_rot = self.root_state_tensor[self.object_indices, 3:7].clone()

        self.actions = actions.clone().to(self.device)

        if self.control_mode == "relative":  # default legacy behavior

            self.actions = self.actions * self.act_moving_average + self.last_actions * (1.0 - self.act_moving_average)
            self.relative_scale_tensor = torch.full_like(self.relative_scale_tensor, self.relative_scale) * \
                                         (1 + (torch.rand_like(self.relative_scale_tensor) - 0.5) * 0.1)
            targets = self.prev_targets + self.relative_scale_tensor * self.actions

            # # # yinjie
            # scaled_actions = torch.clamp(self.relative_scale_tensor * self.actions, -0.05, 0.05)
            # targets = self.prev_targets + scaled_actions

            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.arm_hand_dof_lower_limits[
                                                                              self.actuated_dof_indices],
                                                                          self.arm_hand_dof_upper_limits[
                                                                              self.actuated_dof_indices])
            
            self.prev_targets = self.cur_targets.clone()
            self.last_actions = self.actions.clone().to(self.device)

        elif self.control_mode == "absolute":
            self.actions = self.actions * self.act_moving_average + self.last_actions * (1.0 - self.act_moving_average)
            base_targets = self.arm_hand_dof_default_pos.unsqueeze(0).repeat(self.num_envs, 1)
            targets = base_targets + self.absolute_scale * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = targets
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                self.arm_hand_dof_upper_limits[self.actuated_dof_indices])
            self.last_actions = self.actions.clone().to(self.device)

        else:
            raise RuntimeError(f"Unknown control_mode: {self.control_mode}")
            

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        # self.cur_targets[:, [7+3,7+7,7+11,7+15]] = 0
        self._apply_gravity_equivalent_force()
        if self.force_scale > 0.0:
            self.rb_forces *= self.force_decay
            obj_mass = to_torch(
                [self.gym.get_actor_rigid_body_properties(env, self.gym.find_actor_handle(env, 'object'))[0].mass for
                 env in self.envs], device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (torch.less(torch.rand(self.num_envs, device=self.device), prob)).nonzero()  # Below threshold.
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * obj_mass[force_indices, None] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def debug(self, info):
        print(info, self.root_state_tensor[self.object_indices, 3:7])

    def _build_spin_trace_finger_layout(self):
        finger_defs = [
            ("thumb", "right_thumb_"),
            ("index", "right_index_"),
            ("middle", "right_middle_"),
            ("ring", "right_ring_"),
            ("pinky", "right_pinky_"),
        ]
        hand_names = list(getattr(self, "arm_hand_dof_names", [])[:self.num_actions])
        finger_layout = {}

        def _joint_rank(joint_name):
            if "_CMC" in joint_name:
                return 0
            if "_MCP" in joint_name:
                return 1
            if "_PIP" in joint_name:
                return 2
            if "_DIP" in joint_name:
                return 3
            if "_IP" in joint_name:
                return 4
            return 99

        def _component_rank(joint_name):
            if joint_name.endswith("_FE"):
                return 0
            if joint_name.endswith("_AA"):
                return 1
            return 2

        for finger_key, finger_prefix in finger_defs:
            joint_infos = []
            for joint_idx, joint_name in enumerate(hand_names):
                if not joint_name.startswith(finger_prefix):
                    continue
                joint_infos.append(
                    (
                        _joint_rank(joint_name),
                        _component_rank(joint_name),
                        joint_name,
                        joint_idx,
                    )
                )
            joint_infos.sort(key=lambda x: (x[0], x[1], x[2]))
            finger_layout[finger_key] = {
                "joint_indices": [item[3] for item in joint_infos],
                "joint_names": [item[2] for item in joint_infos],
            }
        self._spin_trace_finger_layout = finger_layout

    def _build_spin_trace_step_row(self, env_id):
        step_idx = int(self.progress_buf[env_id].item())
        time_s = step_idx * self.control_dt

        angvel_env = self.object_angvel[env_id]
        wx = float(angvel_env[0].item())
        wy = float(angvel_env[1].item())
        wz = float(angvel_env[2].item())
        object_pos_env = self.object_pos[env_id]
        obj_x = float(object_pos_env[0].item())
        obj_y = float(object_pos_env[1].item())
        obj_z = float(object_pos_env[2].item())
        spin_axis_scaled_new = float(self.last_spin_delta_axis[env_id].item())
        spin_offaxis_scaled_new = float(self.last_spin_delta_offaxis[env_id].item())
        dt_safe = max(float(self.control_dt), 1e-8)

        spin_axis_deg_s_new = float(np.degrees((spin_axis_scaled_new / 20.0) / dt_safe))
        spin_offaxis_deg_s_new = float(np.degrees((spin_offaxis_scaled_new / 20.0) / dt_safe))
        row = {
            "step": float(step_idx),
            "time_s": float(time_s),
            "obj_x": obj_x,
            "obj_y": obj_y,
            "obj_z": obj_z,
            "wx_deg_s": float(np.degrees(wx)),
            "wy_deg_s": float(np.degrees(wy)),
            "wz_deg_s": float(np.degrees(wz)),
            "spin_rate_axis_deg_s": spin_axis_deg_s_new,
            "spin_rate_offaxis_deg_s": spin_offaxis_deg_s_new,
        }
        contacts_env = None
        try:
            contacts_raw = self.contact_tensor.view(self.num_envs, -1, 3)[env_id, self.sensor_handle_indices, :]
            contacts_env = torch.norm(contacts_raw, dim=-1) * self.sensor_valid_mask
        except Exception:
            contacts_env = self.sensed_contacts[env_id]
        hand_joint_state_env = self.arm_hand_dof_pos[env_id, :self.num_actions]
        hand_joint_action_env = self.cur_targets[env_id, :self.num_actions]
        fingertip_local_env = self.fingertip_pos_local[env_id]

        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        for finger_idx, finger_key in enumerate(finger_order):
            row[f"{finger_key}_touch"] = float(contacts_env[finger_idx].item())
            row[f"{finger_key}_tip_x"] = float(fingertip_local_env[finger_idx, 0].item())
            row[f"{finger_key}_tip_y"] = float(fingertip_local_env[finger_idx, 1].item())
            row[f"{finger_key}_tip_z"] = float(fingertip_local_env[finger_idx, 2].item())
            layout = self._spin_trace_finger_layout.get(finger_key, {})
            joint_indices = layout.get("joint_indices", [])
            for joint_idx in joint_indices:
                joint_name = self.arm_hand_dof_names[joint_idx]
                state_deg = float(np.degrees(hand_joint_state_env[joint_idx].item()))
                action_deg = float(np.degrees(hand_joint_action_env[joint_idx].item()))
                # Keep legacy key for compatibility; it represents simulation state.
                row[f"{finger_key}_{joint_name}_deg"] = state_deg
                row[f"{finger_key}_{joint_name}_state_deg"] = state_deg
                row[f"{finger_key}_{joint_name}_action_deg"] = action_deg
        return row

    def _record_spin_trace_step(self):
        if not self.spin_trace_enabled:
            return
        if (
            not hasattr(self, "last_spin_delta_axis")
            or not hasattr(self, "last_spin_delta_offaxis")
        ):
            return
        if self.num_envs <= 0:
            return
        if self._spin_trace_finger_layout is None:
            self._build_spin_trace_finger_layout()

        for raw_env_id in self.spin_trace_env_ids:
            env_id = max(0, min(int(raw_env_id), self.num_envs - 1))
            row = self._build_spin_trace_step_row(env_id)
            rows = self._spin_trace_rows_per_env.setdefault(env_id, [])
            rows.append(row)
            # Maintain legacy single-env attribute pointing at the primary env.
            if env_id == int(self.spin_trace_env_id):
                self._spin_trace_rows = rows
            if int(self.reset_buf[env_id].item()) > 0:
                self._flush_spin_trace_episode_for_env(env_id)

    def _record_spin_trace_reset_step(self, env_id):
        if not self.spin_trace_enabled:
            return
        if self.num_envs <= 0:
            return
        if self._spin_trace_finger_layout is None:
            self._build_spin_trace_finger_layout()

        env_id = max(0, min(int(env_id), self.num_envs - 1))
        if env_id not in self.spin_trace_env_ids:
            return

        # Reset snapshot is the true episode initialization state.
        object_state_env = self.root_state_tensor[self.object_indices[env_id]]
        if object_state_env.dim() == 2:
            object_state_env = object_state_env.mean(dim=0)
        row = {
            "step": 0.0,
            "time_s": 0.0,
            "obj_x": float(object_state_env[0].item()),
            "obj_y": float(object_state_env[1].item()),
            "obj_z": float(object_state_env[2].item()),
            "wx_deg_s": float(np.degrees(float(object_state_env[10].item()))),
            "wy_deg_s": float(np.degrees(float(object_state_env[11].item()))),
            "wz_deg_s": float(np.degrees(float(object_state_env[12].item()))),
            "spin_rate_axis_deg_s": 0.0,
            "spin_rate_offaxis_deg_s": 0.0,
        }

        contacts_env = None
        try:
            contacts_raw = self.contact_tensor.view(self.num_envs, -1, 3)[env_id, self.sensor_handle_indices, :]
            contacts_env = torch.norm(contacts_raw, dim=-1) * self.sensor_valid_mask
        except Exception:
            if hasattr(self, "sensed_contacts"):
                contacts_env = self.sensed_contacts[env_id]
            elif hasattr(self, "sensor_valid_mask"):
                contacts_env = torch.zeros_like(self.sensor_valid_mask, dtype=torch.float, device=self.device)
            else:
                sensor_dim = int(self.sensor_handle_indices.shape[0]) if hasattr(self, "sensor_handle_indices") else 5
                contacts_env = torch.zeros((sensor_dim,), dtype=torch.float, device=self.device)
        hand_joint_state_env = self.arm_hand_dof_pos[env_id, :self.num_actions]
        hand_joint_action_env = self.cur_targets[env_id, :self.num_actions]

        self.gym.refresh_rigid_body_state_tensor(self.sim)
        fingertip_handles = self.fingertip_handles.to(self.rigid_body_states.device)
        fingertip_pos_env = self.rigid_body_states[env_id, fingertip_handles][:, 0:3]
        base_pose_env = self.rigid_body_states[env_id, self.virtual_hand_base_handle, 0:7]
        base_pos_env = base_pose_env[0:3]
        base_quat_env = base_pose_env[3:7]
        fingertip_local_env = quat_rotate_inverse(
            base_quat_env.unsqueeze(0).expand(fingertip_pos_env.shape[0], -1),
            fingertip_pos_env - base_pos_env.unsqueeze(0),
        )

        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        for finger_idx, finger_key in enumerate(finger_order):
            row[f"{finger_key}_touch"] = float(contacts_env[finger_idx].item())
            row[f"{finger_key}_tip_x"] = float(fingertip_local_env[finger_idx, 0].item())
            row[f"{finger_key}_tip_y"] = float(fingertip_local_env[finger_idx, 1].item())
            row[f"{finger_key}_tip_z"] = float(fingertip_local_env[finger_idx, 2].item())
            layout = self._spin_trace_finger_layout.get(finger_key, {})
            joint_indices = layout.get("joint_indices", [])
            for joint_idx in joint_indices:
                joint_name = self.arm_hand_dof_names[joint_idx]
                state_deg = float(np.degrees(hand_joint_state_env[joint_idx].item()))
                action_deg = float(np.degrees(hand_joint_action_env[joint_idx].item()))
                row[f"{finger_key}_{joint_name}_deg"] = state_deg
                row[f"{finger_key}_{joint_name}_state_deg"] = state_deg
                row[f"{finger_key}_{joint_name}_action_deg"] = action_deg

        rows = self._spin_trace_rows_per_env.setdefault(env_id, [])
        rows.append(row)
        if env_id == int(self.spin_trace_env_id):
            self._spin_trace_rows = rows

    def _flush_spin_trace_episode_for_env(self, env_id):
        env_id = int(env_id)
        rows = self._spin_trace_rows_per_env.get(env_id)
        if not rows:
            return

        episode_id = int(
            self._spin_trace_episode_idx_per_env.get(env_id, self.spin_trace_episode_idx)
        )
        self._spin_trace_episode_idx_per_env[env_id] = episode_id + 1
        # Keep the legacy single-env counter in sync with the primary env so
        # downstream code that reads spin_trace_episode_idx still advances.
        if env_id == int(self.spin_trace_env_id):
            self.spin_trace_episode_idx = episode_id + 1
        # Drain the per-env buffer.
        self._spin_trace_rows_per_env[env_id] = []
        if env_id == int(self.spin_trace_env_id):
            self._spin_trace_rows = self._spin_trace_rows_per_env[env_id]

        # Resolve output paths. If a subclass installed a resolver (camera task
        # routes spin trace next to the matching demo video), use it; else
        # fall back to the legacy single-folder layout.
        csv_path = None
        png_path = None
        if callable(self._spin_trace_output_path_resolver):
            try:
                resolved = self._spin_trace_output_path_resolver(env_id, episode_id)
                if isinstance(resolved, dict):
                    csv_path = resolved.get("csv_path")
                    png_path = resolved.get("png_path")
            except Exception as exc:
                print(f"[spin_trace] path resolver failed env={env_id}: {exc}")
                csv_path = None
                png_path = None

        csv_path, png_path, err = flush_spin_trace_episode(
            rows=rows,
            output_dir=self.spin_trace_output_dir,
            episode_idx=episode_id,
            finger_layout=self._spin_trace_finger_layout,
            title=f"Spin Trace env={env_id}",
            csv_path=csv_path,
            png_path=png_path,
        )
        if err is not None:
            print(f"[spin_trace] plot failed env={env_id} ({err}), csv saved to {csv_path}")
            return
        print(f"[spin_trace] saved env={env_id} csv={csv_path}, png={png_path}")

    def _flush_spin_trace_episode(self):
        # Backward-compatible flush for legacy single-env callers.
        self._flush_spin_trace_episode_for_env(int(self.spin_trace_env_id))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        # print(f'progress_buf:{self.progress_buf[0].item()}')

        if self.rotation_axis == 'all':
            env_ids = list(torch.where(torch.rand(self.num_envs) < 1 / 500)[0])
            self.reset_spin_axis(env_ids)

        # Keep observation and reward aligned to the same spin axis.
        self.compute_observations()

        self.compute_reward(self.actions)
        self._record_spin_trace_step()
        self.fetch_camera_observations()

        condition = True if (self.viewer and (not self.headless)) else False
        if condition and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            main_vector = self.spin_axis.clone()

            for i in range(self.num_envs):

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # visualize target spin axis in world frame (fixed, not object-frame rotated)
                objectm = (self.object_pos[i] + main_vector[i]).cpu().numpy()
                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objectm[0], objectm[1], objectm[2]], [0.85, 0.1, 0.85])

        # We do some debug visualization.
        if condition and not getattr(self, "preserve_urdf_colors", False):
            for env in range(len(self.envs)):
                for i, contact_idx in enumerate(list(self.sensor_handle_indices)):
                    
                    if self.debug_contacts[env, i] > 0.0:
                        self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(0.0, 1.0, 0.0))
                    else:
                        self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                                                      contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                                                      gymapi.Vec3(1.0, 0.0, 0.0))

                import math
                if self.debug_viz:
                    if math.fabs(float(self.spin_axis[env, 0])) > 0.0:
                        color = (0.0, 0.0, 1.0)
                    elif math.fabs(float(self.spin_axis[env, 1])) > 0.0:
                        color = (0.0, 1.0, 0.0)
                    else:
                        color = (1.0, 0.0, 0.0)

                    # for i, contact_idx in enumerate(list(self.arm_handle_indices)):
                    #     self.gym.set_rigid_body_color(self.envs[env], self.arm_hands[env],
                    #                                 contact_idx, gymapi.MESH_VISUAL_AND_COLLISION,
                    #                                 gymapi.Vec3(*color))
#####################################################################
###=========================jit functions=========================###
#####################################################################

@conditional_jit
def compute_hand_reward_finger(
    spin_coef, vel_coef, torque_coef, work_coef, contact_coef, finger_coef,
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, fingertip_pos, object_pos, object_rot, object_init_pos, object_init_rot, object_linvel, object_angvel, target_pos, target_rot,
    finger_contacts,
    control_error, control_penalty_scale: float, actions, action_penalty_scale: float,
    hand_pose_coef, hand_pose_delta_sq, obj_init_pos_dev_coef, distance_reward_deadzone, distance_reward_penalty_width, axis_dev_penalty_coef,
    reward_max_spin_rate,
    no_spin_counter, no_spin_angvel_thresh, no_spin_max_steps: int, no_spin_reset_penalty,
    fall_dist: float,
    fall_penalty: float, main_vector, spin_delta_axis, spin_delta_offaxis, torque_penalty, work_penalty,
    max_consecutive_successes: int, av_factor: float
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    object_pos_repeat = object_pos.reshape(-1, 1, 3).repeat(1, 5, 1)
    distance = torch.sqrt(((object_pos_repeat - fingertip_pos) ** 2).sum(-1))
    # Scheme A: no penalty inside deadzone; quadratic penalty outside.
    distance_excess = torch.clamp(distance - distance_reward_deadzone, min=0.0)
    normalized_distance_excess = distance_excess / torch.clamp(distance_reward_penalty_width, min=1e-6)
    distance_penalty = torch.clamp(normalized_distance_excess * normalized_distance_excess, 0.0, 1.0).mean(-1)
    distance_reward = -finger_coef * distance_penalty

    inverse_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_init_rot)).transpose(1, 2)
    forward_rotation_matrix = transform.quaternion_to_matrix(xyzw_to_wxyz(object_rot))

    inverse_main_vector = torch.bmm(inverse_rotation_matrix, main_vector.unsqueeze(-1))
    current_main_vector = torch.bmm(forward_rotation_matrix, inverse_main_vector).squeeze()
    angle_cos = torch.sum(main_vector * current_main_vector, dim=-1)
    angle_cos = torch.clamp(angle_cos, -1.0, 1.0)
    angle_difference = torch.arccos(angle_cos) # The cosine similarity.
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    # spin_delta_axis = torch.clip(spin_delta_axis, -3.14, 3.14)
    spin_delta_axis = torch.clip(spin_delta_axis, -6.28, reward_max_spin_rate)
    spin_reward = spin_coef * spin_delta_axis
    axis_dev_penalty = -axis_dev_penalty_coef * (spin_delta_offaxis ** 2)
    vel_reward = vel_coef * torch.norm(object_linvel, dim=-1)
    action_penalty = torch.sum(actions ** 2, dim=-1)

    finger_contact_sum = finger_contacts.sum(dim=-1).float()
    finger_contact_sum = torch.clip(finger_contact_sum, 0.0, 5.0)

    # The hand must hold the object. Otherwise it is penalized.
    contact_reward = finger_contact_sum * contact_coef 
    hand_pose_reward = torch.exp(-hand_pose_delta_sq)

    # Penalize object translation away from episode initial position: -coef * ||p - p0||^2
    dpos = object_pos - object_init_pos
    dist_sq = (dpos * dpos).sum(-1)
    obj_init_pos_penalty = -obj_init_pos_dev_coef * dist_sq
    axis_spin_speed = torch.abs((object_angvel * main_vector).sum(-1))
    no_spin_counter = torch.where(
        axis_spin_speed < no_spin_angvel_thresh,
        no_spin_counter + torch.ones_like(no_spin_counter),
        torch.zeros_like(no_spin_counter),
    )
    no_spin_resets = no_spin_counter >= no_spin_max_steps
    no_spin_penalty = torch.where(
        no_spin_resets,
        torch.ones_like(rew_buf) * no_spin_reset_penalty,
        torch.zeros_like(rew_buf),
    )

    reward = spin_reward + vel_reward + contact_reward + distance_reward + \
             torque_penalty * torque_coef + work_penalty * work_coef + \
             action_penalty * action_penalty_scale + control_error * control_penalty_scale + \
             hand_pose_coef * hand_pose_reward + obj_init_pos_penalty + axis_dev_penalty + no_spin_penalty
    reward = torch.nan_to_num(reward, nan=-1e3, posinf=1e3, neginf=-1e3)

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) > 100.0, torch.ones_like(reset_goal_buf), reset_goal_buf)

    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(angle_difference > 0.4 * 3.1415926, torch.ones_like(reset_buf), resets)
    resets = torch.where(no_spin_resets, torch.ones_like(reset_buf), resets)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_corand_floatsnsecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) > 100.0, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    num_resets = torch.sum(resets)

    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    torque_penalty_term = torque_penalty * torque_coef
    work_penalty_term = work_penalty * work_coef
    action_penalty_term = action_penalty * action_penalty_scale
    control_penalty_term = control_error * control_penalty_scale
    hand_pose_reward_term = hand_pose_coef * hand_pose_reward
    obj_init_pos_penalty_term = obj_init_pos_penalty
    axis_dev_penalty_term = axis_dev_penalty
    no_spin_penalty_term = no_spin_penalty
    reward_total = reward

    return reward, resets, goal_resets, progress_buf, successes, cons_successes, \
           spin_reward, vel_reward, contact_reward, distance_reward, \
           torque_penalty_term, work_penalty_term, action_penalty_term, control_penalty_term, \
           hand_pose_reward_term, obj_init_pos_penalty_term, axis_dev_penalty_term, no_spin_penalty_term, reward_total, no_spin_counter

@conditional_jit
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@conditional_jit
def randomize_z_rotation(rand0, rand1, y_unit_tensor, z_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, z_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

@conditional_jit
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


def jit_identity(x):
    return x
