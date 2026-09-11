# Copyright (c) 2026 The WM-Craftnet Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import math
import atexit
import pathlib

import gym
import numpy as np
import torch
import torch.nn.functional as F
try:
    # torchvision >= 0.15
    from torchvision.transforms import v2 as tv_transforms
    _HAS_TORCHVISION_V2 = True
except ImportError:
    # Older torchvision releases expose the equivalent API under transforms.
    from torchvision import transforms as tv_transforms
    _HAS_TORCHVISION_V2 = False

from .wm_craftnet_rotation import RealmanSharpaHa4Rotation
from isaacgymenvs.utils.recorders.evaluation import VideoEpisodeRecorder, scan_next_game_index
from isaacgymenvs.utils.recorders.recording import EpisodeRecordingManager
from isaacgymenvs.utils.recorders.metrics import EvalMetricsAggregator
from isaacgymenvs.utils.world_model import WorldModelAdapter
from isaacgym import gymapi, gymtorch


class RealmanSharpaHa4CameraRotation(RealmanSharpaHa4Rotation):
    """WM-Craftnet rotation task with camera observations and evaluation recording.

    Recording has four independent channels:
      - demo: high-resolution RGB from a third-person camera, for web showcase.
      - inference: low-resolution view matching the policy's camera.
      - wm_depth_pred: WorldModel posterior depth reconstruction.
      - wm_depth_gt: corresponding clean ground-truth depth.
    Either channel can be disabled. Each channel writes one mp4 per episode per env.
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        local_cfg = copy.deepcopy(cfg)
        cam_policy_cfg = local_cfg["env"].get("cameraPolicy", {})
        cam_demo_cfg = local_cfg["env"].get("cameraDemo", {})
        if not isinstance(cam_policy_cfg, dict):
            cam_policy_cfg = {}
        if not isinstance(cam_demo_cfg, dict):
            cam_demo_cfg = {}
        wm_cfg_raw = cam_policy_cfg.get("worldModel", {})
        if not isinstance(wm_cfg_raw, dict):
            wm_cfg_raw = {}
        self.world_model_enabled = bool(wm_cfg_raw.get("enabled", False))
        self._wm_cfg = wm_cfg_raw
        self.wm_adapter = None
        self.wm_tac_target_thresh = float(
            wm_cfg_raw.get("wm_tac_target_thresh", local_cfg["env"].get("TacThresh", 1.0))
        )
        policy_sensor_cfg = cam_policy_cfg.get("sensor", {})
        if not isinstance(policy_sensor_cfg, dict):
            policy_sensor_cfg = {}
        if len(policy_sensor_cfg) == 0:
            policy_sensor_cfg = copy.deepcopy(local_cfg["env"].get("camera", {}))
        local_cfg["env"]["camera"] = copy.deepcopy(policy_sensor_cfg)

        # === Core observation config ===
        self.camera_policy_enabled = bool(cam_policy_cfg.get("enabled", True))
        self.is_test_run = bool(local_cfg["env"].get("isTestRun", False))
        self.preserve_urdf_colors = True
        self.camera_mode = cam_policy_cfg.get("mode", "rgbd")
        self.max_depth = float(cam_policy_cfg.get("max_depth", 5.0))
        depth_pre_cfg = cam_policy_cfg.get("depth_preprocess", {})
        if not isinstance(depth_pre_cfg, dict):
            depth_pre_cfg = {}
        depth_crop_cfg = depth_pre_cfg.get("crop", {})
        if not isinstance(depth_crop_cfg, dict):
            depth_crop_cfg = {}
        depth_noise_cfg = depth_pre_cfg.get("noise", {})
        if not isinstance(depth_noise_cfg, dict):
            depth_noise_cfg = {}

        self.depth_crop_enabled = bool(depth_crop_cfg.get("enabled", False))
        self.depth_noise_enabled = bool(depth_noise_cfg.get("enabled", False))

        self.depth_discontinuity_invalidate_enabled = bool(
            depth_noise_cfg.get("discontinuity_invalidate_enabled", False)
        )
        self.depth_discontinuity_thresh_m = float(
            depth_noise_cfg.get("discontinuity_thresh_m", 0.02)
        )
        if self.depth_discontinuity_thresh_m < 0.0:
            raise ValueError(
                "cameraPolicy.depth_preprocess.noise.discontinuity_thresh_m must be >= 0.0, "
                f"got {self.depth_discontinuity_thresh_m}."
            )

        self.temporal_noise_enabled = bool(depth_noise_cfg.get("temporal_noise_enabled", False))
        self.temporal_noise_p_dropout = min(
            1.0, max(0.0, float(depth_noise_cfg.get("temporal_noise_p_dropout", 0.02)))
        )
        self.temporal_noise_corr = min(
            1.0, max(0.0, float(depth_noise_cfg.get("temporal_noise_corr", 0.98)))
        )
        self.depth_gaussian_noise_enabled = bool(
            depth_noise_cfg.get("gaussian_noise_enabled", False)
        )
        self.depth_gaussian_noise_std = max(
            0.0, float(depth_noise_cfg.get("gaussian_noise_std", 0.005))
        )
        self.depth_random_rotation_enabled = bool(
            depth_noise_cfg.get("random_rotation_enabled", False)
        )
        self.depth_random_rotation_deg = max(
            0.0, float(depth_noise_cfg.get("random_rotation_deg", 3.0))
        )
        self.depth_blur_enabled = bool(depth_noise_cfg.get("gaussian_blur_enabled", False))
        self.depth_blur_kernel = int(depth_noise_cfg.get("gaussian_kernel", 5))
        self.depth_blur_sigma = float(depth_noise_cfg.get("gaussian_sigma", 1.0))
        self.persistent_noise_mask = None
        if self.depth_blur_kernel < 1:
            self.depth_blur_kernel = 1
        if self.depth_blur_kernel % 2 == 0:
            self.depth_blur_kernel += 1

        cam_cfg_w = int(local_cfg["env"]["camera"].get("width", 96))
        cam_cfg_h = int(local_cfg["env"]["camera"].get("height", 72))
        self.depth_crop_top_px = max(0, int(depth_crop_cfg.get("top_px", 0)))
        self.depth_crop_bottom_px = max(0, int(depth_crop_cfg.get("bottom_px", 0)))
        self.depth_crop_left_px = max(0, int(depth_crop_cfg.get("left_px", 0)))
        self.depth_crop_right_px = max(0, int(depth_crop_cfg.get("right_px", 0)))
        self.depth_obs_width = cam_cfg_w
        self.depth_obs_height = cam_cfg_h
        if self.depth_crop_enabled and self.camera_mode in {"depth", "rgbd"}:
            self.depth_obs_width = cam_cfg_w - self.depth_crop_left_px - self.depth_crop_right_px
            self.depth_obs_height = cam_cfg_h - self.depth_crop_top_px - self.depth_crop_bottom_px
            if self.depth_obs_width <= 0 or self.depth_obs_height <= 0:
                raise ValueError(
                    "Invalid cameraPolicy.depth_preprocess.crop pixel crop: "
                    f"camera=({cam_cfg_w}, {cam_cfg_h}), "
                    f"crop_left/right=({self.depth_crop_left_px}, {self.depth_crop_right_px}), "
                    f"crop_top/bottom=({self.depth_crop_top_px}, {self.depth_crop_bottom_px})."
                )

        self.wm_enable_tac_pred = bool(wm_cfg_raw.get("wm_tac_pred", False))
        self.enable_tac_pred = self.wm_enable_tac_pred
        self.tac_pred_links = list(wm_cfg_raw.get("wm_tac_pred_links", []))
        self.tac_pred_link_handle_indices = None
        self.camera_step = 0

        wm_feature_record_cfg = local_cfg["env"].get("wmFeatureRecord", {}) or {}
        if not isinstance(wm_feature_record_cfg, dict):
            wm_feature_record_cfg = {}
        self.wm_feature_record_enabled = bool(wm_feature_record_cfg.get("enabled", False))
        self.wm_feature_record_output_path = str(wm_feature_record_cfg.get("output_path", "") or "")
        self.wm_feature_record_sample_interval = max(1, int(wm_feature_record_cfg.get("sample_interval", 5)))
        self.wm_feature_record_dtype = str(wm_feature_record_cfg.get("dtype", "float16")).lower()
        if self.wm_feature_record_dtype not in {"float16", "float32"}:
            raise ValueError(
                "task.env.wmFeatureRecord.dtype must be 'float16' or 'float32', "
                f"got {self.wm_feature_record_dtype!r}."
            )
        self._wm_feature_record_buffers = None
        self._wm_feature_record_flushed = False

        wm_depth_video_cfg = local_cfg["env"].get("wmDepthVideoRecord", {}) or {}
        if not isinstance(wm_depth_video_cfg, dict):
            wm_depth_video_cfg = {}
        self.wm_depth_video_record_cfg_enabled = bool(wm_depth_video_cfg.get("enabled", True))
        self.wm_depth_video_record_enabled = self.is_test_run and self.wm_depth_video_record_cfg_enabled

        # === Recording config (new schema) ===
        record_cfg = cam_demo_cfg.get("record", {})
        self.record_root_dir = record_cfg.get("root_dir", "outputs/wm_craftnet_videos")
        self.record_fps = int(record_cfg.get("fps", 30))
        self.record_bitrate = str(record_cfg.get("bitrate", "8M"))
        self.record_game_idx_base = int(record_cfg.get("game_idx_base", 0))
        self.record_filename_prefix = str(record_cfg.get("filename_prefix", ""))

        demo_cam_cfg = cam_demo_cfg.get("camera", {})
        if not isinstance(demo_cam_cfg, dict):
            demo_cam_cfg = {}
        self.demo_recorder_cfg_enabled = bool(cam_demo_cfg.get("enabled", False))
        self.demo_recorder_enabled = self.is_test_run and self.demo_recorder_cfg_enabled
        self.demo_camera_width = int(demo_cam_cfg.get("width", 2560))
        self.demo_camera_height = int(demo_cam_cfg.get("height", 2560))
        self.demo_camera_fov = float(demo_cam_cfg.get("fov", 69.4))
        self.demo_camera_pos = list(demo_cam_cfg.get("pos", [-0.07, -0.4, 1.40]))
        self.demo_camera_rot = list(demo_cam_cfg.get("rot", [0.0, -40.0, 90.0]))
        self.demo_camera_pose_noise_std = float(demo_cam_cfg.get("pose_noise_std", 0.0))
        demo_lookat_cfg = demo_cam_cfg.get("lookat", {})
        self.demo_camera_lookat_enabled = bool(demo_lookat_cfg.get("enabled", False))
        self.demo_camera_lookat_target_pos = list(
            demo_lookat_cfg.get("target_pos", [0.0, 0.0, 1.0])
        )

        infv_cfg = cam_demo_cfg.get("inference_video", {})
        self.inference_recorder_cfg_enabled = bool(infv_cfg.get("enabled", False))
        self.inference_recorder_enabled = self.is_test_run and self.inference_recorder_cfg_enabled

        # Axes overlay: draw object body-frame RGB axes directly onto demo frames
        # using same-frame object pose (no sim actors, no 1-frame lag).
        env_cfg = local_cfg["env"]
        self.show_object_axes = bool(env_cfg.get("showObjectAxes", False))
        self.object_axes_length = float(env_cfg.get("objectAxesLength", 0.12))
        self.object_axes_line_thickness_frac = float(
            env_cfg.get("objectAxesLineThicknessFrac", 0.008)
        )
        self.object_axes_arrow_half_base_frac = float(
            env_cfg.get("objectAxesArrowHalfBaseFrac", 0.005)
        )
        # Print one startup line for axes overlay, then stay quiet.
        self._axes_startup_logged = False
        # Lazily populated per-env (view_4x4, proj_4x4) numpy tuples. Cached
        # because demo camera is fixed to env and envs don't move.
        self._axes_view_proj_cache = None
        self._demo_cam_local_pos_per_env = []
        # Base task creates the hand actor at local position (0, 0, 1.0) in each env.
        # We use this as a robust anchor to estimate per-env world origin from
        # hand root states (more reliable than assuming grid layout).
        self._hand_local_anchor_pos = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._camera_task_closed = False
        # Pillow is already an imageio dep, so always available in practice; we
        # still guard the import defensively so a stripped env doesn't crash.
        self._axes_drawer_available = False
        if self.show_object_axes:
            try:
                from PIL import Image as _PilImage  # noqa: F401
                from PIL import ImageDraw as _PilDraw  # noqa: F401
                self._axes_drawer_available = True
            except Exception as exc:
                print(f"[axes-overlay] PIL unavailable ({exc!r}); overlay disabled.")
                self.show_object_axes = False

        self.demo_camera_handles = []
        self.demo_recorder = None
        self.inference_recorder = None
        self.wm_depth_pred_recorder = None
        self.wm_depth_gt_recorder = None
        self._record_output_root = None
        self._record_env_ids = []
        self._record_env_id_set = set(self._record_env_ids)
        self.recording_manager = EpisodeRecordingManager(
            owner=self,
            env_obj_names_fn=self._env_obj_names,
            hdf5_step_data_fn=self._hdf5_step_data_for_env,
            joint_names_fn=self._hdf5_joint_names,
            on_enter_recording_fn=self._on_recording_env_enter,
            on_finish_recording_fn=self._on_recording_env_finish,
        )
        self._periodic_eval_active = False
        self._periodic_eval_env_ids = []
        self._periodic_eval_record_env_ids = []
        # Per-env eval lifecycle state:
        #   "pending"   : env is finishing its pre-eval episode; no recording.
        #   "recording" : env's eval episode is in progress; collect data.
        #   "done"      : env finished its eval episode; freeze.
        # Eval ends when every metrics env reaches "done".
        self._periodic_eval_state = {}
        self._periodic_eval_prev_spin_trace = None
        self._periodic_eval_hdf5_enabled = False
        self._periodic_eval_hdf5_dir = ""
        self._periodic_eval_hdf5_recorder = None
        self._periodic_eval_hdf5_episode_idx = {}
        self._periodic_eval_hdf5_buffers = {}

        # Eval metrics aggregator (lazy-init in _init_eval_metrics; persisted across
        # train-time periodic evals so each eval window starts a fresh aggregator).
        eval_metrics_cfg = local_cfg["env"].get("evalMetrics", {}) or {}
        if not isinstance(eval_metrics_cfg, dict):
            eval_metrics_cfg = {}
        self.eval_metrics_cfg_enabled = bool(eval_metrics_cfg.get("enabled", True))
        self.eval_metrics_subdir = str(eval_metrics_cfg.get("outputSubdir", "metrics"))
        self.eval_metrics_invalid_max_steps = int(
            eval_metrics_cfg.get("invalidEpisodeMaxSteps", 10)
        )
        self.eval_metrics_window_size = int(
            eval_metrics_cfg.get("windowSize", 1024) or 1024
        )
        self.periodic_eval_max_steps_cfg = int(
            eval_metrics_cfg.get("periodicMaxSteps", 0) or 0
        )
        self.eval_metrics_aggregator = None
        self._metrics_env_ids = []
        self._metrics_env_id_set = set()
        self._metrics_env_ids_tensor = None
        self._metrics_steps = None
        self._metrics_sum_spin_reward = None
        self._metrics_sum_rot_radians = None
        self._metrics_sum_obj_angvel_norm = None
        self._metrics_sum_sq_obj_angvel_norm = None
        self._metrics_sum_obj_linvel_norm = None
        self._metrics_sum_torque_mean_abs = None
        self._metrics_sum_torque_norm = None
        self._last_eval_metrics_summary = None
        self._last_eval_metrics_epoch = None
        self._periodic_eval_epoch = -1
        self._periodic_eval_tb_writer = None
        self._periodic_eval_steps = 0
        self._periodic_eval_max_steps = 0
        self._periodic_eval_prev_is_test_run = False
        # Set per-step in post_physics_step right after compute_reward, consumed
        # in on_episode_reset to determine whether termination was a fall.
        self._last_terminated_by_fall = None

        need_camera_sensors = (
            self.camera_policy_enabled
            or self.demo_recorder_cfg_enabled
            or self.inference_recorder_cfg_enabled
        )
        local_cfg.setdefault("task", {})["enableCameraSensors"] = bool(need_camera_sensors)
        if self.camera_policy_enabled:
            if self.camera_mode == "rgb":
                local_cfg["env"]["observation"] = {"obs": {}, "rgb": {}}
            elif self.camera_mode == "depth":
                local_cfg["env"]["observation"] = {"obs": {}, "depth": {}}
            elif self.camera_mode == "rgbd":
                local_cfg["env"]["observation"] = {"obs": {}, "rgb": {}, "depth": {}}
            else:
                raise ValueError(f"Unsupported camera mode: {self.camera_mode}")
        else:
            # Policy camera off: drop any YAML placeholders (e.g. `obs:`) so VecTask only
            # enables rgb/depth/pointcloud from explicit modality keys.
            local_cfg["env"]["observation"] = {}

        super().__init__(
            local_cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        # Standalone test mode lifecycle (separate from periodic eval).
        # Test mode runs continuously; we just gate recording / metrics so that
        # each env contributes exactly one episode (its first natural one
        # post-startup) and exit once all 1024 envs are done.
        self._test_run_pending_exit = False
        self._test_eval_active = False
        # state per env: "pending" (still on the warm-up episode that started
        # at sim init), "recording" (collecting its one eval episode),
        # "done" (already flushed). Eval ends when all metrics envs are done.
        self._test_eval_state = {}
        self._test_eval_record_env_ids = []
        self._test_eval_metrics_env_ids = []

        # WorldModel adapter is only useful when the policy camera is on, since
        # we feed the post-preprocess depth (obs_dict[obs][depth]) as the image
        # observation. If you want a prop-only WM later, drop this guard.
        if self.world_model_enabled:
            if not (self.camera_policy_enabled and self.camera_mode in {"depth", "rgbd"}):
                raise ValueError(
                    "task.env.worldModel.enabled=True requires cameraPolicy.enabled=True "
                    "and cameraPolicy.mode in {depth, rgbd}; got "
                    f"camera_policy_enabled={self.camera_policy_enabled}, mode={self.camera_mode}."
                )
            repo_root = pathlib.Path(__file__).resolve().parents[2]
            # 'prop' for WM = the single-step proprioceptive vector (last_obs_buf),
            # NOT the stacked obs that the actor sees. Stacking would inflate
            # prop_dim and break the prop/image symmetry expected by WSM.
            prop_dim = int(self.n_obs_dim)
            image_shape = (int(self.depth_obs_height), int(self.depth_obs_width), 1)
            self.wm_adapter = WorldModelAdapter(
                repo_root=repo_root,
                num_envs=int(self.num_envs),
                num_actions=int(self.num_actions),
                prop_dim=prop_dim,
                image_shape=image_shape,
                wm_cfg=dict(self._wm_cfg),
                device=self.device,
                tac_pred_n_links=len(self.tac_pred_links),
                obj_pred_dim=int(getattr(self, "wm_obj_bps_target_dim", 0)),
            )
            # First-call defaults: every row is treated as first-of-episode
            # until the first append clears it. on_episode_reset will toggle
            # individual rows back to 1 thereafter.
            self._wm_pending_is_first = torch.ones(
                (self.num_envs,), device=self.device, dtype=torch.float32
            )
        if self.wm_depth_video_record_enabled and self.wm_adapter is None:
            raise ValueError(
                "task.env.wmDepthVideoRecord.enabled=True requires "
                "cameraPolicy.worldModel.enabled=True in test mode."
            )
        if self.camera_policy_enabled and self.camera_mode in {"depth", "rgbd"}:
            self.persistent_noise_mask = torch.zeros(
                (self.num_envs, self.depth_obs_height, self.depth_obs_width),
                dtype=torch.bool,
                device=self.device,
            )

        if self.camera_policy_enabled:
            if isinstance(self.obs_space, gym.spaces.Dict):
                cam_h = self.depth_obs_height if self.camera_mode in {"depth", "rgbd"} else int(self.cfg["env"]["camera"]["height"])
                cam_w = self.depth_obs_width if self.camera_mode in {"depth", "rgbd"} else int(self.cfg["env"]["camera"]["width"])
                updated_spaces = dict(self.obs_space.spaces)
                if self.camera_mode in {"rgb", "rgbd"}:
                    updated_spaces["rgb"] = gym.spaces.Box(
                        low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8
                    )
                if self.camera_mode in {"depth", "rgbd"}:
                    updated_spaces["depth"] = gym.spaces.Box(
                        low=0.0, high=1.0, shape=(cam_h, cam_w, 1), dtype=np.float32
                    )
                if self.world_model_enabled and self.wm_adapter is not None:
                    updated_spaces["wm_feature"] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.wm_adapter.wm_feature_dim,),
                        dtype=np.float32,
                    )
                if self.camera_mode in {"depth", "rgbd"}:
                    updated_spaces["pose_target"] = gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
                    )
                if self.enable_tac_pred:
                    updated_spaces["tac_contact_target"] = gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(len(self.tac_pred_links),),
                        dtype=np.float32,
                    )
                self.obs_space = gym.spaces.Dict(updated_spaces)
                if self.enable_tac_pred:
                    env0 = self.envs[0] if len(self.envs) > 0 else None
                    actor0 = self.arm_hands[0] if len(self.arm_hands) > 0 else None
                    if env0 is None or actor0 is None:
                        raise RuntimeError("Failed to resolve env/actor handles for tac_pred initialization.")
                    link_handles = [
                        self.gym.find_actor_rigid_body_handle(env0, actor0, link_name)
                        for link_name in self.tac_pred_links
                    ]
                    missing_links = [
                        link_name
                        for link_name, handle in zip(self.tac_pred_links, link_handles)
                        if handle < 0
                    ]
                    if missing_links:
                        raise ValueError(
                            f"tac_pred_links contains unknown rigid bodies: {missing_links}"
                        )
                    self.tac_pred_link_handle_indices = torch.tensor(
                        link_handles, device=self.device, dtype=torch.long
                    )

        if (not self.is_test_run) and self.eval_metrics_cfg_enabled:
            self._init_eval_metrics(range(int(self.num_envs)))

        if self.is_test_run and self._should_use_standalone_test_eval():
            self._init_standalone_test_eval()
            atexit.register(self.close)
        else:
            if self.demo_recorder_enabled:
                self._init_demo_cameras()
            if (
                self.demo_recorder_enabled
                or self.inference_recorder_enabled
                or self.wm_depth_video_record_enabled
            ):
                self._init_recording()
                atexit.register(self.close)

    # ------------------------------------------------------------------
    # Recording infrastructure
    # ------------------------------------------------------------------

    def _env_obj_names(self, env_ids):
        names = []
        for env_id in env_ids:
            names.append(self.used_training_objects[int(env_id) % len(self.used_training_objects)])
        return names

    def _hdf5_step_data_for_env(self, env_id_int):
        action_row = (
            self.cur_targets[env_id_int, :self.num_actions]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        state_row = (
            self.arm_hand_dof_pos[env_id_int, :self.num_actions]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        done_flag = bool(int(self.reset_buf[env_id_int].item()) > 0)
        return action_row, state_row, done_flag

    def _hdf5_joint_names(self):
        return [str(name) for name in self.arm_hand_dof_names[:self.num_actions]]

    def _on_recording_env_enter(self, env_id_int):
        self._spin_trace_rows_per_env[env_id_int] = []
        if env_id_int == int(self.spin_trace_env_id):
            self._spin_trace_rows = self._spin_trace_rows_per_env[env_id_int]
        if env_id_int not in self.spin_trace_env_ids:
            self.spin_trace_env_ids.append(env_id_int)
        if self.wm_depth_pred_recorder is not None:
            self.wm_depth_pred_recorder._close_env(env_id_int, drop_if_empty=True)
        if self.wm_depth_gt_recorder is not None:
            self.wm_depth_gt_recorder._close_env(env_id_int, drop_if_empty=True)

    def _on_recording_env_finish(self, env_id_int):
        if env_id_int in self.spin_trace_env_ids:
            try:
                self.spin_trace_env_ids.remove(env_id_int)
            except ValueError:
                pass
        if env_id_int in self._spin_trace_rows_per_env:
            self._spin_trace_rows_per_env[env_id_int] = []
        if env_id_int == int(self.spin_trace_env_id):
            self._spin_trace_rows = self._spin_trace_rows_per_env.get(env_id_int, [])
        if self.wm_depth_pred_recorder is not None:
            self.wm_depth_pred_recorder.end_episode(env_id_int)
        if self.wm_depth_gt_recorder is not None:
            self.wm_depth_gt_recorder.end_episode(env_id_int)

    def _init_recording(self):
        env_ids = list(self._record_env_ids) if self._record_env_ids else list(range(self.num_envs))
        self._record_env_ids = env_ids
        self._record_env_id_set = set(env_ids)
        self.recording_manager.init_recording(
            env_ids=env_ids,
            record_root=self.record_root_dir,
            demo_enabled=self.demo_recorder_enabled,
            inference_enabled=self.inference_recorder_enabled,
            fps=self.record_fps,
            bitrate=self.record_bitrate,
            game_idx_base=self.record_game_idx_base,
            filename_prefix=self.record_filename_prefix,
        )
        self.wm_depth_pred_recorder = None
        self.wm_depth_gt_recorder = None
        if self.wm_depth_video_record_enabled:
            env_obj_names = self._env_obj_names(env_ids)
            self.wm_depth_pred_recorder = VideoEpisodeRecorder(
                channel_name="wm_depth_pred",
                fps=self.record_fps,
                bitrate=self.record_bitrate,
                game_idx_base=self.record_game_idx_base,
                filename_prefix="",
            )
            self.wm_depth_pred_recorder.init(
                self.recording_manager.record_output_root,
                env_obj_names,
                env_ids=env_ids,
            )
            self.wm_depth_gt_recorder = VideoEpisodeRecorder(
                channel_name="wm_depth_gt",
                fps=self.record_fps,
                bitrate=self.record_bitrate,
                game_idx_base=self.record_game_idx_base,
                filename_prefix="",
            )
            self.wm_depth_gt_recorder.init(
                self.recording_manager.record_output_root,
                env_obj_names,
                env_ids=env_ids,
            )
        self._record_output_root = self.recording_manager.record_output_root

    def _close_all_recorders(self):
        self.recording_manager.close_recorders()
        if self.wm_depth_pred_recorder is not None:
            self.wm_depth_pred_recorder.close_all()
            self.wm_depth_pred_recorder = None
        if self.wm_depth_gt_recorder is not None:
            self.wm_depth_gt_recorder.close_all()
            self.wm_depth_gt_recorder = None

    def _destroy_demo_cameras(self):
        destroy_camera = getattr(self.gym, "destroy_camera_sensor", None)
        if callable(destroy_camera) and hasattr(self, "demo_camera_handles"):
            for env_id, camera_handle in enumerate(self.demo_camera_handles):
                if camera_handle is None:
                    continue
                try:
                    destroy_camera(self.sim, self.envs[env_id], camera_handle)
                except Exception:
                    pass

        self.demo_camera_handles = []
        self._demo_cam_local_pos_per_env = []
        self._axes_view_proj_cache = []

    def _init_eval_metrics(self, env_ids):
        if not self.eval_metrics_cfg_enabled:
            self.eval_metrics_aggregator = None
            self._metrics_env_ids = []
            self._metrics_env_id_set = set()
            self._metrics_env_ids_tensor = None
            return
        env_ids = [int(e) for e in env_ids]
        if not env_ids:
            self.eval_metrics_aggregator = None
            self._metrics_env_ids = []
            self._metrics_env_id_set = set()
            self._metrics_env_ids_tensor = None
            return
        self._metrics_env_ids = env_ids
        self._metrics_env_id_set = set(env_ids)
        self._metrics_env_ids_tensor = torch.tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        self.eval_metrics_aggregator = EvalMetricsAggregator(
            env_ids=env_ids,
            invalid_episode_max_steps=self.eval_metrics_invalid_max_steps,
            window_size=self.eval_metrics_window_size,
        )
        self._metrics_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._metrics_sum_spin_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_rot_radians = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_obj_angvel_norm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_sq_obj_angvel_norm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_obj_linvel_norm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_torque_mean_abs = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_torque_norm = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_spin_offaxis = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._metrics_sum_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        if self._last_terminated_by_fall is None:
            self._last_terminated_by_fall = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )

    def _log_eval_metrics_to_tb(self, metrics_summary, epoch):
        writer = self._periodic_eval_tb_writer
        if writer is None or not isinstance(metrics_summary, dict) or epoch is None:
            return
        for metric_name in (
            "fall_rate",
            "RotR",
            "Rotations_rad",
            "OffAxis_mean",
            "AngVel_var",
            "ObjVel_mean",
            "Torque_mean_abs",
            "EpisodeLen",
            "Return",
        ):
            value = metrics_summary.get(metric_name, None)
            if value is None:
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(scalar):
                continue
            writer.add_scalar(f"eval/{metric_name}", scalar, int(epoch))

    def _publish_eval_metrics_extras(self, summary=None):
        if self.eval_metrics_aggregator is None:
            return
        if summary is None:
            summary = self.eval_metrics_aggregator.last_summary()
        if not isinstance(summary, dict):
            return
        for metric_name, value in summary.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(scalar):
                continue
            self.extras[f"eval/{metric_name}"] = torch.tensor(
                scalar, device=self.device, dtype=torch.float32
            )

    def _reset_eval_metric_buffers(self, env_ids_t):
        if env_ids_t is None or env_ids_t.numel() == 0:
            return
        for buf in (
            self._metrics_steps,
            self._metrics_sum_spin_reward,
            self._metrics_sum_rot_radians,
            self._metrics_sum_obj_angvel_norm,
            self._metrics_sum_sq_obj_angvel_norm,
            self._metrics_sum_obj_linvel_norm,
            self._metrics_sum_torque_mean_abs,
            self._metrics_sum_torque_norm,
            self._metrics_sum_spin_offaxis,
            self._metrics_sum_reward,
        ):
            if buf is not None:
                buf[env_ids_t] = 0

    def _record_eval_metrics_episode_end(self, env_ids_t):
        if (
            self.eval_metrics_aggregator is None
            or env_ids_t is None
            or env_ids_t.numel() == 0
            or self._metrics_steps is None
        ):
            return
        env_ids_t = env_ids_t.to(device=self.device, dtype=torch.long).view(-1)
        env_ids_t = env_ids_t[(env_ids_t >= 0) & (env_ids_t < self.num_envs)]
        if env_ids_t.numel() == 0:
            return
        if self._test_eval_active and self._test_eval_state is not None:
            states = [
                self._test_eval_state.get(int(env_id), None) == "recording"
                for env_id in env_ids_t.detach().cpu().tolist()
            ]
            allowed_mask = torch.tensor(states, device=self.device, dtype=torch.bool)
            skipped_envs = env_ids_t[~allowed_mask]
            if skipped_envs.numel() > 0:
                self._reset_eval_metric_buffers(skipped_envs)
            env_ids_t = env_ids_t[allowed_mask]
            if env_ids_t.numel() == 0:
                return

        steps_t = self._metrics_steps[env_ids_t]
        non_empty = steps_t > 0
        if not bool(non_empty.any().item()):
            self._reset_eval_metric_buffers(env_ids_t)
            return

        valid_envs = env_ids_t[non_empty]
        steps = steps_t[non_empty].to(torch.float32)
        mean_angvel = self._metrics_sum_obj_angvel_norm[valid_envs] / steps
        angvel_var = torch.clamp(
            self._metrics_sum_sq_obj_angvel_norm[valid_envs] / steps - mean_angvel * mean_angvel,
            min=0.0,
        )
        if self._last_terminated_by_fall is not None:
            fall_flags = self._last_terminated_by_fall[valid_envs]
        else:
            fall_flags = torch.zeros_like(valid_envs, dtype=torch.bool, device=self.device)

        completed = False
        for i, env_id in enumerate(valid_envs.detach().cpu().tolist()):
            step_count = int(steps[i].item())
            completed = self.eval_metrics_aggregator.on_episode_end(
                env_id=env_id,
                steps=step_count,
                terminated_by_fall=bool(fall_flags[i].item()),
                rotr=(self._metrics_sum_spin_reward[valid_envs[i]] / steps[i]).item(),
                rotations_rad=self._metrics_sum_rot_radians[valid_envs[i]].item(),
                angvel_var=angvel_var[i].item(),
                objvel_mean=(self._metrics_sum_obj_linvel_norm[valid_envs[i]] / steps[i]).item(),
                torque_mean_abs=(self._metrics_sum_torque_mean_abs[valid_envs[i]] / steps[i]).item(),
                torque_norm=(self._metrics_sum_torque_norm[valid_envs[i]] / steps[i]).item(),
                offaxis_mean=(self._metrics_sum_spin_offaxis[valid_envs[i]] / steps[i]).item(),
                episode_return=self._metrics_sum_reward[valid_envs[i]].item(),
            ) or completed

        self._reset_eval_metric_buffers(env_ids_t)
        if completed:
            self._last_eval_metrics_summary = self.eval_metrics_aggregator.last_summary()
            self._publish_eval_metrics_extras(self._last_eval_metrics_summary)

    def _flush_eval_metrics(self, game_idx):
        if self.eval_metrics_aggregator is None:
            return
        out_dir = os.path.join(self.record_root_dir, self.eval_metrics_subdir)
        out_path = os.path.join(out_dir, f"summary_{int(game_idx):05d}.csv")
        summary = None
        self._last_eval_metrics_summary = None
        self._last_eval_metrics_epoch = None
        try:
            summary = self.eval_metrics_aggregator.flush(out_path)
            print(f"[eval-metrics] wrote {out_path}")
            if isinstance(summary, dict):
                epoch = self._periodic_eval_epoch
                if epoch is None or int(epoch) < 0:
                    epoch = int(game_idx)
                self._last_eval_metrics_summary = dict(summary)
                self._last_eval_metrics_epoch = int(epoch)
                self._log_eval_metrics_to_tb(summary, epoch)
        except Exception as exc:
            print(f"[eval-metrics] flush failed: {exc}")
        self.eval_metrics_aggregator = None
        self._metrics_env_ids = []
        self._metrics_env_id_set = set()

    def _record_eval_metrics_step(self):
        if self.eval_metrics_aggregator is None or self._metrics_env_ids_tensor is None:
            return
        if not hasattr(self, "last_spin_delta_axis") or self.last_spin_delta_axis is None:
            return
        if not hasattr(self, "last_spin_delta_offaxis") or self.last_spin_delta_offaxis is None:
            return
        if self._metrics_steps is None:
            return

        spin_delta_clipped = torch.clamp(
            self.last_spin_delta_axis,
            min=-6.28,
            max=float(self.reward_max_spin_rate),
        )
        spin_reward_per_env = float(self.spin_coef) * spin_delta_clipped

        obj_linvel_norm = torch.norm(self.object_linvel, dim=-1)
        obj_angvel_norm = torch.norm(self.object_angvel, dim=-1)

        torques = self.torques
        torque_mean_abs_per_env = torch.abs(torques).mean(dim=-1)
        torque_norm_per_env = torch.norm(torques, dim=-1)

        env_ids = self._metrics_env_ids_tensor
        self._metrics_steps[env_ids] += 1
        self._metrics_sum_spin_reward[env_ids] += spin_reward_per_env[env_ids]
        # spin_delta_axis comes from reward path with SPIN_DELTA_SCALE=20.
        self._metrics_sum_rot_radians[env_ids] += spin_delta_clipped[env_ids] / 20.0
        self._metrics_sum_spin_offaxis[env_ids] += self.last_spin_delta_offaxis[env_ids]
        self._metrics_sum_reward[env_ids] += self.rew_buf[env_ids]
        self._metrics_sum_obj_angvel_norm[env_ids] += obj_angvel_norm[env_ids]
        self._metrics_sum_sq_obj_angvel_norm[env_ids] += obj_angvel_norm[env_ids] * obj_angvel_norm[env_ids]
        self._metrics_sum_obj_linvel_norm[env_ids] += obj_linvel_norm[env_ids]
        self._metrics_sum_torque_mean_abs[env_ids] += torque_mean_abs_per_env[env_ids]
        self._metrics_sum_torque_norm[env_ids] += torque_norm_per_env[env_ids]
        self._publish_eval_metrics_extras()

    def _capture_terminated_by_fall(self):
        """Record per-env fall flag based on current pre-reset object pose.

        Called from post_physics_step() right after compute_reward(), before
        the next step's pre_physics_step performs reset_idx and overwrites
        object_pos with the new init pose.
        """
        if self._last_terminated_by_fall is None:
            return
        diff = self.object_pos - self.goal_pos
        goal_dist = torch.norm(diff, p=2, dim=-1)
        self._last_terminated_by_fall = goal_dist >= float(self.fall_dist)

    def _per_env_episode_dir(self, env_id, episode_idx):
        return self.recording_manager.per_env_episode_dir(env_id, episode_idx, self.record_root_dir)

    def _resolve_periodic_eval_artifact_paths_spin(self, env_id, episode_idx):
        ep_dir = self._per_env_episode_dir(env_id, episode_idx)
        return {
            "csv_path": os.path.join(ep_dir, "spin_trace.csv"),
            "png_path": os.path.join(ep_dir, "spin_trace.png"),
        }

    def _sync_recording_manager_hdf5_aliases(self):
        # Legacy attributes are kept as aliases so existing code paths and
        # debugging prints continue to work while the lifecycle lives in the manager.
        self._periodic_eval_hdf5_enabled = self.recording_manager.hdf5_enabled
        self._periodic_eval_hdf5_dir = self.recording_manager.hdf5_dir
        self._periodic_eval_hdf5_recorder = self.recording_manager.hdf5_recorder
        self._periodic_eval_hdf5_episode_idx = self.recording_manager.hdf5_episode_idx
        self._periodic_eval_hdf5_buffers = self.recording_manager.hdf5_buffers

    def _init_periodic_eval_hdf5(self, eval_plan):
        output_dir = str(eval_plan.get("replay_traj_output_dir", "")).strip()
        if not output_dir:
            output_dir = os.path.join(self.record_root_dir, "replay_traj")
        base_idx = int(eval_plan.get("record_game_idx_base", self.record_game_idx_base))
        self.recording_manager.init_hdf5(
            record_env_ids=self._periodic_eval_record_env_ids,
            output_dir=output_dir,
            base_idx=base_idx,
        )
        self._sync_recording_manager_hdf5_aliases()

    def _record_periodic_eval_hdf5_step(self):
        if self._periodic_eval_active:
            state_dict = self._periodic_eval_state
            record_envs = self._periodic_eval_record_env_ids
        elif self._test_eval_active:
            state_dict = self._test_eval_state
            record_envs = self._test_eval_record_env_ids
        else:
            return
        self.recording_manager.record_hdf5_step(state_dict=state_dict, record_env_ids=record_envs)

    def _flush_periodic_eval_hdf5_env(self, env_id, force=False):
        result = self.recording_manager.flush_hdf5_env(
            env_id=env_id,
            force=force,
            episode_dir_fn=self._per_env_episode_dir,
        )
        self._sync_recording_manager_hdf5_aliases()
        return result

    def _flush_all_periodic_eval_hdf5(self, force=False):
        self.recording_manager.flush_all_hdf5(force=force, episode_dir_fn=self._per_env_episode_dir)
        self._sync_recording_manager_hdf5_aliases()

    def close(self):
        if getattr(self, "_camera_task_closed", False):
            return
        self._camera_task_closed = True
        try:
            self._flush_all_periodic_eval_hdf5(force=True)
        except Exception:
            pass
        try:
            self._flush_wm_feature_recording()
        except Exception:
            pass
        try:
            self._close_all_recorders()
        except Exception:
            pass

        # Standalone play exits immediately after rollout. Release demo cameras
        # before interpreter teardown to reduce native shutdown segfault risk.
        try:
            self._destroy_demo_cameras()
        except Exception:
            pass

    def _periodic_eval_primary_recorder(self):
        if self.demo_recorder is not None:
            return self.demo_recorder
        return self.inference_recorder

    def _recording_allowed_for_env(self, env_id):
        return self.recording_manager.recording_allowed_for_env(
            env_id=env_id,
            periodic_active=self._periodic_eval_active,
            periodic_state=self._periodic_eval_state,
            test_active=self._test_eval_active,
            test_state=self._test_eval_state,
        )

    def _wm_feature_record_active(self):
        return (
            self.is_test_run
            and self.wm_feature_record_enabled
            and self._test_eval_active
            and self.wm_adapter is not None
        )

    def _init_wm_feature_recording(self):
        if not (self.is_test_run and self.wm_feature_record_enabled):
            return
        if self.wm_adapter is None:
            raise ValueError("task.env.wmFeatureRecord.enabled=True requires worldModel.enabled=True.")
        self._wm_feature_record_buffers = {
            "h_t": [],
            "env_id": [],
            "obj_name": [],
            "obj_id": [],
            "step": [],
            "episode_state": [],
            "camera_step": [],
        }
        self._wm_feature_record_flushed = False
        print(
            "[wm-feature-record] enabled "
            f"sample_interval={self.wm_feature_record_sample_interval} "
            f"dtype={self.wm_feature_record_dtype}"
        )

    def _resolve_wm_feature_record_output_path(self):
        if self.wm_feature_record_output_path:
            return self.wm_feature_record_output_path
        return os.path.join(
            str(self.record_root_dir),
            "wm_features",
            f"ht_features_{int(self.record_game_idx_base):05d}.npz",
        )

    def _record_wm_feature_snapshot(self, wm_feature):
        if not self._wm_feature_record_active():
            return
        if self.camera_step % self.wm_feature_record_sample_interval != 0:
            return
        if self._wm_feature_record_buffers is None:
            self._init_wm_feature_recording()
        recording_envs = [
            int(env_id)
            for env_id, state in self._test_eval_state.items()
            if state == "recording"
        ]
        if not recording_envs:
            return
        env_ids_np = np.asarray(recording_envs, dtype=np.int32)
        env_ids_t = torch.as_tensor(recording_envs, device=wm_feature.device, dtype=torch.long)
        h_np = wm_feature.detach()[env_ids_t].cpu().numpy()
        if self.wm_feature_record_dtype == "float16":
            h_np = h_np.astype(np.float16, copy=False)
        else:
            h_np = h_np.astype(np.float32, copy=False)
        obj_names = np.asarray(self._env_obj_names(recording_envs), dtype=object)
        obj_ids = np.asarray(
            [int(env_id) % len(self.used_training_objects) for env_id in recording_envs],
            dtype=np.int32,
        )
        states = np.asarray(
            [self._test_eval_state.get(int(env_id), "") for env_id in recording_envs],
            dtype=object,
        )
        steps = np.full((len(recording_envs),), int(self.camera_step), dtype=np.int32)

        buffers = self._wm_feature_record_buffers
        buffers["h_t"].append(h_np)
        buffers["env_id"].append(env_ids_np)
        buffers["obj_name"].append(obj_names)
        buffers["obj_id"].append(obj_ids)
        buffers["step"].append(steps)
        buffers["episode_state"].append(states)
        buffers["camera_step"].append(steps.copy())

    def _flush_wm_feature_recording(self):
        if not self.wm_feature_record_enabled or self._wm_feature_record_flushed:
            return
        self._wm_feature_record_flushed = True
        buffers = self._wm_feature_record_buffers
        if not buffers or not buffers.get("h_t"):
            return
        output_path = self._resolve_wm_feature_record_output_path()
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        h_t = np.concatenate(buffers["h_t"], axis=0)
        env_id = np.concatenate(buffers["env_id"], axis=0)
        obj_name = np.concatenate(buffers["obj_name"], axis=0).astype(str)
        obj_id = np.concatenate(buffers["obj_id"], axis=0)
        step = np.concatenate(buffers["step"], axis=0)
        episode_state = np.concatenate(buffers["episode_state"], axis=0).astype(str)
        camera_step = np.concatenate(buffers["camera_step"], axis=0)
        np.savez_compressed(
            output_path,
            h_t=h_t,
            env_id=env_id,
            obj_name=obj_name,
            obj_id=obj_id,
            step=step,
            episode_state=episode_state,
            camera_step=camera_step,
            sample_interval=np.asarray(self.wm_feature_record_sample_interval, dtype=np.int32),
            dtype=np.asarray(str(h_t.dtype)),
        )
        print(f"[wm-feature-record] wrote {output_path} rows={h_t.shape[0]} dim={h_t.shape[1]}")
        self._wm_feature_record_buffers = None

    def _enter_recording_state(self, env_id_int):
        self.recording_manager.enter_recording_state(env_id_int, self._periodic_eval_state)
        self._sync_recording_manager_hdf5_aliases()

    def _metrics_allowed_for_env(self, env_id):
        return int(env_id) in self._metrics_env_id_set

    def _enter_test_recording_state(self, env_id_int):
        self.recording_manager.enter_recording_state(env_id_int, self._test_eval_state)
        self._sync_recording_manager_hdf5_aliases()

    def _finish_test_eval(self):
        """Flush metrics + cleanly end the standalone test run.

        Only the standalone test lifecycle ever calls this (it is wired in as
        the on_complete hook for the test-eval state machine in
        on_episode_reset). The is_test_run check is defensive: if anything
        else ever invokes it during training we stop before flipping the
        SystemExit flag.
        """
        if not self.is_test_run:
            print("[test-eval] _finish_test_eval invoked outside test mode; ignoring.")
            self._test_eval_active = False
            return
        try:
            self._flush_eval_metrics(self.record_game_idx_base)
        except Exception as exc:
            print(f"[test-eval] metrics flush failed: {exc}")
        try:
            self._flush_wm_feature_recording()
        except Exception as exc:
            print(f"[test-eval] wm feature record flush failed: {exc}")
        self._test_eval_active = False
        self._test_run_pending_exit = True
        print("[test-eval] all envs finished one episode; exiting.")

    def _should_use_standalone_test_eval(self):
        """Hook for subclasses to opt out of the test-eval lifecycle.

        Default: enabled for the rotation task. Subclasses that have their
        own per-step pipeline (like the replay task) override this to return
        False so they keep using their legacy recording path.
        """
        return True

    def _init_standalone_test_eval(self):
        """Wire up recorders, metrics, hdf5, and spin trace for test mode.

        The player may issue an initial reset after task construction. Start
        envs as pending so that reset only arms recording; the next natural
        reset closes the measured episode, exports metrics, and exits.

        Layout follows the periodic-eval folder convention:
          <record_root>/<NNN_obj>/episode_<idx>/{demo.mp4, inference.mp4,
                                                 spin_trace.csv, spin_trace.png,
                                                 replay_traj.h5}
          <record_root>/metrics/summary_<idx>.csv
        """
        record_root = str(self.record_root_dir or "").strip()
        if not record_root:
            record_root = os.path.abspath("outputs/wm_craftnet_videos")
        os.makedirs(record_root, exist_ok=True)
        self.record_root_dir = record_root

        # Resolve the episode index used in the per-episode folder name. We
        # scan for existing episode_<idx> folders so re-runs against the same
        # output dir don't clobber prior artifacts.
        next_idx = scan_next_game_index(record_root, channel_name="demo")
        if self.record_game_idx_base > 0:
            next_idx = max(next_idx, int(self.record_game_idx_base))
        self.record_game_idx_base = int(next_idx)

        # Metrics envs cover all envs; artifact recording is one env per object.
        obj_num = max(1, int(len(self.used_training_objects)))
        record_count = min(int(self.num_envs), obj_num)
        self._test_eval_metrics_env_ids = list(range(int(self.num_envs)))
        self._test_eval_record_env_ids = list(range(record_count))
        self._test_eval_state = {
            env_id: "pending" for env_id in self._test_eval_metrics_env_ids
        }
        self._test_eval_active = True
        self._init_wm_feature_recording()

        # Recording surface: only record envs maintain video / hdf5 / spin
        # trace. Same _record_env_ids set used by _recording_allowed_for_env.
        self._record_env_ids = list(self._test_eval_record_env_ids)
        self._record_env_id_set = set(self._record_env_ids)

        if self.demo_recorder_enabled:
            self._init_demo_cameras(env_ids=self._test_eval_record_env_ids)
        if (
            self.demo_recorder_enabled
            or self.inference_recorder_enabled
            or self.wm_depth_video_record_enabled
        ):
            self._init_recording()

        # Reuse the periodic-eval hdf5 helper (it's a generic per-env episode
        # recorder; its only "periodic" tie was the buffer lifecycle which we
        # drive here with our own state machine).
        self.recording_manager.init_hdf5(
            record_env_ids=self._test_eval_record_env_ids,
            output_dir=os.path.join(record_root, "replay_traj"),
            base_idx=self.record_game_idx_base,
        )
        self._sync_recording_manager_hdf5_aliases()

        # Metrics aggregator over all 1024 envs.
        self._init_eval_metrics(self._test_eval_metrics_env_ids)

        # Spin trace: each record env gets its own (csv, png), routed into
        # the same per-(env, episode) folder as the demo video.
        self.spin_trace_enabled = bool(getattr(self, "spin_trace_cfg_enabled", False))
        self.spin_trace_env_ids = list(self._test_eval_record_env_ids)
        if self._test_eval_record_env_ids:
            self.spin_trace_env_id = int(self._test_eval_record_env_ids[0])
        self.spin_trace_episode_idx = int(self.record_game_idx_base)
        self._spin_trace_rows_per_env = {
            env_id: [] for env_id in self._test_eval_record_env_ids
        }
        self._spin_trace_episode_idx_per_env = {
            env_id: int(self.record_game_idx_base)
            for env_id in self._test_eval_record_env_ids
        }
        if self.spin_trace_env_id in self._spin_trace_rows_per_env:
            self._spin_trace_rows = self._spin_trace_rows_per_env[self.spin_trace_env_id]
        else:
            self._spin_trace_rows = []
        self._spin_trace_output_path_resolver = self._resolve_periodic_eval_artifact_paths_spin

        print(
            f"[test-eval] start metrics_envs={len(self._test_eval_metrics_env_ids)} "
            f"record_envs={self._test_eval_record_env_ids} "
            f"episode_base_idx={self.record_game_idx_base} "
            f"output_root={record_root}"
        )

    def _start_periodic_eval(self, eval_plan):
        axis_now = str(self.cfg["env"].get("axis", ""))
        obj_set_now = str(self.cfg["env"].get("objSet", ""))
        axis_target = str(eval_plan.get("axis", axis_now))
        obj_set_target = str(eval_plan.get("obj_set", obj_set_now))
        if axis_now != axis_target or obj_set_now != obj_set_target:
            print(
                f"[periodic-eval] axis/obj_set frozen=({axis_target},{obj_set_target}) "
                f"!= runtime=({axis_now},{obj_set_now}); continue with runtime env."
            )

        obj_num = max(1, int(len(self.used_training_objects)))
        selected_env_count = min(int(self.num_envs), obj_num)
        self._periodic_eval_record_env_ids = list(range(selected_env_count))
        # Periodic eval only owns artifact lifecycles. Rolling metrics are
        # collected continuously during training, so the eval state machine
        # should only wait for envs that actually record video/h5/trace files.
        self._periodic_eval_env_ids = list(self._periodic_eval_record_env_ids)
        self._periodic_eval_state = {
            env_id: "pending" for env_id in self._periodic_eval_env_ids
        }
        self._periodic_eval_active = True
        self._periodic_eval_epoch = int(eval_plan.get("epoch", -1))
        self._periodic_eval_tb_writer = eval_plan.get("tb_writer", None)
        self._periodic_eval_prev_is_test_run = bool(self.is_test_run)
        self._periodic_eval_steps = 0
        max_steps_cfg = int(getattr(self, "periodic_eval_max_steps_cfg", 0) or 0)
        if max_steps_cfg > 0:
            self._periodic_eval_max_steps = max_steps_cfg
        else:
            self._periodic_eval_max_steps = int(getattr(self, "max_episode_length", 500)) * 2 + 16
        self.is_test_run = True
        self.wm_depth_video_record_enabled = self.wm_depth_video_record_cfg_enabled

        self.record_root_dir = str(eval_plan.get("record_root_dir", self.record_root_dir))
        self.record_game_idx_base = int(eval_plan.get("record_game_idx_base", self.record_game_idx_base))
        self._record_env_ids = list(self._periodic_eval_record_env_ids)
        self._record_env_id_set = set(self._record_env_ids)

        self.demo_recorder_enabled = bool(self.demo_recorder_cfg_enabled)
        self.inference_recorder_enabled = bool(self.inference_recorder_cfg_enabled)

        if self.demo_recorder_enabled:
            self._init_demo_cameras(env_ids=self._periodic_eval_record_env_ids)

        self._close_all_recorders()
        if (
            self.demo_recorder_enabled
            or self.inference_recorder_enabled
            or self.wm_depth_video_record_enabled
        ):
            self._init_recording()
        self._init_periodic_eval_hdf5(eval_plan)

        self._periodic_eval_prev_spin_trace = {
            "enabled": bool(self.spin_trace_enabled),
            "env_id": int(self.spin_trace_env_id),
            "env_ids": list(self.spin_trace_env_ids),
            "output_dir": str(self.spin_trace_output_dir),
            "episode_idx": int(self.spin_trace_episode_idx),
            "episode_idx_per_env": dict(self._spin_trace_episode_idx_per_env),
            "rows_per_env": dict(self._spin_trace_rows_per_env),
            "resolver": self._spin_trace_output_path_resolver,
        }
        # Keep periodic-eval spin trace gated by config spinTrace.enabled.
        self.spin_trace_enabled = bool(getattr(self, "spin_trace_cfg_enabled", self.spin_trace_enabled))
        # spin_trace_env_ids is empty at eval start: each record env is added
        # only when it transitions pending->recording so rows from the
        # pre-eval episode never bleed into the eval csv.
        self.spin_trace_env_ids = []
        if self._periodic_eval_record_env_ids:
            self.spin_trace_env_id = int(self._periodic_eval_record_env_ids[0])
        self.spin_trace_output_dir = str(
            eval_plan.get("spin_trace_output_dir", self.spin_trace_output_dir)
        )
        # All per-episode artifacts share the same starting index = current
        # record game index (= epoch in periodic eval / next test counter).
        starting_episode_idx = int(self.record_game_idx_base)
        self.spin_trace_episode_idx = starting_episode_idx
        self._spin_trace_rows_per_env = {
            env_id: [] for env_id in self._periodic_eval_record_env_ids
        }
        self._spin_trace_episode_idx_per_env = {
            env_id: starting_episode_idx for env_id in self._periodic_eval_record_env_ids
        }
        # Legacy alias kept consistent so existing _spin_trace_rows readers
        # don't crash when there are no record envs (e.g. obj_num=0 corner case).
        if self.spin_trace_env_id in self._spin_trace_rows_per_env:
            self._spin_trace_rows = self._spin_trace_rows_per_env[self.spin_trace_env_id]
        else:
            self._spin_trace_rows = []
        # Route per-env spin trace into the same per-episode subfolder as the
        # demo / inference videos. spin_trace_output_dir itself is never
        # written to anymore (resolver handles all path generation), so we
        # do not eagerly create it.
        self._spin_trace_output_path_resolver = self._resolve_periodic_eval_artifact_paths_spin

        print(
            f"[periodic-eval] start epoch={int(eval_plan.get('epoch', -1))} "
            f"metrics_envs={len(self._periodic_eval_env_ids)} "
            f"record_envs={self._periodic_eval_record_env_ids} "
            f"max_steps={self._periodic_eval_max_steps}"
        )

    def _stop_periodic_eval(self):
        self._flush_all_periodic_eval_hdf5(force=True)
        self.recording_manager.hdf5_enabled = False
        self.recording_manager.hdf5_dir = ""
        self.recording_manager.hdf5_recorder = None
        self.recording_manager.hdf5_episode_idx = {}
        self.recording_manager.hdf5_buffers = {}
        self._sync_recording_manager_hdf5_aliases()
        self._periodic_eval_active = False
        self._periodic_eval_state = {}
        self._periodic_eval_env_ids = []
        self._periodic_eval_record_env_ids = []
        self._periodic_eval_epoch = -1
        self._periodic_eval_tb_writer = None
        self._periodic_eval_steps = 0
        self._periodic_eval_max_steps = 0
        self.is_test_run = bool(self._periodic_eval_prev_is_test_run)
        self.wm_depth_video_record_enabled = (
            self.is_test_run and self.wm_depth_video_record_cfg_enabled
        )
        self._periodic_eval_prev_is_test_run = False

        self._close_all_recorders()
        self._destroy_demo_cameras()
        self.demo_recorder_enabled = False
        self.inference_recorder_enabled = False
        self._record_env_ids = []
        self._record_env_id_set = set()

        if isinstance(self._periodic_eval_prev_spin_trace, dict):
            prev = self._periodic_eval_prev_spin_trace
            self.spin_trace_enabled = bool(prev.get("enabled", False))
            self.spin_trace_env_id = int(prev.get("env_id", 0))
            self.spin_trace_env_ids = list(prev.get("env_ids", [self.spin_trace_env_id]))
            self.spin_trace_output_dir = str(prev.get("output_dir", self.spin_trace_output_dir))
            self.spin_trace_episode_idx = int(prev.get("episode_idx", 0))
            self._spin_trace_episode_idx_per_env = dict(prev.get("episode_idx_per_env", {}))
            self._spin_trace_rows_per_env = dict(prev.get("rows_per_env", {}))
            if self.spin_trace_env_id in self._spin_trace_rows_per_env:
                self._spin_trace_rows = self._spin_trace_rows_per_env[self.spin_trace_env_id]
            else:
                self._spin_trace_rows = []
            self._spin_trace_output_path_resolver = prev.get("resolver", None)
        self._periodic_eval_prev_spin_trace = None

        print("[periodic-eval] finished and resumed training mode")

    def run_periodic_eval(self, eval_plan):
        if not (
            self.camera_policy_enabled
            or self.demo_recorder_cfg_enabled
            or self.inference_recorder_cfg_enabled
        ):
            print("[periodic-eval] skip: all camera pipelines disabled")
            return
        if self._periodic_eval_active:
            print("[periodic-eval] skip: previous eval still running")
            return
        self._start_periodic_eval(eval_plan)

    # ------------------------------------------------------------------
    # Demo camera setup
    # ------------------------------------------------------------------

    def _init_demo_cameras(self, env_ids=None):
        self._destroy_demo_cameras()
        if len(self.demo_camera_pos) != 3:
            raise ValueError(
                f"cameraDemo.camera.pos must be length-3, got {self.demo_camera_pos}"
            )
        if not self.demo_camera_lookat_enabled and len(self.demo_camera_rot) != 3:
            raise ValueError(
                f"cameraDemo.camera.rot must be length-3 [z,y,x], got {self.demo_camera_rot}"
            )
        if self.demo_camera_lookat_enabled and len(self.demo_camera_lookat_target_pos) != 3:
            raise ValueError(
                "cameraDemo.camera.lookat.target_pos must be length-3, "
                f"got {self.demo_camera_lookat_target_pos}"
            )

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.demo_camera_width
        camera_props.height = self.demo_camera_height
        camera_props.horizontal_fov = self.demo_camera_fov
        # Demo camera is read via CPU `get_camera_image` (see _get_demo_rgb_frame
        # for the reason). Leaving enable_tensors=False also avoids allocating
        # large GPU framebuffers for each high-res demo cam.
        camera_props.enable_tensors = False
        total_envs = len(self.envs)
        if env_ids is None:
            target_env_ids = list(range(total_envs))
        else:
            target_env_ids = sorted({int(env_id) for env_id in env_ids if 0 <= int(env_id) < total_envs})
        # Keep index-by-env_id contract for downstream methods.
        self.demo_camera_handles = [None] * total_envs
        self._demo_cam_local_pos_per_env = [None] * total_envs
        # Camera matrices cache must align with handle slots.
        self._axes_view_proj_cache = [None] * total_envs

        for env_id in target_env_ids:
            camera_handle = self.gym.create_camera_sensor(self.envs[env_id], camera_props)
            if camera_handle < 0:
                raise RuntimeError(f"Failed to create demo camera for env {env_id}")
            self.demo_camera_handles[env_id] = camera_handle
            cam_pos = gymapi.Vec3(*self.demo_camera_pos)
            if self.demo_camera_pose_noise_std > 0.0:
                cam_pos.x += self.demo_camera_pose_noise_std * np.random.normal()
                cam_pos.y += self.demo_camera_pose_noise_std * np.random.normal()
                cam_pos.z += self.demo_camera_pose_noise_std * np.random.normal()
            self._demo_cam_local_pos_per_env[env_id] = (
                np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32)
            )
            if self.demo_camera_lookat_enabled:
                target_pos = gymapi.Vec3(*self.demo_camera_lookat_target_pos)
                self.gym.set_camera_location(
                    camera_handle, self.envs[env_id], cam_pos, target_pos
                )
            else:
                cam_quat = gymapi.Quat.from_euler_zyx(*np.radians(self.demo_camera_rot))
                self.gym.set_camera_transform(
                    camera_handle,
                    self.envs[env_id],
                    gymapi.Transform(cam_pos, cam_quat),
                )

    def _get_demo_rgb_frame(self, env_id):
        """Read one RGB frame from the demo camera via CPU path.

        Isaac Gym Preview 4 has a driver-level bug: when multiple cameras with
        different resolutions are attached to the same env, the GPU tensor path
        (`get_camera_image_gpu_tensor`) returns the wrong tensor (it aliases to
        the first camera created per env, i.e. the inference 96x72 one). The
        CPU path (`get_camera_image`) returns the correct resolution. Since the
        demo camera is intentionally high-resolution for offline showcase, we
        eat the CPU copy cost here; the inference camera is unaffected because
        base VecTask still uses GPU tensors for it.
        """
        camera_handle = self.demo_camera_handles[env_id]
        if camera_handle is None:
            raise RuntimeError(f"Demo camera is not initialized for env {env_id}")
        cpu_img = self.gym.get_camera_image(
            self.sim, self.envs[env_id], camera_handle, gymapi.IMAGE_COLOR
        )
        return self._format_cpu_camera_rgb(
            cpu_img, self.demo_camera_height, self.demo_camera_width, env_id
        )

    def _format_cpu_camera_rgb(self, image, expected_h, expected_w, env_id):
        """Normalise the CPU `get_camera_image` IMAGE_COLOR output to H x W x 3 uint8.

        Isaac Gym's `get_camera_image(..., IMAGE_COLOR)` returns RGBA uint8 in one of:
          - flat uint8 buffer of length H * W * 4
          - (H, W * 4) uint8 row-major
          - (H, W, 4) already-shaped RGBA
        We just reshape to (H, W, 4) and slice off the alpha channel. No heuristic
        detection (the previous std/mean scan on 4K frames dominated frame time).
        """
        if image is None:
            raise RuntimeError(f"Demo CPU camera image is None for env {env_id}.")
        arr = np.asarray(image)
        h = int(expected_h)
        w = int(expected_w)

        if arr.ndim == 1 and arr.size == h * w * 4:
            arr = arr.reshape((h, w, 4))
        elif arr.ndim == 2 and arr.shape == (h, w * 4):
            arr = arr.reshape((h, w, 4))
        if arr.ndim != 3 or arr.shape[0] != h or arr.shape[1] != w or arr.shape[2] not in (3, 4):
            raise RuntimeError(
                f"Demo CPU camera image has unexpected shape {arr.shape} for env {env_id}; "
                f"expected ({h}, {w}, 3|4)."
            )

        rgb = arr[..., :3] if arr.shape[2] == 4 else arr
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Axes overlay on demo frames (world->pixel projection, no sim actors).
    # ------------------------------------------------------------------

    def _get_or_build_demo_cam_matrices(self, env_id):
        """Return cached (view_4x4, proj_4x4) numpy matrices for env's demo cam.

        Demo cameras are pinned to the env transform and envs don't move, so
        these matrices are constant for the whole run. Compute once per env.
        """
        if self._axes_view_proj_cache is None:
            self._axes_view_proj_cache = [None] * len(self.demo_camera_handles)
        cached = self._axes_view_proj_cache[env_id]
        if cached is not None:
            return cached
        env = self.envs[env_id]
        cam = self.demo_camera_handles[env_id]
        view = np.asarray(
            self.gym.get_camera_view_matrix(self.sim, env, cam), dtype=np.float32
        ).reshape(4, 4)
        proj = np.asarray(
            self.gym.get_camera_proj_matrix(self.sim, env, cam), dtype=np.float32
        ).reshape(4, 4)
        self._axes_view_proj_cache[env_id] = (view, proj)
        return view, proj

    def _grid_env_origin(self, env_id):
        """Deterministic env origin from create_env grid layout."""
        spacing = float(self.cfg["env"].get("envSpacing", 1.0))
        num_per_row = max(1, int(math.sqrt(self.num_envs)))
        row = env_id // num_per_row
        col = env_id % num_per_row
        return np.array([col * 2.0 * spacing, row * 2.0 * spacing, 0.0], dtype=np.float32)

    def _estimate_env_origin_from_hand(self, env_id):
        """Estimate env origin from fixed-base hand root state."""
        try:
            hand_sim_idx = int(self.hand_indices[env_id].item())
            hand_pos = (
                self.root_state_tensor[hand_sim_idx, 0:3]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            return hand_pos - self._hand_local_anchor_pos
        except Exception:
            # Fallback to deterministic layout if hand indices are unavailable.
            return self._grid_env_origin(env_id)

    def _project_env_local_points_with_demo_lookat(self, pts_env_local, env_id):
        """Project env-local points using demo look-at camera params (no Gym matrices)."""
        pts = np.asarray(pts_env_local, dtype=np.float32).reshape(-1, 3)
        if 0 <= env_id < len(self._demo_cam_local_pos_per_env):
            cam_pos = self._demo_cam_local_pos_per_env[env_id].astype(np.float32)
        else:
            cam_pos = np.asarray(self.demo_camera_pos, dtype=np.float32)
        target = np.asarray(self.demo_camera_lookat_target_pos, dtype=np.float32)
        forward = target - cam_pos
        fwd_norm = float(np.linalg.norm(forward))
        if fwd_norm < 1e-8:
            return np.zeros((pts.shape[0], 2), dtype=np.int32), np.zeros((pts.shape[0],), dtype=bool)
        forward /= fwd_norm

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(forward, world_up))) > 0.98:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right_norm = float(np.linalg.norm(right))
        if right_norm < 1e-8:
            return np.zeros((pts.shape[0], 2), dtype=np.int32), np.zeros((pts.shape[0],), dtype=bool)
        right /= right_norm
        up = np.cross(right, forward)

        rel = pts - cam_pos[None, :]
        x_cam = rel @ right
        y_cam = rel @ up
        z_cam = rel @ forward
        valid = z_cam > 1e-6
        safe_z = np.where(valid, z_cam, np.float32(1.0))

        hfov = math.radians(float(self.demo_camera_fov))
        fx = (0.5 * float(self.demo_camera_width)) / max(1e-8, math.tan(0.5 * hfov))
        vfov = 2.0 * math.atan(
            math.tan(0.5 * hfov) * (float(self.demo_camera_height) / float(self.demo_camera_width))
        )
        fy = (0.5 * float(self.demo_camera_height)) / max(1e-8, math.tan(0.5 * vfov))

        px = 0.5 * float(self.demo_camera_width) + (x_cam * fx / safe_z)
        py = 0.5 * float(self.demo_camera_height) - (y_cam * fy / safe_z)
        pix = np.stack([px, py], axis=1).astype(np.int32)
        return pix, valid

    @staticmethod
    def _project_world_points_to_pixels(pts_world, view, proj, width, height):
        """Project world-space points to pixel coords (Isaac Gym row-vector convention).

        Isaac Gym returns view/proj such that
            p_clip = [x, y, z, 1] @ view @ proj
        Front-facing points have clip.w > 0.
        Returns (pixels[N, 2] int32, valid[N] bool) where valid means
        "in front of the camera".
        """
        pts_world = np.asarray(pts_world, dtype=np.float32).reshape(-1, 3)
        n = pts_world.shape[0]
        homog = np.concatenate([pts_world, np.ones((n, 1), dtype=np.float32)], axis=1)
        clip = homog @ view @ proj  # [N, 4]
        w_col = clip[:, 3]
        valid = w_col > 1e-6
        # Avoid div-by-zero; invalid rows get placeholder 1.0 and are masked later.
        safe_w = np.where(valid, w_col, np.float32(1.0))
        ndc_x = clip[:, 0] / safe_w
        ndc_y = clip[:, 1] / safe_w
        px = (ndc_x * 0.5 + 0.5) * float(width)
        py = (1.0 - (ndc_y * 0.5 + 0.5)) * float(height)
        pix = np.stack([px, py], axis=1).astype(np.int32)
        return pix, valid

    @staticmethod
    def _axes_quat_to_rotmat(q_xyzw):
        """Convert quaternion (x, y, z, w) into a 3x3 rotation matrix (column-vector)."""
        x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
        n = x * x + y * y + z * z + w * w
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
        s = 2.0 / n
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        return np.array([
            [1.0 - (yy + zz), xy - wz,         xz + wy        ],
            [xy + wz,         1.0 - (xx + zz), yz - wx        ],
            [xz - wy,         yz + wx,         1.0 - (xx + yy)],
        ], dtype=np.float32)

    def _draw_object_axes_on_frame(self, frame, env_id):
        """Overlay body-frame RGB axes onto a demo RGB frame.

        Uses the *current* `self.object_pos` / `self.object_rot` (refreshed in
        `post_physics_step` before camera fetch), so the axes are perfectly
        rigidly attached to the object regardless of rotation speed.
        """
        if not self.show_object_axes or not self._axes_drawer_available:
            return frame
        if env_id >= len(self.demo_camera_handles):
            return frame
        if not self._axes_startup_logged:
            proj_mode = "manual_lookat" if self.demo_camera_lookat_enabled else "matrix_world"
            print(f"[axes-info] overlay enabled mode={proj_mode} envs={self.num_envs}")
            self._axes_startup_logged = True

        # Object pose in world frame. Read straight from root_state_tensor via
        # object_indices — those are always valid after env construction,
        # whereas `self.object_pos` / `self.object_rot` are only populated by
        # compute_observations and may not exist yet on the very first frame
        # (reset -> fetch_camera_observations is called before any step).
        root = self.root_state_tensor
        obj_idx_entry = self.object_indices[env_id]
        if isinstance(obj_idx_entry, torch.Tensor):
            obj_sim_indices = obj_idx_entry.detach().view(-1).to(torch.long).cpu().tolist()
        elif isinstance(obj_idx_entry, (list, tuple, np.ndarray)):
            obj_sim_indices = [int(i) for i in obj_idx_entry]
        else:
            obj_sim_indices = [int(obj_idx_entry)]
        if len(obj_sim_indices) == 0:
            return frame

        from PIL import Image, ImageDraw
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        thickness = max(
            1,
            int(round(self.object_axes_line_thickness_frac
                      * max(self.demo_camera_height, self.demo_camera_width))),
        )
        arrow_half_base = max(
            1.0,
            self.object_axes_arrow_half_base_frac
            * float(max(self.demo_camera_height, self.demo_camera_width)),
        )
        # Fixed half-apex-angle = 15 deg. Arrow size depends only on half-base.
        arrow_head_len = arrow_half_base / math.tan(math.radians(15.0))
        colors = [(232, 40, 40), (40, 210, 40), (40, 90, 235)]  # X=red, Y=green, Z=blue
        L = float(self.object_axes_length)
        # Local axis tips expressed as row vectors; apply R.T on the right to
        # rotate row vectors, i.e. world_tip = obj_pos + axis_local @ R.T.
        axes_local = np.array(
            [[L, 0.0, 0.0], [0.0, L, 0.0], [0.0, 0.0, L]], dtype=np.float32
        )

        for obj_sim_idx in obj_sim_indices:
            obj_pos = root[obj_sim_idx, 0:3].detach().cpu().numpy().astype(np.float32)
            obj_rot = root[obj_sim_idx, 3:7].detach().cpu().numpy().astype(np.float32)
            R = self._axes_quat_to_rotmat(obj_rot)  # rotates column vectors
            tips_world = obj_pos[None, :] + axes_local @ R.T
            pts = np.concatenate([obj_pos[None, :], tips_world], axis=0)  # [4, 3]
            env_origin = self._estimate_env_origin_from_hand(env_id)
            pts_env_local = pts - env_origin[None, :]
            if self.demo_camera_lookat_enabled:
                pix, valid = self._project_env_local_points_with_demo_lookat(pts_env_local, env_id)
            else:
                view, proj = self._get_or_build_demo_cam_matrices(env_id)
                pix_world, valid_world = self._project_world_points_to_pixels(
                    pts, view, proj, self.demo_camera_width, self.demo_camera_height
                )
                pix = pix_world
                valid = valid_world
            center_in_frame = (
                0 <= int(pix[0, 0]) < self.demo_camera_width
                and 0 <= int(pix[0, 1]) < self.demo_camera_height
            )
            # Draw only when center is in front of camera and inside image bounds.
            if (not bool(valid[0])) or (not center_in_frame):
                continue

            cx, cy = int(pix[0, 0]), int(pix[0, 1])
            for k in range(3):
                if not bool(valid[1 + k]):
                    continue
                tx_f, ty_f = float(pix[1 + k, 0]), float(pix[1 + k, 1])
                dx = tx_f - cx
                dy = ty_f - cy
                line_len = math.hypot(dx, dy)
                if line_len < 1.0:
                    continue
                head_len = arrow_head_len
                ux = dx / line_len
                uy = dy / line_len
                base_x = tx_f - head_len * ux
                base_y = ty_f - head_len * uy
                sx = int(round(base_x))
                sy = int(round(base_y))
                draw.line([(cx, cy), (sx, sy)], fill=colors[k], width=thickness)
                # Fixed-arrow geometry: half-apex-angle=15 deg, half-base configured directly.
                perp_x = -uy
                perp_y = ux
                half_base = arrow_half_base
                b1 = (base_x + half_base * perp_x, base_y + half_base * perp_y)
                b2 = (base_x - half_base * perp_x, base_y - half_base * perp_y)
                draw.polygon([(tx_f, ty_f), b1, b2], fill=colors[k])
            # Small white disc at each object origin for a clean anchor on top.
            r = max(1, int(round(thickness * 1.2)))
            draw.ellipse(
                [(cx - r, cy - r), (cx + r, cy + r)], fill=(255, 255, 255)
            )
        return np.array(img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Obs / recording pipeline
    # ------------------------------------------------------------------

    def _ensure_obs_container(self):
        if "obs" not in self.obs_dict:
            self.obs_dict["obs"] = {}
        elif not isinstance(self.obs_dict["obs"], dict):
            self.obs_dict["obs"] = {"obs": self.obs_dict["obs"]}

    def _normalize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        return torch.clamp(depth, min=0.0, max=self.max_depth) / self.max_depth

    @staticmethod
    def _safe_crop_bounds(
        height: int,
        width: int,
        top_px: int,
        bottom_px: int,
        left_px: int,
        right_px: int,
    ):
        top = max(0, int(top_px))
        bottom = max(0, int(bottom_px))
        left = max(0, int(left_px))
        right = max(0, int(right_px))
        y0 = min(max(0, top), max(0, height - 1))
        y1 = max(y0 + 1, min(height, height - max(0, bottom)))
        x0 = min(max(0, left), max(0, width - 1))
        x1 = max(x0 + 1, min(width, width - max(0, right)))
        return y0, y1, x0, x1

    @staticmethod
    def _build_gaussian_kernel(kernel_size: int, sigma: float, device, dtype):
        if kernel_size <= 1:
            return None
        sigma = max(float(sigma), 1e-6)
        coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
        gauss_1d = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        gauss_1d = gauss_1d / torch.clamp(gauss_1d.sum(), min=1e-8)
        kernel_2d = torch.outer(gauss_1d, gauss_1d)
        kernel_2d = kernel_2d / torch.clamp(kernel_2d.sum(), min=1e-8)
        return kernel_2d.view(1, 1, kernel_size, kernel_size)

    def _apply_depth_crop(self, depth_norm: torch.Tensor) -> torch.Tensor:
        """Crop depth image. Result has shape (depth_obs_height, depth_obs_width)."""
        if not self.depth_crop_enabled:
            return depth_norm

        _, h, w, _ = depth_norm.shape
        y0, y1, x0, x1 = self._safe_crop_bounds(
            h,
            w,
            self.depth_crop_top_px,
            self.depth_crop_bottom_px,
            self.depth_crop_left_px,
            self.depth_crop_right_px,
        )
        depth_norm = depth_norm[:, y0:y1, x0:x1, :]

        if (
            depth_norm.shape[1] != self.depth_obs_height
            or depth_norm.shape[2] != self.depth_obs_width
        ):
            raise RuntimeError(
                "Depth crop output shape mismatch: "
                f"got ({depth_norm.shape[1]}, {depth_norm.shape[2]}), "
                f"expected ({self.depth_obs_height}, {self.depth_obs_width})."
            )
        return depth_norm

    def _apply_depth_noise(self, depth_norm: torch.Tensor) -> torch.Tensor:
        """Apply all noise augmentations after crop. Does not change spatial dimensions."""
        if not self.depth_noise_enabled:
            return depth_norm

        depth_norm = self._invalidate_depth_discontinuities(depth_norm)

        if self.depth_blur_enabled and self.depth_blur_kernel > 1:
            blur_kernel = self._build_gaussian_kernel(
                self.depth_blur_kernel,
                self.depth_blur_sigma,
                depth_norm.device,
                depth_norm.dtype,
            )
            if blur_kernel is not None:
                pad = self.depth_blur_kernel // 2
                depth_chw = depth_norm.permute(0, 3, 1, 2).contiguous()
                depth_chw = F.pad(depth_chw, (pad, pad, pad, pad), mode="replicate")
                depth_chw = F.conv2d(depth_chw, blur_kernel)
                depth_norm = depth_chw.permute(0, 2, 3, 1).contiguous()

        if self.depth_gaussian_noise_enabled and self.depth_gaussian_noise_std > 0.0:
            depth_norm = torch.clamp(
                depth_norm
                + torch.randn_like(depth_norm) * self.depth_gaussian_noise_std,
                0.0,
                1.0,
            )

        if self.depth_random_rotation_enabled and self.depth_random_rotation_deg > 0.0:
            depth_norm = self._apply_depth_random_rotation(depth_norm)

        depth_norm = self._add_temporal_depth_noise(depth_norm)

        return depth_norm

    def _invalidate_depth_discontinuities(self, depth_norm: torch.Tensor) -> torch.Tensor:
        if not self.depth_discontinuity_invalidate_enabled:
            return depth_norm
        if depth_norm.ndim != 4 or depth_norm.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected depth shape for discontinuity invalidation: {tuple(depth_norm.shape)}"
            )

        depth_hw = depth_norm[..., 0]
        valid = torch.isfinite(depth_hw) & (depth_hw < 1.0)
        if depth_hw.shape[1] <= 1 and depth_hw.shape[2] <= 1:
            return depth_norm

        thresh_norm = self.depth_discontinuity_thresh_m / max(self.max_depth, 1e-8)
        invalidate = torch.zeros_like(valid, dtype=torch.bool)

        if depth_hw.shape[2] > 1:
            diff_x = torch.abs(depth_hw[:, :, 1:] - depth_hw[:, :, :-1])
            jump_x = (diff_x > thresh_norm) & valid[:, :, 1:] & valid[:, :, :-1]
            invalidate[:, :, 1:] |= jump_x
            invalidate[:, :, :-1] |= jump_x

        if depth_hw.shape[1] > 1:
            diff_y = torch.abs(depth_hw[:, 1:, :] - depth_hw[:, :-1, :])
            jump_y = (diff_y > thresh_norm) & valid[:, 1:, :] & valid[:, :-1, :]
            invalidate[:, 1:, :] |= jump_y
            invalidate[:, :-1, :] |= jump_y

        depth_hw = torch.where(invalidate, torch.ones_like(depth_hw), depth_hw)
        return depth_hw.unsqueeze(-1)

    def _apply_depth_random_rotation(self, depth_norm: torch.Tensor) -> torch.Tensor:
        if depth_norm.ndim != 4 or depth_norm.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected depth shape for random rotation: {tuple(depth_norm.shape)}"
            )
        num_envs, depth_h, depth_w, _ = depth_norm.shape
        if num_envs == 0:
            return depth_norm

        max_rad = math.radians(self.depth_random_rotation_deg)
        angles = (torch.rand((num_envs,), device=depth_norm.device) * 2.0 - 1.0) * max_rad
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)

        theta = torch.zeros((num_envs, 2, 3), device=depth_norm.device, dtype=depth_norm.dtype)
        theta[:, 0, 0] = cos_theta
        theta[:, 0, 1] = -sin_theta
        theta[:, 1, 0] = sin_theta
        theta[:, 1, 1] = cos_theta

        depth_chw = depth_norm.permute(0, 3, 1, 2).contiguous()
        grid = F.affine_grid(theta, size=(num_envs, 1, depth_h, depth_w), align_corners=False)
        rotated = F.grid_sample(
            depth_chw,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return rotated.permute(0, 2, 3, 1).contiguous()

    def _add_temporal_depth_noise(self, depth_norm: torch.Tensor) -> torch.Tensor:
        if not self.temporal_noise_enabled:
            return depth_norm
        if depth_norm.ndim != 4 or depth_norm.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected depth shape for temporal noise: {tuple(depth_norm.shape)}"
            )
        num_envs, depth_h, depth_w, _ = depth_norm.shape
        if (
            self.persistent_noise_mask is None
            or self.persistent_noise_mask.shape != (num_envs, depth_h, depth_w)
            or self.persistent_noise_mask.device != depth_norm.device
        ):
            self.persistent_noise_mask = torch.zeros(
                (num_envs, depth_h, depth_w),
                dtype=torch.bool,
                device=depth_norm.device,
            )
        new_random_mask = (
            torch.rand((num_envs, depth_h, depth_w), device=depth_norm.device)
            < self.temporal_noise_p_dropout
        )
        keep_state_mask = (
            torch.rand((num_envs, depth_h, depth_w), device=depth_norm.device)
            < self.temporal_noise_corr
        )
        self.persistent_noise_mask = torch.where(
            keep_state_mask, self.persistent_noise_mask, new_random_mask
        )
        noise_mask = self.persistent_noise_mask.unsqueeze(-1)
        return torch.where(noise_mask, torch.ones_like(depth_norm), depth_norm)

    def _apply_rgb_crop_if_needed(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        if not self.depth_crop_enabled or self.camera_mode != "rgbd":
            return rgb_obs
        _, h, w, _ = rgb_obs.shape
        y0, y1, x0, x1 = self._safe_crop_bounds(
            h,
            w,
            self.depth_crop_top_px,
            self.depth_crop_bottom_px,
            self.depth_crop_left_px,
            self.depth_crop_right_px,
        )
        rgb_obs = rgb_obs[:, y0:y1, x0:x1, :]
        if rgb_obs.shape[1] != self.depth_obs_height or rgb_obs.shape[2] != self.depth_obs_width:
            raise RuntimeError(
                "RGB crop output shape mismatch under rgbd mode: "
                f"got ({rgb_obs.shape[1]}, {rgb_obs.shape[2]}), "
                f"expected ({self.depth_obs_height}, {self.depth_obs_width})."
            )
        return rgb_obs

    def _stack_camera_tensors(self, tensor_list, tensor_name, target_device):
        none_indices = [idx for idx, t in enumerate(tensor_list) if t is None]
        if none_indices:
            raise RuntimeError(
                f"Camera {tensor_name} tensor acquisition failed for env ids "
                f"{none_indices[:8]} (showing up to 8)."
            )
        batch = torch.stack(tensor_list, dim=0).clone().detach()
        if batch.device != target_device:
            batch = batch.to(target_device, non_blocking=True)
        return batch

    def _compute_tac_contact_target(self) -> torch.Tensor:
        if not self.enable_tac_pred:
            raise RuntimeError("_compute_tac_contact_target called while enable_tac_pred=False.")
        if self.tac_pred_link_handle_indices is None:
            raise RuntimeError("tac_pred link indices are not initialized.")
        contacts = self.contact_tensor.view(self.num_envs, -1, 3)
        contacts = contacts[:, self.tac_pred_link_handle_indices, :]
        contact_norm = torch.norm(contacts, dim=-1)
        return torch.where(
            contact_norm >= self.wm_tac_target_thresh,
            torch.ones_like(contact_norm),
            torch.zeros_like(contact_norm),
        )

    def _depth_tensor_to_frame(self, depth_frame: torch.Tensor) -> np.ndarray:
        depth_img = torch.clamp(depth_frame[..., 0], 0.0, 1.0)
        depth_np = (depth_img.detach().cpu().numpy() * 255.0).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(depth_np[..., None], 3, axis=-1))

    def _wm_depth_tensor_to_display_frame(self, depth_frame: torch.Tensor) -> np.ndarray:
        depth_img = torch.clamp(depth_frame[..., 0], 0.0, 1.0)
        display_img = 1.0 - depth_img
        depth_np = (display_img.detach().cpu().numpy() * 255.0).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(depth_np[..., None], 3, axis=-1))

    @staticmethod
    def _quat_xyzw_to_rot6d(quat_xyzw: torch.Tensor) -> torch.Tensor:
        quat = quat_xyzw / torch.clamp(quat_xyzw.norm(dim=-1, keepdim=True), min=1e-8)
        x = quat[:, 0]
        y = quat[:, 1]
        z = quat[:, 2]
        w = quat[:, 3]

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        col0 = torch.stack((1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy)), dim=-1)
        col1 = torch.stack((2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx)), dim=-1)
        return torch.cat((col0, col1), dim=-1)

    def _compute_pose_target(self) -> torch.Tensor:
        root = self.root_state_tensor
        if isinstance(self.object_indices, torch.Tensor):
            obj_indices = self.object_indices.to(dtype=torch.long, device=root.device)
        else:
            obj_indices = torch.tensor(
                [int(idx) for idx in self.object_indices],
                dtype=torch.long,
                device=root.device,
            )
        obj_root = root[obj_indices]
        obj_pos = obj_root[:, 0:3]
        obj_rot6d = self._quat_xyzw_to_rot6d(obj_root[:, 3:7])
        return torch.cat((obj_pos, obj_rot6d), dim=-1)

    def _rgb_tensor_to_frame(self, rgb_tensor) -> np.ndarray:
        frame = rgb_tensor[..., :3]
        frame_np = frame.detach().cpu().numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)[..., :3]
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame_np)

    def _record_demo_frames(self):
        if self.demo_recorder is None:
            return
        # Refresh once per demo batch so the overlay uses the same physics state
        # that was rendered into the RGB frame. `fetch_camera_observations` may
        # be entered before any `compute_observations` has run (e.g. from the
        # first `reset()`), in which case object_pos/rot attrs don't exist yet
        # and root_state_tensor may also be stale. This refresh makes the
        # overlay robust to both situations.
        if self.show_object_axes:
            self.gym.refresh_actor_root_state_tensor(self.sim)
        for env_id in range(self.num_envs):
            if not self._recording_allowed_for_env(env_id):
                continue
            frame = self._get_demo_rgb_frame(env_id)
            if self.show_object_axes:
                frame = self._draw_object_axes_on_frame(frame, env_id)
            self.demo_recorder.append(env_id, frame)

    def _record_inference_frames(self, rgb_obs, depth_obs):
        if self.inference_recorder is None:
            return
        for env_id in range(self.num_envs):
            if not self._recording_allowed_for_env(env_id):
                continue
            if depth_obs is not None:
                frame = self._depth_tensor_to_frame(depth_obs[env_id])
            elif rgb_obs is not None:
                frame = self._rgb_tensor_to_frame(rgb_obs[env_id])
            else:
                continue
            self.inference_recorder.append(env_id, frame)

    def _record_wm_depth_frames(self, pred_depth, gt_depth):
        if self.wm_depth_pred_recorder is None and self.wm_depth_gt_recorder is None:
            return
        if pred_depth is None and gt_depth is None:
            return
        for env_id in range(self.num_envs):
            if not self._recording_allowed_for_env(env_id):
                continue
            if self.wm_depth_pred_recorder is not None and pred_depth is not None:
                pred_frame = self._wm_depth_tensor_to_display_frame(pred_depth[env_id])
                self.wm_depth_pred_recorder.append(env_id, pred_frame)
            if self.wm_depth_gt_recorder is not None and gt_depth is not None:
                gt_frame = self._wm_depth_tensor_to_display_frame(gt_depth[env_id])
                self.wm_depth_gt_recorder.append(env_id, gt_frame)

    def fetch_camera_observations(self, imagined_pc=None, fsr_pc=None):
        has_active_recording = (
            self.demo_recorder is not None
            or self.inference_recorder is not None
            or self.wm_depth_pred_recorder is not None
            or self.wm_depth_gt_recorder is not None
        )
        if not self.camera_policy_enabled and not has_active_recording:
            return
        if self.camera_policy_enabled:
            self._ensure_obs_container()

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        rgb_obs = None
        depth_obs = None
        depth_obs_noisy = None
        depth_obs_model = None
        depth_obs_clean_model = None
        wm_depth_pred_model = None
        pose_target = None
        target_device = torch.device(self.device)
        if self.camera_policy_enabled:
            self.gym.start_access_image_tensors(self.sim)
            if self.camera_mode in {"rgb", "rgbd"}:
                batch_rgb = self._stack_camera_tensors(self.rgb_image_list, "rgb", target_device)
                rgb_obs = batch_rgb[..., :3]
                rgb_obs = self._apply_rgb_crop_if_needed(rgb_obs)
                self.obs_dict["obs"]["rgb"] = rgb_obs

            if self.camera_mode in {"depth", "rgbd"}:
                batch_depth = self._stack_camera_tensors(self.depth_image_list, "depth", target_device)
                batch_depth = -batch_depth.unsqueeze(-1)
                # batch_depth = self._apply_depth_dr(batch_depth)
                depth_obs = self._normalize_depth(batch_depth)
                # Step 1: crop only → clean base image for WM decoder target
                depth_obs = self._apply_depth_crop(depth_obs)
                # depth_obs_clean: crop done, no noise, no flip → WM decoder target (flipped below)
                depth_obs_clean = depth_obs
                # Step 2: all noise (discontinuity, blur, gaussian, rotation, temporal)
                depth_obs_noisy = self._apply_depth_noise(depth_obs)
                # Keep recorded/visualized depth semantics unchanged (near=0, far=1),
                # but flip only model input so near objects become high response.
                depth_obs_model = 1.0 - depth_obs_noisy
                # WM decoder target: clean image, same flip convention as model input (near=1, far=0)
                depth_obs_clean_model = 1.0 - depth_obs_clean
                self.obs_dict["obs"]["depth"] = depth_obs_model
                pose_target = self._compute_pose_target()
                self.obs_dict["obs"]["pose_target"] = pose_target
            if self.enable_tac_pred:
                self.gym.refresh_net_contact_force_tensor(self.sim)
                self.obs_dict["obs"]["tac_contact_target"] = self._compute_tac_contact_target()

            # WorldModel encoder + RSSM obs_step. Runs strictly under no_grad
            # (PPO never reaches WM through this path; the deter feature h is
            # injected as a detached input to the actor).
            if self.wm_adapter is not None and depth_obs_model is not None:
                # prop = single-step proprio (NOT the stacked obs the actor sees).
                prop_vec = self.last_obs_buf
                # `self.actions` is the action that drove the just-completed
                # physics step (set in pre_physics_step). That is exactly
                # RSSM's "prev_action" semantics for the current obs_step.
                # On the very first reset (before any pre_physics_step), it may
                # not exist yet; fall back to zeros.
                actions_now = getattr(self, "actions", None)
                if actions_now is None:
                    actions_now = torch.zeros(
                        (self.num_envs, self.num_actions),
                        device=self.device,
                        dtype=torch.float32,
                    )
                self.wm_adapter.cache_prev_action(actions_now)
                wm_feature = self.wm_adapter.step(prop_vec, depth_obs_model)
                if self.wm_depth_video_record_enabled:
                    wm_depth_pred_model = self.wm_adapter.decode_posterior_image()
                self.obs_dict["obs"]["wm_feature"] = wm_feature
                self._record_wm_feature_snapshot(wm_feature)
                tac_target_for_wm = self.obs_dict["obs"].get("tac_contact_target", None)
                self._wm_append_buffered(
                    prop_vec,
                    depth_obs_model,
                    depth_obs_clean_model,
                    pose_target,
                    tac_target_for_wm,
                )

            self.gym.end_access_image_tensors(self.sim)

        self._record_demo_frames()
        self._record_inference_frames(
            rgb_obs, depth_obs_noisy if self.camera_mode in {"depth", "rgbd"} else depth_obs
        )
        self._record_wm_depth_frames(wm_depth_pred_model, depth_obs_clean_model)
        self.camera_step += 1

    # ------------------------------------------------------------------
    # Episode transition hook (called from base reset_idx)
    # ------------------------------------------------------------------

    def _wm_append_buffered(self, prop_vec, image_tensor, image_clean_tensor=None,
                            pose_target=None, tac_contact_target=None):
        """Append the just-computed (prop, image, action, reward, is_first, ...) row.

        Called from fetch_camera_observations right after wm_adapter.step(). We
        defer the is_first bookkeeping to a per-env flag set by on_episode_reset,
        which fires before the post-reset physics step.

        image_clean_tensor: clean (crop-only, no noise, near=1 far=0) depth for WM
            decoder supervision. If None, falls back to image_tensor.
        pose_target: object pose target (N, 9) for WM pose_pred aux head.
        tac_contact_target: tactile contact target (N, n_links) for WM tac_pred aux head.
        """
        if self.wm_adapter is None:
            return
        if not hasattr(self, "_wm_pending_is_first") or self._wm_pending_is_first is None:
            self._wm_pending_is_first = torch.ones(
                (self.num_envs,), device=self.device, dtype=torch.float32
            )
        actions_now = getattr(self, "actions", None)
        if actions_now is None:
            actions_now = torch.zeros(
                (self.num_envs, self.num_actions),
                device=self.device,
                dtype=torch.float32,
            )
        self.wm_adapter.append(
            prop=prop_vec,
            image=image_tensor,
            image_clean=image_clean_tensor if image_clean_tensor is not None else image_tensor,
            action=actions_now,
            reward=self.rew_buf,
            is_first=self._wm_pending_is_first,
            pose_target=pose_target,
            tac_contact_target=tac_contact_target,
            obj_shape_target=getattr(self, "object_bps_vector_per_env", None),
        )
        # After consuming the flag once, this episode's subsequent steps are not first.
        self._wm_pending_is_first.zero_()

    def on_episode_reset(self, env_ids):
        if env_ids is None:
            env_ids_metrics = None
        elif isinstance(env_ids, torch.Tensor):
            env_ids_metrics = env_ids.to(device=self.device, dtype=torch.long).view(-1)
        else:
            env_ids_metrics = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
        self._record_eval_metrics_episode_end(env_ids_metrics)

        # WM: tag the next ring-append row as the first frame for these envs,
        # and reset the recurrent state so obs_step rebuilds from initial().
        if self.wm_adapter is not None and env_ids is not None:
            if not hasattr(self, "_wm_pending_is_first") or self._wm_pending_is_first is None:
                self._wm_pending_is_first = torch.ones(
                    (self.num_envs,), device=self.device, dtype=torch.float32
                )
            if isinstance(env_ids, torch.Tensor):
                env_ids_wm = env_ids.to(device=self.device, dtype=torch.long).view(-1)
            else:
                env_ids_wm = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).view(-1)
            if env_ids_wm.numel() > 0:
                self._wm_pending_is_first[env_ids_wm] = 1.0
                self.wm_adapter.reset_envs(env_ids_wm)
        if self.persistent_noise_mask is not None and env_ids is not None:
            if isinstance(env_ids, torch.Tensor):
                env_ids_t = env_ids.to(device=self.persistent_noise_mask.device, dtype=torch.long).view(-1)
            else:
                env_ids_t = torch.tensor(
                    env_ids, device=self.persistent_noise_mask.device, dtype=torch.long
                ).view(-1)
            if env_ids_t.numel() > 0:
                env_ids_t = env_ids_t[
                    (env_ids_t >= 0) & (env_ids_t < self.persistent_noise_mask.shape[0])
                ]
            if env_ids_t.numel() > 0:
                self.persistent_noise_mask[env_ids_t] = False
        if env_ids is None:
            return
        if isinstance(env_ids, torch.Tensor):
            env_ids_iter = env_ids.detach().cpu().tolist()
        else:
            env_ids_iter = list(env_ids)

        if self._periodic_eval_active:
            self._step_eval_state_machine(
                env_ids_iter,
                state_dict=self._periodic_eval_state,
                enter_fn=self._enter_recording_state,
                on_complete=self._stop_periodic_eval,
                log_prefix="periodic-eval",
            )
            return

        if self._test_eval_active:
            self._step_eval_state_machine(
                env_ids_iter,
                state_dict=self._test_eval_state,
                enter_fn=self._enter_test_recording_state,
                on_complete=self._finish_test_eval,
                log_prefix="test-eval",
            )
            return

        # Outside of any eval window: nothing to flush; recorders are off.

    def _step_eval_state_machine(
        self,
        env_ids_iter,
        state_dict,
        enter_fn,
        on_complete,
        log_prefix,
    ):
        def _on_pending_entered(env_id_int):
            # The base task already ran _record_spin_trace_reset_step for this
            # reset before the env entered the active spin-trace set. Rerun it
            # so periodic-eval csv captures the step=0 initialization row.
            if env_id_int in self.spin_trace_env_ids and getattr(self, "spin_trace_enabled", False):
                try:
                    self._record_spin_trace_reset_step(env_id_int)
                except Exception as exc:
                    print(f"[{log_prefix}][spin_trace] reset snapshot failed env={env_id_int}: {exc}")

        self.recording_manager.step_state_machine(
            env_ids_iter=env_ids_iter,
            state_dict=state_dict,
            on_pending_entered=_on_pending_entered,
            on_complete=on_complete,
            episode_dir_fn=self._per_env_episode_dir,
        )
        self._sync_recording_manager_hdf5_aliases()

    def _check_periodic_eval_timeout(self):
        if not self._periodic_eval_active:
            return
        self._periodic_eval_steps += 1
        if (
            self._periodic_eval_max_steps > 0
            and self._periodic_eval_steps >= self._periodic_eval_max_steps
        ):
            pending = sum(1 for v in self._periodic_eval_state.values() if v == "pending")
            recording = sum(1 for v in self._periodic_eval_state.values() if v == "recording")
            print(
                "[periodic-eval] timeout; forcing cleanup "
                f"steps={self._periodic_eval_steps} "
                f"pending={pending} recording={recording}"
            )
            self._stop_periodic_eval()

    def reset(self):
        obs = super().reset()
        if (
            self.camera_policy_enabled
            or self.demo_recorder is not None
            or self.inference_recorder is not None
            or self.wm_depth_pred_recorder is not None
            or self.wm_depth_gt_recorder is not None
        ):
            self.fetch_camera_observations()
            if self.wm_adapter is not None and "wm_feature" not in self.obs_dict["obs"]:
                self.obs_dict["obs"]["wm_feature"] = torch.zeros(
                    (self.num_envs, self.wm_adapter.wm_feature_dim),
                    device=self.device,
                    dtype=torch.float,
                )
        return obs

    def reset_done(self):
        obs, done_ids = super().reset_done()
        if (
            self.camera_policy_enabled
            or self.demo_recorder is not None
            or self.inference_recorder is not None
            or self.wm_depth_pred_recorder is not None
            or self.wm_depth_gt_recorder is not None
        ):
            self.fetch_camera_observations()
            if self.wm_adapter is not None and "wm_feature" not in self.obs_dict["obs"]:
                self.obs_dict["obs"]["wm_feature"] = torch.zeros(
                    (self.num_envs, self.wm_adapter.wm_feature_dim),
                    device=self.device,
                    dtype=torch.float,
                )
        return obs, done_ids

    def post_physics_step(self):
        super().post_physics_step()
        self._record_periodic_eval_hdf5_step()
        self._capture_terminated_by_fall()
        self._record_eval_metrics_step()
        self._check_periodic_eval_timeout()
        # Standalone test only: tear down cleanly and exit once the test eval
        # window finishes. is_test_run guards against this ever firing during
        # training even if some other code path mistakenly flips the flag.
        if self._test_run_pending_exit:
            self._test_run_pending_exit = False
            if not self.is_test_run:
                # Defensive: training never wants to SystemExit. Just clear
                # the flag and continue.
                return
            print("[test-run] eval window finished; exiting.")
            try:
                self.close()
            except Exception:
                pass
            raise SystemExit(0)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
