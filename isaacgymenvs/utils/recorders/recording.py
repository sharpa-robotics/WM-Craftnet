import os

import numpy as np

from isaacgymenvs.utils.recorders.evaluation import Hdf5EpisodeRecorder, VideoEpisodeRecorder


class EpisodeRecordingManager:
    """Shared per-env episode artifact lifecycle for WM-Craftnet tasks.

    The task owns camera capture, metrics, and task-specific traces. This manager
    owns the common recording state: demo/inference mp4 writers, HDF5 replay
    buffers, per-episode directories, and pending/recording/done transitions.
    """

    def __init__(
        self,
        *,
        owner,
        env_obj_names_fn,
        hdf5_step_data_fn,
        joint_names_fn,
        on_enter_recording_fn=None,
        on_finish_recording_fn=None,
    ):
        self.owner = owner
        self.env_obj_names_fn = env_obj_names_fn
        self.hdf5_step_data_fn = hdf5_step_data_fn
        self.joint_names_fn = joint_names_fn
        self.on_enter_recording_fn = on_enter_recording_fn
        self.on_finish_recording_fn = on_finish_recording_fn

        self.record_output_root = None
        self.record_env_ids = []
        self.record_env_id_set = set()
        self.hdf5_enabled = False
        self.hdf5_dir = ""
        self.hdf5_recorder = None
        self.hdf5_episode_idx = {}
        self.hdf5_buffers = {}

    def init_recording(self, *, env_ids, record_root, demo_enabled, inference_enabled, fps, bitrate, game_idx_base, filename_prefix):
        self.record_output_root = str(record_root)
        os.makedirs(self.record_output_root, exist_ok=True)
        self.record_env_ids = [int(env_id) for env_id in env_ids]
        self.record_env_id_set = set(self.record_env_ids)
        env_obj_names = self.env_obj_names_fn(self.record_env_ids)

        self.owner.demo_recorder = None
        self.owner.inference_recorder = None
        if demo_enabled:
            self.owner.demo_recorder = VideoEpisodeRecorder(
                channel_name="demo",
                fps=fps,
                bitrate=bitrate,
                game_idx_base=game_idx_base,
                filename_prefix=filename_prefix,
            )
            self.owner.demo_recorder.init(self.record_output_root, env_obj_names, env_ids=self.record_env_ids)
        if inference_enabled:
            self.owner.inference_recorder = VideoEpisodeRecorder(
                channel_name="inference",
                fps=fps,
                bitrate=bitrate,
                game_idx_base=game_idx_base,
                filename_prefix=filename_prefix,
            )
            self.owner.inference_recorder.init(self.record_output_root, env_obj_names, env_ids=self.record_env_ids)
        print(f"[video] recording to {self.record_output_root} (codec=libx264)")

    def close_recorders(self):
        if self.owner.demo_recorder is not None:
            self.owner.demo_recorder.close_all()
            self.owner.demo_recorder = None
        if self.owner.inference_recorder is not None:
            self.owner.inference_recorder.close_all()
            self.owner.inference_recorder = None

    def per_env_episode_dir(self, env_id, episode_idx, fallback_record_root):
        env_id_int = int(env_id)
        obj_name = None
        if self.owner.demo_recorder is not None:
            obj_name = self.owner.demo_recorder.env_obj_name_by_id.get(env_id_int)
        if obj_name is None and self.owner.inference_recorder is not None:
            obj_name = self.owner.inference_recorder.env_obj_name_by_id.get(env_id_int)
        if obj_name is None:
            try:
                obj_name = self.env_obj_names_fn([env_id_int])[0]
            except Exception:
                obj_name = f"env{env_id_int}"
        env_dir = os.path.join(str(fallback_record_root), f"{env_id_int:03d}_{obj_name}")
        ep_dir = os.path.join(env_dir, f"episode_{int(episode_idx)}")
        os.makedirs(ep_dir, exist_ok=True)
        return ep_dir

    def init_hdf5(self, *, record_env_ids, output_dir, base_idx):
        self.hdf5_enabled = False
        self.hdf5_dir = ""
        self.hdf5_recorder = None
        self.hdf5_episode_idx = {}
        self.hdf5_buffers = {}
        try:
            import h5py  # noqa: F401
        except Exception:
            print("[episode-recording][hdf5] disabled: h5py is unavailable")
            return

        self.hdf5_recorder = Hdf5EpisodeRecorder(output_dir, episode_idx_base=0)
        for env_id in record_env_ids:
            env_id_int = int(env_id)
            self.hdf5_episode_idx[env_id_int] = int(base_idx)
            self.hdf5_buffers[env_id_int] = {"action": [], "state": [], "done": []}
        self.hdf5_dir = str(output_dir)
        self.hdf5_enabled = True

    def reset_hdf5_buffers_for_env(self, env_id):
        env_id_int = int(env_id)
        if env_id_int in self.hdf5_buffers:
            self.hdf5_buffers[env_id_int] = {"action": [], "state": [], "done": []}

    def record_hdf5_step(self, *, state_dict, record_env_ids):
        if not self.hdf5_enabled:
            return
        for env_id in record_env_ids:
            env_id_int = int(env_id)
            if env_id_int not in self.hdf5_buffers:
                continue
            if state_dict.get(env_id_int) != "recording":
                continue
            action_row, state_row, done_flag = self.hdf5_step_data_fn(env_id_int)
            self.hdf5_buffers[env_id_int]["action"].append(np.asarray(action_row, dtype=np.float32))
            self.hdf5_buffers[env_id_int]["state"].append(np.asarray(state_row, dtype=np.float32))
            self.hdf5_buffers[env_id_int]["done"].append(bool(done_flag))

    def flush_hdf5_env(self, *, env_id, force, episode_dir_fn):
        env_id_int = int(env_id)
        if not self.hdf5_enabled:
            return False
        buf = self.hdf5_buffers.get(env_id_int)
        if not buf or len(buf["action"]) == 0:
            return False
        if (not force) and (not bool(buf["done"][-1])):
            return False
        episode_idx = int(self.hdf5_episode_idx.get(env_id_int, 0))
        output_path = os.path.join(episode_dir_fn(env_id_int, episode_idx), "replay_traj.h5")
        self.hdf5_recorder.save_episode(
            episode_idx,
            np.asarray(buf["action"], dtype=np.float32),
            np.asarray(buf["state"], dtype=np.float32),
            np.asarray(buf["done"], dtype=np.bool_),
            joint_names=self.joint_names_fn(),
            output_path=output_path,
        )
        self.hdf5_episode_idx[env_id_int] = episode_idx + 1
        self.hdf5_buffers[env_id_int] = {"action": [], "state": [], "done": []}
        return True

    def flush_all_hdf5(self, *, force, episode_dir_fn):
        if not self.hdf5_enabled:
            return
        for env_id in list(self.hdf5_buffers.keys()):
            self.flush_hdf5_env(env_id=env_id, force=force, episode_dir_fn=episode_dir_fn)

    def recording_allowed_for_env(self, *, env_id, periodic_active, periodic_state, test_active, test_state):
        env_id_int = int(env_id)
        if env_id_int not in self.record_env_id_set:
            return False
        if periodic_active:
            return periodic_state.get(env_id_int) == "recording"
        if test_active:
            return test_state.get(env_id_int) == "recording"
        return True

    def enter_recording_state(self, env_id, state_dict):
        env_id_int = int(env_id)
        self.reset_hdf5_buffers_for_env(env_id_int)
        if env_id_int in self.record_env_id_set:
            if self.owner.demo_recorder is not None:
                self.owner.demo_recorder._close_env(env_id_int, drop_if_empty=True)
            if self.owner.inference_recorder is not None:
                self.owner.inference_recorder._close_env(env_id_int, drop_if_empty=True)
            if callable(self.on_enter_recording_fn):
                self.on_enter_recording_fn(env_id_int)
        state_dict[env_id_int] = "recording"

    def finish_recording_env(self, env_id, *, state_dict, episode_dir_fn):
        env_id_int = int(env_id)
        self.flush_hdf5_env(env_id=env_id_int, force=True, episode_dir_fn=episode_dir_fn)
        if env_id_int in self.record_env_id_set:
            if self.owner.demo_recorder is not None:
                self.owner.demo_recorder.end_episode(env_id_int)
            if self.owner.inference_recorder is not None:
                self.owner.inference_recorder.end_episode(env_id_int)
            if callable(self.on_finish_recording_fn):
                self.on_finish_recording_fn(env_id_int)
        state_dict[env_id_int] = "done"

    def step_state_machine(
        self,
        *,
        env_ids_iter,
        state_dict,
        on_pending_entered=None,
        on_complete,
        episode_dir_fn,
    ):
        any_state_change = False
        for env_id in env_ids_iter:
            env_id_int = int(env_id)
            state = state_dict.get(env_id_int)
            if state is None or state == "done":
                continue
            if state == "pending":
                self.enter_recording_state(env_id_int, state_dict)
                if callable(on_pending_entered):
                    on_pending_entered(env_id_int)
                any_state_change = True
                continue
            self.finish_recording_env(env_id_int, state_dict=state_dict, episode_dir_fn=episode_dir_fn)
            any_state_change = True
        if any_state_change and all(v == "done" for v in state_dict.values()):
            on_complete()
