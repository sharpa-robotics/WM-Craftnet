import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import h5py
except Exception:
    h5py = None


MIN_VALID_VIDEO_FRAMES = 2


def scan_next_game_index(record_root, filename_prefix="", channel_name="demo"):
    """Scan record_root for the next available episode index.

    Layout (current): <record_root>/<NNN_obj>/episode_<idx>/<channel>.mp4
    Legacy fallback : <record_root>/<NNN_obj>/<prefix><channel>_game_<idx>.mp4
    """
    record_root = str(record_root).strip()
    if not record_root or not os.path.isdir(record_root):
        return 0
    max_idx = -1
    try:
        # New layout: per-episode subfolder named episode_<idx>.
        episode_dir_pattern = re.compile(r"^episode_(\d+)$")
        for entry in os.listdir(record_root):
            sub = os.path.join(record_root, entry)
            if not os.path.isdir(sub):
                continue
            for child in os.listdir(sub):
                m = episode_dir_pattern.match(child)
                if not m:
                    continue
                ep_dir = os.path.join(sub, child)
                if any(name.endswith(".mp4") for name in os.listdir(ep_dir)):
                    max_idx = max(max_idx, int(m.group(1)))
    except Exception:
        max_idx = -1
    if max_idx >= 0:
        return max_idx + 1

    # Legacy: filename-based scan kept so old run dirs continue to advance.
    pattern = re.compile(rf"^{re.escape(filename_prefix)}{re.escape(channel_name)}_game_(\d+)\.mp4$")
    try:
        for root, _, files in os.walk(record_root):
            for name in files:
                match = pattern.match(name)
                if not match:
                    continue
                max_idx = max(max_idx, int(match.group(1)))
    except Exception:
        return 0
    return max_idx + 1


@dataclass
class EpisodeStepData:
    env_id: int
    pd_target: np.ndarray
    qpos: np.ndarray
    done: bool
    policy_action: Optional[np.ndarray] = None


class VideoEpisodeRecorder:
    """Per-env episode-sliced mp4 writer."""

    def __init__(
        self,
        channel_name,
        fps,
        bitrate,
        game_idx_base=0,
        filename_prefix="",
    ):
        self.channel_name = channel_name
        self.codec = "libx264"
        self.fps = int(fps)
        self.bitrate = str(bitrate)
        self.game_idx_base = int(game_idx_base)
        self.filename_prefix = str(filename_prefix)

        self.timestamp_root = None
        self.env_obj_names = []
        self.env_obj_name_by_id = {}
        self.env_ids = []
        self.game_idx_per_env = {}
        self.writers = {}
        self.frames_written_per_env = {}
        self._imageio = None
        self._append_error_reported = False

    def init(self, timestamp_root, env_obj_names, env_ids=None):
        import imageio.v2 as imageio

        self._imageio = imageio
        self.timestamp_root = timestamp_root
        self.env_obj_names = list(env_obj_names)
        if env_ids is None:
            self.env_ids = list(range(len(self.env_obj_names)))
        else:
            self.env_ids = [int(env_id) for env_id in env_ids]
        self.env_obj_name_by_id = {
            env_id: self.env_obj_names[idx] if 0 <= idx < len(self.env_obj_names) else f"env{env_id}"
            for idx, env_id in enumerate(self.env_ids)
        }
        if self.game_idx_base <= 0:
            self.game_idx_per_env = {
                env_id: self._next_game_idx_from_existing_files(env_id)
                for env_id in self.env_ids
            }
        else:
            self.game_idx_per_env = {env_id: self.game_idx_base for env_id in self.env_ids}

    def _env_root_dir(self, env_id):
        obj_name = self.env_obj_name_by_id.get(env_id, f"env{env_id}")
        return os.path.join(self.timestamp_root, f"{env_id:03d}_{obj_name}")

    def _next_game_idx_from_existing_files(self, env_id):
        folder = self._env_root_dir(env_id)
        if not os.path.isdir(folder):
            return 0
        # Layout: <env_dir>/episode_<idx>/<channel>.mp4
        ep_pattern = re.compile(r"^episode_(\d+)$")
        max_idx = -1
        try:
            for name in os.listdir(folder):
                match = ep_pattern.match(name)
                if not match:
                    continue
                max_idx = max(max_idx, int(match.group(1)))
        except Exception:
            return 0
        return max_idx + 1

    def episode_dir(self, env_id, game_idx=None):
        """Return the per-env, per-episode artifact directory.

        Used by external producers (hdf5, spin trace) so that all artifacts
        for a given (env, episode) pair land next to the mp4.
        """
        if game_idx is None:
            game_idx = self.game_idx_per_env.get(env_id, 0)
        folder = os.path.join(self._env_root_dir(env_id), f"episode_{int(game_idx)}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _path_for(self, env_id):
        game_idx = self.game_idx_per_env.get(env_id, 0)
        ep_dir = self.episode_dir(env_id, game_idx)
        # filename_prefix is preserved for compatibility with external scanners
        # (e.g. test-replay's scan_next_game_index) but the channel name alone
        # is enough to disambiguate within a per-episode folder.
        filename = f"{self.filename_prefix}{self.channel_name}.mp4"
        return os.path.join(ep_dir, filename)

    def _open_writer(self, path):
        kwargs = dict(
            fps=self.fps,
            format="FFMPEG",
            codec=self.codec,
            bitrate=self.bitrate,
            quality=10,
            pixelformat="yuv420p",
        )
        writer = self._imageio.get_writer(path, **kwargs)
        return writer, self.codec

    def start_episode(self, env_id):
        if self.timestamp_root is None:
            return
        self._close_env(env_id, drop_if_empty=True)
        path = self._path_for(env_id)
        writer, codec_used = self._open_writer(path)
        self.writers[env_id] = (path, writer, codec_used)
        self.frames_written_per_env[env_id] = 0

    def append(self, env_id, frame_np):
        entry = self.writers.get(env_id)
        if entry is None:
            self.start_episode(env_id)
            entry = self.writers.get(env_id)
            if entry is None:
                return
        _, writer, _ = entry
        try:
            writer.append_data(frame_np)
            self.frames_written_per_env[env_id] = self.frames_written_per_env.get(env_id, 0) + 1
        except Exception as exc:
            if not self._append_error_reported:
                path = entry[0]
                print(
                    f"[video][{self.channel_name}] append failed for env {env_id} path='{path}': {exc}"
                )
                self._append_error_reported = True
            self._close_env(env_id, drop_if_empty=True)

    def end_episode(self, env_id):
        had_frames = self._close_env(env_id, drop_if_empty=True)
        if had_frames:
            self.game_idx_per_env[env_id] = self.game_idx_per_env.get(env_id, 0) + 1
        return had_frames

    def _close_env(self, env_id, drop_if_empty=False):
        entry = self.writers.pop(env_id, None)
        if entry is None:
            return False
        path, writer, _ = entry
        try:
            writer.close()
        except Exception:
            pass
        n_frames = int(self.frames_written_per_env.pop(env_id, 0))
        if drop_if_empty and n_frames < MIN_VALID_VIDEO_FRAMES:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
            return False
        return n_frames > 0

    def close_all(self):
        for env_id in list(self.writers.keys()):
            self._close_env(env_id, drop_if_empty=True)


class Hdf5EpisodeRecorder:
    def __init__(self, output_dir, episode_idx_base=0):
        self.output_dir = str(output_dir)
        self.episode_idx_base = int(episode_idx_base)
        self.h5_enabled = h5py is not None
        # output_dir is created lazily in save_episode (dirname of output_path)
        # so callers that route every write through an explicit per-episode
        # path don't end up with a stray empty fallback dir.

    def save_episode(
        self,
        episode_idx,
        actions,
        qpos,
        dones,
        replay_actions=None,
        joint_names=None,
        output_path=None,
    ):
        if not self.h5_enabled:
            return None
        episode_idx = self.episode_idx_base + int(episode_idx)
        if output_path is None:
            output_path = os.path.join(self.output_dir, f"episode_{episode_idx:06d}.hdf5")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with h5py.File(output_path, "w") as h5_file:
            h5_file.create_dataset("action", data=np.asarray(actions, dtype=np.float32))
            state_data = np.asarray(qpos, dtype=np.float32)
            h5_file.create_dataset("state", data=state_data)
            h5_file.create_dataset("qpos", data=state_data)
            h5_file.create_dataset("done", data=np.asarray(dones, dtype=np.bool_))
            if replay_actions is not None:
                h5_file.create_dataset("policy_action", data=np.asarray(replay_actions, dtype=np.float32))
            if joint_names:
                names_arr = np.asarray(joint_names, dtype=h5py.string_dtype(encoding="utf-8"))
                h5_file.create_dataset("joint_names", data=names_arr)
            h5_file.attrs["action_semantics"] = "pd_target_joint_angle_rad"
            h5_file.attrs["policy_action_semantics"] = "policy_output_sent_to_env_step"
            h5_file.attrs["state_semantics"] = "pd_state_joint_angle_rad"
            h5_file.attrs["episode_index"] = int(episode_idx)
        return output_path


def flush_spin_trace_episode(
    rows,
    output_dir,
    episode_idx,
    finger_layout,
    show_tip_err=False,
    title="Spin Trace",
    csv_path=None,
    png_path=None,
):
    if not rows:
        return None, None, None
    if csv_path is None:
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, f"episode_{int(episode_idx):04d}_spin_trace.csv")
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if png_path is None:
        png_path = os.path.join(output_dir, f"episode_{int(episode_idx):04d}_spin_trace.png")
    headers = list(rows[0].keys())
    np_rows = np.asarray(
        [[row.get(header, np.nan) for header in headers] for row in rows],
        dtype=np.float64,
    )
    np.savetxt(
        csv_path,
        np_rows,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec

        header_to_idx = {name: idx for idx, name in enumerate(headers)}
        t = np_rows[:, header_to_idx["time_s"]]
        finger_order = ["thumb", "index", "middle", "ring", "pinky"]
        max_rows = 1
        for finger_key in finger_order:
            joint_count = len(finger_layout.get(finger_key, {}).get("joint_names", []))
            max_rows = max(max_rows, 2 + joint_count)
        max_rows = max(max_rows, 5)
        fig = plt.figure(figsize=(32, 3.2 * max_rows))
        gs = gridspec.GridSpec(max_rows, 7, figure=fig, wspace=0.35, hspace=0.45)
        fig.suptitle(title, fontsize=14)

        ax_pos_x = fig.add_subplot(gs[0, 0])
        ax_pos_x.plot(t, np_rows[:, header_to_idx["obj_x"]], label="x", color="tab:blue")
        ax_pos_x.set_title("object pos")
        ax_pos_x.set_ylabel("x (m)")
        ax_pos_x.grid(True, alpha=0.3)

        ax_pos_y = fig.add_subplot(gs[1, 0], sharex=ax_pos_x)
        ax_pos_y.plot(t, np_rows[:, header_to_idx["obj_y"]], label="y", color="tab:orange")
        ax_pos_y.set_ylabel("y (m)")
        ax_pos_y.grid(True, alpha=0.3)

        ax_pos_z = fig.add_subplot(gs[2, 0], sharex=ax_pos_x)
        ax_pos_z.plot(t, np_rows[:, header_to_idx["obj_z"]], label="z", color="tab:green")
        ax_pos_z.set_ylabel("z (m)")
        ax_pos_z.grid(True, alpha=0.3)

        for row_idx in range(3, max_rows):
            ax_empty = fig.add_subplot(gs[row_idx, 0], sharex=ax_pos_x)
            ax_empty.axis("off")

        ax_wx = fig.add_subplot(gs[0, 1], sharex=ax_pos_x)
        ax_wx.plot(t, np_rows[:, header_to_idx["wx_deg_s"]], label="wx", color="tab:blue")
        ax_wx.set_title("object ang vel")
        ax_wx.set_ylabel("wx (deg/s)")
        ax_wx.grid(True, alpha=0.3)

        ax_wy = fig.add_subplot(gs[1, 1], sharex=ax_pos_x)
        ax_wy.plot(t, np_rows[:, header_to_idx["wy_deg_s"]], label="wy", color="tab:orange")
        ax_wy.set_ylabel("wy (deg/s)")
        ax_wy.grid(True, alpha=0.3)

        ax_wz = fig.add_subplot(gs[2, 1], sharex=ax_pos_x)
        ax_wz.plot(t, np_rows[:, header_to_idx["wz_deg_s"]], label="wz", color="tab:green")
        ax_wz.set_ylabel("wz (deg/s)")
        ax_wz.grid(True, alpha=0.3)

        ax_axis = fig.add_subplot(gs[3, 1], sharex=ax_pos_x)
        ax_axis.plot(t, np_rows[:, header_to_idx["spin_rate_axis_deg_s"]], color="tab:red")
        ax_axis.set_ylabel("spin axis\n(deg/s)")
        ax_axis.grid(True, alpha=0.3)

        ax_off = fig.add_subplot(gs[4, 1], sharex=ax_pos_x)
        ax_off.plot(t, np_rows[:, header_to_idx["spin_rate_offaxis_deg_s"]], color="tab:purple")
        ax_off.set_ylabel("spin offaxis\n(deg/s)")
        ax_off.set_xlabel("time (s)")
        ax_off.grid(True, alpha=0.3)

        for row_idx in range(5, max_rows):
            ax_empty = fig.add_subplot(gs[row_idx, 1], sharex=ax_pos_x)
            ax_empty.axis("off")

        for finger_col, finger_key in enumerate(finger_order, start=2):
            layout = finger_layout.get(finger_key, {})
            joint_names = layout.get("joint_names", [])
            touch_key = f"{finger_key}_touch"

            ax_touch = fig.add_subplot(gs[0, finger_col], sharex=ax_pos_x)
            ax_touch.plot(t, np_rows[:, header_to_idx[touch_key]], color="tab:blue")
            ax_touch.set_title(finger_key)
            ax_touch.set_ylabel("force (N)")
            ax_touch.grid(True, alpha=0.3)

            ax_tip = fig.add_subplot(gs[1, finger_col], sharex=ax_pos_x)
            ax_tip.plot(t, np_rows[:, header_to_idx[f"{finger_key}_tip_x"]], label="x", color="tab:blue")
            ax_tip.plot(t, np_rows[:, header_to_idx[f"{finger_key}_tip_y"]], label="y", color="tab:orange")
            ax_tip.plot(t, np_rows[:, header_to_idx[f"{finger_key}_tip_z"]], label="z", color="tab:green")
            ax_tip.plot([], [], color="black", linestyle="-", label="sim (solid)")
            if show_tip_err:
                err_key = f"tip_err_{finger_key}"
                if err_key in header_to_idx:
                    err_mean = float(np.nanmean(np_rows[:, header_to_idx[err_key]]))
                    ax_tip.plot([], [], " ", label=f"err={err_mean:.4f}m")
            ax_tip.set_ylabel("tip xyz (m)", fontsize=8)
            ax_tip.grid(True, alpha=0.3)
            ax_tip.legend(loc="upper right", fontsize=7)

            for joint_plot_idx, joint_name in enumerate(joint_names, start=2):
                joint_key = f"{finger_key}_{joint_name}_deg"
                ax_joint = fig.add_subplot(gs[joint_plot_idx, finger_col], sharex=ax_pos_x)
                state_key = f"{finger_key}_{joint_name}_state_deg"
                action_key = f"{finger_key}_{joint_name}_action_deg"
                state_plot_key = state_key if state_key in header_to_idx else joint_key
                ax_joint.plot(
                    t,
                    np_rows[:, header_to_idx[state_plot_key]],
                    linestyle="-",
                    label="state",
                )
                if action_key in header_to_idx:
                    ax_joint.plot(
                        t,
                        np_rows[:, header_to_idx[action_key]],
                        linestyle="--",
                        label="action",
                    )
                ax_joint.set_ylabel(f"{joint_name.replace('right_', '')} (deg)", fontsize=8)
                ax_joint.grid(True, alpha=0.3)
                if action_key in header_to_idx:
                    ax_joint.legend(loc="upper right", fontsize=7)

            for row_idx in range(2 + len(joint_names), max_rows):
                ax_empty = fig.add_subplot(gs[row_idx, finger_col], sharex=ax_pos_x)
                ax_empty.axis("off")
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        return csv_path, None, str(exc)

    return csv_path, png_path, None
