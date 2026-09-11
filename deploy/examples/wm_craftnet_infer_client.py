#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Minimal real-hand ZMQ client.

Client responsibilities:
1) Read real hand joint angles from SDK.
2) Convert Sharpa order -> Isaac order and send to remote ZMQ server.
3) Receive Isaac-order target joint command and execute on real hand.

No local policy, no local target state, no depth/tactile logic.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

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


def sharpa_to_isaac(sharpa_angles: np.ndarray) -> np.ndarray:
    isaac = np.zeros_like(sharpa_angles, dtype=np.float32)
    for sharpa_i, isaac_i in enumerate(ISAACLAB2SHARPA_IDX):
        isaac[isaac_i] = float(sharpa_angles[sharpa_i])
    return isaac


def isaac_to_sharpa(isaac_angles: np.ndarray) -> np.ndarray:
    return isaac_angles[np.asarray(ISAACLAB2SHARPA_IDX, dtype=np.int64)]


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


def sanitize_colon_path_var(name: str, skip_pattern: str) -> None:
    raw = os.environ.get(name, "")
    if not raw:
        return
    kept = []
    for item in raw.split(":"):
        if not item or skip_pattern in item:
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


def _preload_sdk_libs() -> None:
    deploy_root = Path(__file__).resolve().parents[1]
    candidate_dirs = [deploy_root / "sharpa_sdk" / "lib"]
    required = [
        "libsharpa-wave-sdk.so",
        "libSharpaWaveSDKWrapper.so",
    ]
    loaded = 0
    for libname in required:
        for d in candidate_dirs:
            p = d / libname
            if p.exists():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
                break
    if loaded == 0:
        print("[WARN] No SDK libraries were preloaded; using the system library search path.")


def _select_target_sn(devices: list[str], requested_sn: str, allow_glove: bool) -> str:
    if requested_sn:
        if requested_sn not in devices:
            raise RuntimeError(f"Requested SN not found: {requested_sn}, devices={devices}")
        return requested_sn
    normal = [d for d in devices if "GLOVE" not in d.upper()]
    if normal:
        return normal[0]
    if allow_glove and devices:
        return devices[0]
    raise RuntimeError(f"Only glove-like devices found: {devices}, use --allow-glove to proceed.")


def _discover_devices(manager, timeout_s: float, requested_sn: str = "") -> list[str]:
    start = time.time()
    best = []
    while time.time() - start < max(0.1, timeout_s):
        if hasattr(manager, "get_all_device_sn"):
            devices = list(manager.get_all_device_sn() or [])
        elif hasattr(manager, "discover_devices"):
            devices = list(manager.discover_devices())
        else:
            raise RuntimeError(
                "SharpaWaveManager provides no supported device discovery method "
                "(get_all_device_sn/discover_devices is missing)."
            )
        if devices:
            if requested_sn and requested_sn in devices:
                return devices
            best = devices
            if not requested_sn:
                return devices
        time.sleep(0.2)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal real-hand ZMQ client")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("WM_CRAFTNET_ENDPOINT", "tcp://127.0.0.1:5555"),
        help="ZMQ server endpoint (default: WM_CRAFTNET_ENDPOINT or localhost)",
    )
    parser.add_argument("--request-timeout-ms", type=int, default=120)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--diag-log-interval", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means infinite loop")
    parser.add_argument(
        "--no-actuation",
        action="store_true",
        help="dry run: skip all set_joint_position actuation",
    )

    parser.add_argument("--sn", default="")
    parser.add_argument("--allow-glove", action="store_true")
    parser.add_argument("--discover-timeout", type=float, default=10.0)
    parser.add_argument("--speed-coef", type=float, default=0.3)
    parser.add_argument("--current-coef", type=float, default=0.5)
    parser.add_argument(
        "--enable-tactile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="whether to start tactile pipeline (may trigger TRT build in SDK)",
    )
    parser.add_argument(
        "--tactile-config-path",
        default="deploy/configs/tactile.json",
        help="optional tactile config path when --enable-tactile is used",
    )
    return parser.parse_args()


def main() -> None:
    sanitize_colon_path_var("PYTHONPATH", "/_isaac_sim")
    sanitize_colon_path_var("LD_LIBRARY_PATH", "/_isaac_sim")
    _setup_import_paths()
    args = parse_args()
    _preload_sdk_libs()

    from sharpa import (  # pylint: disable=import-error
        ControlMode,
        ControlSource,
        SharpaWaveConfig,
        SharpaWaveManager,
    )

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, int(args.request_timeout_ms))
    sock.setsockopt(zmq.SNDTIMEO, int(args.request_timeout_ms))
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(args.endpoint)
    print(f"[INFO] ZMQ connected: {args.endpoint}")

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
            tactile_config_path = Path(__file__).resolve().parents[2] / tactile_config_path
        if not tactile_config_path.is_file():
            raise RuntimeError(f"Tactile config not found: {tactile_config_path}")
        sdk_config.tactile_config_file = str(tactile_config_path.resolve())
    hand = manager.connect(sn, sdk_config)
    print(f"[INFO] Connected Sharpa device: {sn}")

    dt = 1.0 / max(float(args.rate_hz), 1e-6)
    last_target_isaac: np.ndarray | None = None
    diag_interval = max(1, int(args.diag_log_interval))
    step = 0

    try:
        if args.no_actuation:
            print("[WARN] no-actuation enabled: no set_joint_position will be sent.")
        hand.set_control_mode(ControlMode.POSITION)
        hand.set_control_source(ControlSource.SDK)
        hand.set_speed_coeff(args.speed_coef)
        hand.set_current_coeff(args.current_coef)
        if args.enable_tactile:
            hand.start()
            print("[INFO] tactile pipeline started.")
        else:
            print("[INFO] tactile pipeline disabled (default): skip hand.start().")

        while True:
            if args.max_steps > 0 and step >= int(args.max_steps):
                break
            t0 = time.perf_counter()

            qpos_sharpa = np.asarray(hand.get_states().angles, dtype=np.float32).reshape(-1)
            if qpos_sharpa.shape[0] != 22:
                raise RuntimeError(f"Unexpected dof count from SDK: {qpos_sharpa.shape[0]}")
            qpos_isaac = sharpa_to_isaac(qpos_sharpa)

            payload = {
                "step_id": int(step),
                "client_ts": float(time.time()),
                "qpos_isaac": qpos_isaac.tolist(),
            }

            got_remote_target = False
            cmd_code, cmd_msg = 0, ""
            rtt_ms = -1.0
            try:
                t_req = time.perf_counter()
                sock.send_string(json.dumps(payload, ensure_ascii=True))
                reply = json.loads(sock.recv_string())
                rtt_ms = (time.perf_counter() - t_req) * 1000.0

                if not bool(reply.get("ok", True)):
                    raise RuntimeError(f"server error: {reply.get('error', 'unknown')}")
                if int(reply.get("step_id", -1)) != step:
                    raise RuntimeError(
                        f"step_id mismatch, sent={step}, recv={reply.get('step_id', None)}"
                    )
                target_arr = np.asarray(reply.get("target_isaac", []), dtype=np.float32).reshape(-1)
                if target_arr.shape[0] != 22 or not np.all(np.isfinite(target_arr)):
                    raise RuntimeError(f"invalid target_isaac from server: shape={target_arr.shape}")
                last_target_isaac = target_arr
                got_remote_target = True
            except Exception as exc:  # noqa: BLE001
                if last_target_isaac is None:
                    print(f"[WARN] step={step} no server target yet, skip command: {exc}")
                else:
                    print(f"[WARN] step={step} server timeout/error, hold last target: {exc}")

            if last_target_isaac is not None and not args.no_actuation:
                cmd_ret = hand.set_joint_position(isaac_to_sharpa(last_target_isaac).tolist())
                cmd_code, cmd_msg = _err_code_msg(cmd_ret)
                _ok(cmd_ret, "set_joint_position(zmq_loop)")

            step += 1
            if step % diag_interval == 0:
                print(
                    f"[INFO][diag] step={step} got_remote_target={int(got_remote_target)} "
                    f"| cmd_code={cmd_code} | cmd_msg='{cmd_msg}' | rtt_ms={rtt_ms:.2f}"
                )

            sleep_t = dt - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)
    finally:
        try:
            hand.stop()
        except Exception:
            pass
        try:
            manager.disconnect_all()
        except Exception:
            pass
        sock.close(0)


if __name__ == "__main__":
    main()
