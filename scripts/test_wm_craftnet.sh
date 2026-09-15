#!/usr/bin/env bash
# Copyright (c) 2026 The WM-Craftnet Authors
# SPDX-License-Identifier: Apache-2.0
#
# WM-Craftnet interactive visualization (no metrics / recording).
# Uses the canonical repository config with checkpoint-compatible WSM heads.
#
# Supported presets:
#   set_z  -> 9 objects
#   set_y  -> 9 objects
#
# Usage:
#   bash scripts/test_wm_craftnet.sh
#   TEST_OBJ_SET=set_y bash scripts/test_wm_craftnet.sh
#   CHECKPOINT=path/to/checkpoint.pth TEST_OBJ_SET=set_z bash scripts/test_wm_craftnet.sh
set -euo pipefail

SIM_GPU=0
RL_GPU=""
GRAPHICS_GPU=0
# Viewer on by default for this script.
HEADLESS="${HEADLESS:-false}"
TEST_OBJ_SET="${TEST_OBJ_SET:-set_z}"
TEST_EPISODE_LENGTH="${TEST_EPISODE_LENGTH:-490}"

case "${TEST_OBJ_SET}" in
    set_z)
        CHECKPOINT_DEFAULT="example_ckpt/wm_craftnet_set_z.pth"
        TEST_NUM_ENVS_DEFAULT=9
        TASK_CONFIG="WMCraftnetRotation"
        TEST_AXIS="z"
        WM_HEAD_ARGS=(
            task.env.cameraPolicy.worldModel.wm_prop_pred=True
            task.env.cameraPolicy.worldModel.wm_depth_pred=True
            task.env.cameraPolicy.worldModel.wm_pose_pred=False
            task.env.cameraPolicy.worldModel.wm_tac_pred=False
            task.env.cameraPolicy.worldModel.wm_value_pred=False
            task.env.cameraPolicy.worldModel.wm_obj_pred=False
        )
        ;;
    set_y)
        CHECKPOINT_DEFAULT="example_ckpt/wm_craftnet_set_y.pth"
        TEST_NUM_ENVS_DEFAULT=9
        TASK_CONFIG="WMCraftnetRotationY"
        TEST_AXIS="y"
        WM_HEAD_ARGS=(
            task.env.cameraPolicy.worldModel.wm_prop_pred=True
            task.env.cameraPolicy.worldModel.wm_depth_pred=True
            task.env.cameraPolicy.worldModel.wm_pose_pred=True
            task.env.cameraPolicy.worldModel.wm_tac_pred=True
            task.env.cameraPolicy.worldModel.wm_value_pred=True
            task.env.cameraPolicy.worldModel.wm_obj_pred=True
        )
        ;;
    *)
        echo "Unsupported TEST_OBJ_SET='${TEST_OBJ_SET}'. Use set_z or set_y." >&2
        exit 1
        ;;
esac

CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_DEFAULT}}"
TEST_NUM_ENVS="${TEST_NUM_ENVS:-${TEST_NUM_ENVS_DEFAULT}}"
TEST_MINIBATCH_SIZE="${TEST_MINIBATCH_SIZE:-${TEST_NUM_ENVS}}"

sanitize_colon_path_var() {
    local var_name="$1"
    local skip_pattern="$2"
    local raw_value="${!var_name:-}"
    local cleaned=""
    local entry

    IFS=':' read -r -a entries <<< "${raw_value}"
    for entry in "${entries[@]}"; do
        [ -z "${entry}" ] && continue
        [[ "${entry}" == *"${skip_pattern}"* ]] && continue
        cleaned="${cleaned:+${cleaned}:}${entry}"
    done
    export "${var_name}=${cleaned}"
}

# IsaacLab/IsaacSim paths in parent shell may conflict with Isaac Gym Preview4.
sanitize_colon_path_var "PYTHONPATH" "/_isaac_sim"
sanitize_colon_path_var "LD_LIBRARY_PATH" "/_isaac_sim"
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${RL_GPU}" ]; then
    RL_GPU="${SIM_GPU}"
fi

to_bool() {
    local v="$1"
    if [[ "${v}" =~ ^(1|true|TRUE|True)$ ]]; then
        echo "True"
    else
        echo "False"
    fi
}

HEADLESS_BOOL=$(to_bool "${HEADLESS}")

echo "[test-viz] checkpoint=${CHECKPOINT}"
echo "[test-viz] objSet=${TEST_OBJ_SET} axis=${TEST_AXIS} numEnvs=${TEST_NUM_ENVS} headless=${HEADLESS_BOOL}"

status=0
python ./isaacgymenvs/train.py headless=${HEADLESS_BOOL} \
train=WMCraftnetPPO \
sim_device=cuda:${SIM_GPU} rl_device=cuda:${RL_GPU} graphics_device_id=${GRAPHICS_GPU} \
task=${TASK_CONFIG} \
wandb_activate=False test=True \
checkpoint=${CHECKPOINT} \
train.params.tuning.num_envs=${TEST_NUM_ENVS} \
train.params.tuning.minibatch_size=${TEST_MINIBATCH_SIZE} \
task.env.numEnvs=${TEST_NUM_ENVS} \
task.env.episodeLength=${TEST_EPISODE_LENGTH} \
task.env.objSet=${TEST_OBJ_SET} \
++task.env.isTestRun=True \
"${WM_HEAD_ARGS[@]}" \
task.env.evalMetrics.enabled=False \
task.env.cameraDemo.enabled=False \
task.env.cameraDemo.inference_video.enabled=False \
task.env.wmDepthVideoRecord.enabled=False \
task.env.spinTrace.enabled=False \
+task.env.wmFeatureRecord.enabled=False \
"$@" || status=$?

if [ "${status}" -eq 139 ]; then
    echo "[test-viz] Python exited with 139; treating Isaac Gym teardown segfault as successful."
    exit 0
fi

exit "${status}"
