#!/usr/bin/env bash
# WM-Craftnet y-axis training entry.
# This script intentionally keeps overrides minimal.
# Usage:
#   1) Edit only device switches below.
#   2) Run: ./scripts/train_wm_craftnet_y.sh [optional hydra args...]
set -euo pipefail

# === Runtime/device switches (keep minimal) ===
SIM_GPU=0
RL_GPU=""
GRAPHICS_GPU=0
HEADLESS=true
# Memory-safe defaults for hosts with 32 GiB RAM. Override as needed:
# NUM_ENVS=1024 MINIBATCH_SIZE=4096 bash scripts/train_wm_craftnet_y.sh
NUM_ENVS="${NUM_ENVS:-256}" #1024
MINIBATCH_SIZE="${MINIBATCH_SIZE:-1024}" #4096

sanitize_colon_path_var() {
    # Remove entries matching a pattern from a colon-separated env var.
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
# Keep conda runtime libs discoverable for gym_38.so.
if [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# Keep RL GPU aligned with SIM GPU when left empty.
if [ -z "${RL_GPU}" ]; then
    RL_GPU="${SIM_GPU}"
fi
HEADLESS_BOOL="False"
if [[ "${HEADLESS}" =~ ^(1|true|TRUE|True)$ ]]; then
    HEADLESS_BOOL="True"
fi

python ./isaacgymenvs/train.py \
sim_device=cuda:${SIM_GPU} rl_device=cuda:${RL_GPU} graphics_device_id=${GRAPHICS_GPU} \
task=WMCraftnetRotationY \
train=WMCraftnetPPO \
headless=${HEADLESS_BOOL} \
train.params.tuning.num_envs=${NUM_ENVS} \
train.params.tuning.minibatch_size=${MINIBATCH_SIZE} \
"$@"
