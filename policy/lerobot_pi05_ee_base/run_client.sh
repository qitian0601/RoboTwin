#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

if [[ -n "${ROBOTWIN_CONDA_ENV:-}" && -n "${CONDA_EXE:-}" ]]; then
  source "$(dirname "${CONDA_EXE}")/../etc/profile.d/conda.sh"
  conda activate "${ROBOTWIN_CONDA_ENV}"
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROBOTWIN_LEROBOT_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}:${PYTHONPATH:-}"

POLICY_PATH="${ROBOTWIN_PI05_EE_POLICY_PATH:-/path/to/14d-nero-base-ee-pi05-checkpoint/pretrained_model}"
TASK_NAME="${ROBOTWIN_PI05_TASK_NAME:-place_two_cubes_box}"
TASK_CONFIG="${ROBOTWIN_PI05_TASK_CONFIG:-demo_nero_two_cubes}"
SERVER_ADDRESS="${ROBOTWIN_PI05_SERVER_ADDRESS:-127.0.0.1:8081}"
SEED="${ROBOTWIN_PI05_SEED:-0}"
SCENARIO_SEED="${ROBOTWIN_PI05_SCENARIO_SEED:-}"
TEST_NUM="${ROBOTWIN_PI05_TEST_NUM:-1}"
VIDEO_STRIDE="${ROBOTWIN_PI05_EE_VIDEO_STRIDE:-3}"
VIDEO_WIDTH="${ROBOTWIN_PI05_EE_VIDEO_WIDTH:-640}"
VIDEO_HEIGHT="${ROBOTWIN_PI05_EE_VIDEO_HEIGHT:-400}"
VIDEO_CRF="${ROBOTWIN_PI05_EE_VIDEO_CRF:-28}"
VIDEO_PRESET="${ROBOTWIN_PI05_EE_VIDEO_PRESET:-ultrafast}"
CARRY_LIFT_OFFSET="${ROBOTWIN_PI05_EE_CARRY_LIFT_OFFSET_M:-0}"
ACTION_MERGE_NEW_WEIGHT="${ROBOTWIN_PI05_EE_ACTION_MERGE_NEW_WEIGHT:-0.75}"
CHUNK_SIZE_THRESHOLD="${ROBOTWIN_PI05_EE_CHUNK_SIZE_THRESHOLD:-0.8}"
DEFAULT_CKPT_SETTING="$(basename "$(dirname "${POLICY_PATH}")")"
CKPT_SETTING="${ROBOTWIN_PI05_EE_CKPT_SETTING:-${DEFAULT_CKPT_SETTING}}"

if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "PI0.5 EE checkpoint not found: ${POLICY_PATH}" >&2
  echo "Set ROBOTWIN_PI05_EE_POLICY_PATH to the pretrained_model directory." >&2
  exit 1
fi

SCENARIO_ARGS=()
if [[ -n "${SCENARIO_SEED}" ]]; then
  SCENARIO_ARGS=(--scenario_seed "${SCENARIO_SEED}")
fi

"${ROBOTWIN_PYTHON:-python}" script/eval_policy.py \
  --config policy/lerobot_pi05_ee_base/deploy_policy.yml \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --policy_name lerobot_pi05_ee_base \
  --policy_path "${POLICY_PATH}" \
  --server_address "${SERVER_ADDRESS}" \
  --ckpt_setting "${CKPT_SETTING}" \
  --seed "${SEED}" \
  --test_num "${TEST_NUM}" \
  --eval_video_stride "${VIDEO_STRIDE}" \
  --eval_video_output_width "${VIDEO_WIDTH}" \
  --eval_video_output_height "${VIDEO_HEIGHT}" \
  --eval_video_crf "${VIDEO_CRF}" \
  --eval_video_preset "${VIDEO_PRESET}" \
  --carry_lift_offset_m "${CARRY_LIFT_OFFSET}" \
  --action_merge_new_weight "${ACTION_MERGE_NEW_WEIGHT}" \
  --chunk_size_threshold "${CHUNK_SIZE_THRESHOLD}" \
  "${SCENARIO_ARGS[@]}"
