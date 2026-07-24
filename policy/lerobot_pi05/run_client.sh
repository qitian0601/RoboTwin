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

POLICY_PATH="${ROBOTWIN_PI05_POLICY_PATH:-/path/to/16d-nero-pi05-checkpoint/pretrained_model}"
TASK_NAME="${ROBOTWIN_PI05_TASK_NAME:-place_two_cubes_box}"
TASK_CONFIG="${ROBOTWIN_PI05_TASK_CONFIG:-demo_nero_two_cubes}"
SERVER_ADDRESS="${ROBOTWIN_PI05_SERVER_ADDRESS:-127.0.0.1:8081}"
SEED="${ROBOTWIN_PI05_SEED:-0}"
TEST_NUM="${ROBOTWIN_PI05_TEST_NUM:-1}"

"${ROBOTWIN_PYTHON:-python}" script/eval_policy.py \
  --config policy/lerobot_pi05/deploy_policy.yml \
  --overrides \
  --task_name "${TASK_NAME}" \
  --task_config "${TASK_CONFIG}" \
  --policy_name lerobot_pi05 \
  --policy_path "${POLICY_PATH}" \
  --server_address "${SERVER_ADDRESS}" \
  --ckpt_setting "$(basename "$(dirname "${POLICY_PATH}")")" \
  --seed "${SEED}" \
  --test_num "${TEST_NUM}"
