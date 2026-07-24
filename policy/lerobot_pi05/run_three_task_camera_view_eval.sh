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

POLICY_PATH="${ROBOTWIN_PI05_POLICY_PATH:?Set ROBOTWIN_PI05_POLICY_PATH to pretrained_model or deployment_checkpoint}"

"${ROBOTWIN_PYTHON:-python}" script/evaluate_pi05_three_tasks_camera_views.py \
  --policy-path "${POLICY_PATH}" \
  --server-address "${ROBOTWIN_PI05_SERVER_ADDRESS:-127.0.0.1:8081}" \
  --episodes-per-view "${ROBOTWIN_CAMERA_EVAL_EPISODES:-20}" \
  --seed "${ROBOTWIN_CAMERA_EVAL_SEED:-0}" \
  "$@"
