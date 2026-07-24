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

BASE_POLICY_PATH="${ROBOTWIN_PI05_BASE_POLICY_PATH:?Set ROBOTWIN_PI05_BASE_POLICY_PATH}"
ADAPTER_POLICY_PATH="${ROBOTWIN_PI05_ADAPTER_POLICY_PATH:?Set ROBOTWIN_PI05_ADAPTER_POLICY_PATH}"
SEEDS_FILE="${ROBOTWIN_UNSEEN_EVAL_SEEDS_FILE:?Set ROBOTWIN_UNSEEN_EVAL_SEEDS_FILE}"

"${ROBOTWIN_PYTHON:-python}" script/evaluate_pi05_unseen_view_ab.py \
  --base-policy-path "${BASE_POLICY_PATH}" \
  --adapter-policy-path "${ADAPTER_POLICY_PATH}" \
  --scenario-seeds-file "${SEEDS_FILE}" \
  --server-address "${ROBOTWIN_PI05_SERVER_ADDRESS:-127.0.0.1:8081}" \
  --episodes-per-view "${ROBOTWIN_UNSEEN_EVAL_EPISODES:-10}" \
  --policy-inference-seed "${ROBOTWIN_PI05_INFERENCE_SEED:-0}" \
  "$@"
