#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GENERATION_PID="${1:?usage: run_camera_adapter_ab_100_after_generation.sh GENERATION_PID [RUN_DIR]}"
RUN_DIR="${2:-${ROOT}/outputs/camera_view_eval/ab_100_20260722}"
PORT="${ROBOTWIN_PI05_PORT:-8081}"
BASE_POLICY="${ROOT}/outputs/pi05_exact_code_test/pretrained_model"
ADAPTER_POLICY="${ROOT}/outputs/pi05_feature_adapter/train_six_views_teacher_first_20260720/deployment_checkpoint"
SERVER_PID=""

mkdir -p "${RUN_DIR}/no_adapter" "${RUN_DIR}/with_adapter"
echo "$$" >"${RUN_DIR}/queue.pid"
cd "${ROOT}"

cleanup_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

start_server() {
  local log_file="$1"
  env \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONNOUSERSITE=1 \
    HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}" \
    HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}" \
    PYTHONPATH="${ROBOTWIN_PI05_SERVER_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}" \
    "${ROBOTWIN_PI05_SERVER_PYTHON:-python}" \
      -m lerobot.async_inference.policy_server \
      --host=127.0.0.1 \
      --port="${PORT}" \
      --fps=30 \
      --inference_latency=0.033333 \
      --obs_queue_timeout=300 \
      --obs_similarity_atol=0.09 \
      --async_rtc.enabled=false \
      >"${log_file}" 2>&1 &
  SERVER_PID=$!

  for _ in $(seq 1 300); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "Policy server exited during startup; see ${log_file}" >&2
      return 1
    fi
    if ss -ltn | grep -q ":${PORT} "; then
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for policy server on port ${PORT}" >&2
  return 1
}

verify_multiview_dataset() {
  local task="$1"
  local dataset_root="${ROOT}/data/${task}_adapter_multiview_80"
  local completed
  completed="$(find "${dataset_root}" -type f -name _SUCCESS 2>/dev/null | wc -l)"
  if [[ "${completed}" != "80" ]]; then
    echo "${task}: expected 80 completed multi-view episodes, got ${completed}" >&2
    return 1
  fi
  "${ROBOTWIN_PYTHON:-python}" - "${dataset_root}/validation.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing validation file: {path}")
payload = json.loads(path.read_text())
if payload.get("checked_episodes") != 80 or len(payload.get("episodes", [])) != 80:
    raise SystemExit(f"Incomplete validation result: {path}")
PY
}

if [[ "${GENERATION_PID}" != "0" ]]; then
  while kill -0 "${GENERATION_PID}" 2>/dev/null; do
    echo "[$(date '+%F %T')] waiting for data generation PID ${GENERATION_PID}"
    sleep 60
  done
fi

# A reboot or one failed conversion can leave one task partially complete.
# Retry each incomplete data pipeline once; persistent failures remain warnings
# and must not prevent the requested policy evaluation.
for task in beat_block_hammer pick_dual_bottles; do
  if ! verify_multiview_dataset "${task}"; then
    echo "[$(date '+%F %T')] ${task} data is incomplete; retrying its resumable pipeline once"
    if "${ROOT}/run_new_tasks_adapter_data.sh" "${task}" 80 64; then
      echo "[$(date '+%F %T')] ${task} retry completed"
    else
      status=$?
      echo "[$(date '+%F %T')] WARNING: ${task} retry exited with status ${status}; evaluation will still continue"
    fi
  fi
done

DATASETS_VALID=true
for task in beat_block_hammer pick_dual_bottles; do
  if ! verify_multiview_dataset "${task}"; then
    DATASETS_VALID=false
  fi
done
if [[ "${DATASETS_VALID}" == "true" ]]; then
  echo "[$(date '+%F %T')] both 80-episode multi-view datasets passed validation"
else
  echo "[$(date '+%F %T')] WARNING: data generation exited with incomplete or invalid data; continuing evaluation as requested"
fi

echo "[$(date '+%F %T')] starting no-adapter 100x7 evaluation"
start_server "${RUN_DIR}/no_adapter_server.log"
ROBOTWIN_PI05_POLICY_PATH="${BASE_POLICY}" \
ROBOTWIN_CAMERA_EVAL_EPISODES=100 \
ROBOTWIN_CAMERA_EVAL_SEED=0 \
ROBOTWIN_PI05_SERVER_ADDRESS="127.0.0.1:${PORT}" \
  policy/lerobot_pi05/run_camera_view_eval.sh \
    --output-dir "${RUN_DIR}/no_adapter" \
    >"${RUN_DIR}/no_adapter_eval.log" 2>&1
cleanup_server
sleep 5

BASELINE_RESULTS="$({
  find "${RUN_DIR}/no_adapter" -type f -name results.json -printf '%T@ %p\n' \
    | sort -n | tail -1 | cut -d' ' -f2-
} || true)"
if [[ -z "${BASELINE_RESULTS}" ]]; then
  echo "Could not find baseline results.json" >&2
  exit 1
fi
cp "${BASELINE_RESULTS}" "${RUN_DIR}/no_adapter_results.json"
cp "$(dirname "${BASELINE_RESULTS}")/summary.csv" "${RUN_DIR}/no_adapter_summary.csv"

echo "[$(date '+%F %T')] starting adapter 100x7 evaluation"
start_server "${RUN_DIR}/with_adapter_server.log"
ROBOTWIN_PI05_POLICY_PATH="${ADAPTER_POLICY}" \
ROBOTWIN_CAMERA_EVAL_EPISODES=100 \
ROBOTWIN_CAMERA_EVAL_SEED=0 \
ROBOTWIN_PI05_SERVER_ADDRESS="127.0.0.1:${PORT}" \
  policy/lerobot_pi05/run_camera_view_eval.sh \
    --output-dir "${RUN_DIR}/with_adapter" \
    --scenario-seeds-file "${BASELINE_RESULTS}" \
    >"${RUN_DIR}/with_adapter_eval.log" 2>&1
cleanup_server

ADAPTER_RESULTS="$({
  find "${RUN_DIR}/with_adapter" -type f -name results.json -printf '%T@ %p\n' \
    | sort -n | tail -1 | cut -d' ' -f2-
} || true)"
if [[ -z "${ADAPTER_RESULTS}" ]]; then
  echo "Could not find adapter results.json" >&2
  exit 1
fi
cp "${ADAPTER_RESULTS}" "${RUN_DIR}/with_adapter_results.json"
cp "$(dirname "${ADAPTER_RESULTS}")/summary.csv" "${RUN_DIR}/with_adapter_summary.csv"

echo "[$(date '+%F %T')] both 100x7 evaluations completed"
