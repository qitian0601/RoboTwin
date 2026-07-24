#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="${1:-${ROOT}/outputs/camera_view_eval/ab_100_20260722}"
mkdir -p "${RUN_DIR}"
echo "$$" >"${RUN_DIR}/pipeline.pid"
cd "${ROOT}"

run_data_stage() {
  local task="$1"
  echo "[$(date '+%F %T')] starting/resuming ${task}: 80 episodes, 64 train, 16 val"
  if "${ROOT}/run_new_tasks_adapter_data.sh" "${task}" 80 64; then
    echo "[$(date '+%F %T')] ${task} data pipeline completed"
  else
    local status=$?
    echo "[$(date '+%F %T')] WARNING: ${task} data pipeline exited with status ${status}; continuing"
  fi
}

run_data_stage beat_block_hammer
run_data_stage pick_dual_bottles

echo "[$(date '+%F %T')] data stages ended; entering no-adapter/adapter evaluation pipeline"
exec "${ROOT}/tools/run_camera_adapter_ab_100_after_generation.sh" 0 "${RUN_DIR}"
