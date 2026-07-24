#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SELECTED_TASK="${1:-all}"
EPISODES="${2:-50}"
TRAIN_EPISODES="${3:-40}"

TASKS=(click_bell pick_dual_bottles beat_block_hammer)
declare -A CONFIGS=(
  [beat_block_hammer]=demo_nero_beat_block_hammer_c0
  [click_bell]=demo_nero_click_bell_c0
  [pick_dual_bottles]=demo_nero_pick_dual_bottles_c0
)

if [[ "${SELECTED_TASK}" != "all" ]]; then
  if [[ -z "${CONFIGS[${SELECTED_TASK}]+x}" ]]; then
    echo "Unknown task: ${SELECTED_TASK}" >&2
    echo "Choose: all, ${TASKS[*]}" >&2
    exit 2
  fi
  TASKS=("${SELECTED_TASK}")
fi

for task in "${TASKS[@]}"; do
  config="${CONFIGS[${task}]}"
  echo "===== ${task}: collect ${EPISODES} C0 episodes ====="
  "${ROOT}/tools/collect_nero_task_episodes.sh" "${task}" "${config}" "${EPISODES}"

  echo "===== ${task}: convert to LeRobot v3 ====="
  "${ROOT}/tools/convert_nero_task_lerobot_v3.sh" "${task}" "${config}" "${EPISODES}"

  echo "===== ${task}: generate C0-C6 synchronized views ====="
  "${ROOT}/tools/generate_nero_task_adapter_multiview.sh" \
    "${task}" "${config}" "${EPISODES}" "${TRAIN_EPISODES}"
done

if [[ "${SELECTED_TASK}" == "all" && "${EPISODES}" == "50" ]]; then
  CACHE_ROOT="${ROOT}/outputs/pi05_feature_adapter/adapter_multitask_cache"
  DATASET_ARGS=()
  for task in "${TASKS[@]}"; do
    DATASET_ARGS+=(--dataset-root "${ROOT}/data/${task}_adapter_multiview_${EPISODES}")
  done
  "${ROBOTWIN_PYTHON:-python}" \
    script/export_pi05_adapter_cache.py \
    "${DATASET_ARGS[@]}" --split train --output "${CACHE_ROOT}"
  "${ROBOTWIN_PYTHON:-python}" \
    script/export_pi05_adapter_cache.py \
    "${DATASET_ARGS[@]}" --split val --output "${CACHE_ROOT}"
  echo "Combined adapter cache: ${CACHE_ROOT}"
fi

echo "All requested pipelines completed."
