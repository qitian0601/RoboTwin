#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_NAME="${1:?usage: generate_nero_task_adapter_multiview.sh TASK_NAME TASK_CONFIG [EPISODES] [TRAIN_EPISODES]}"
TASK_CONFIG="${2:?usage: generate_nero_task_adapter_multiview.sh TASK_NAME TASK_CONFIG [EPISODES] [TRAIN_EPISODES]}"
EPISODES="${3:-50}"
TRAIN_EPISODES="${4:-40}"
TEMPLATE_ROOT="${TEMPLATE_ROOT:-${ROOT}/data/${TASK_NAME}_lerobot_v3}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/data/${TASK_NAME}/${TASK_CONFIG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/data/${TASK_NAME}_adapter_multiview_${EPISODES}}"

cd "${ROOT}"
export PYTHONNOUSERSITE=1

"${ROBOTWIN_PYTHON:-python}" -u \
  script/replay_adapter_multiview_dataset.py \
  --task-name "${TASK_NAME}" \
  --task-config "${TASK_CONFIG}" \
  --template-root "${TEMPLATE_ROOT}" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --episode-count "${EPISODES}" \
  --train-count "${TRAIN_EPISODES}" \
  --sample-hz 30 \
  "${@:5}"

echo "Adapter multi-view dataset: ${OUTPUT_ROOT}"
