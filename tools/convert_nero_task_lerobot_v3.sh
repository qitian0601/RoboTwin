#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_NAME="${1:?usage: convert_nero_task_lerobot_v3.sh TASK_NAME TASK_CONFIG [EPISODES]}"
TASK_CONFIG="${2:?usage: convert_nero_task_lerobot_v3.sh TASK_NAME TASK_CONFIG [EPISODES]}"
EPISODES="${3:-50}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/data/${TASK_NAME}/${TASK_CONFIG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/data/${TASK_NAME}_lerobot_v3}"
REPO_ID="${REPO_ID:-${TASK_NAME}_lerobot_v3}"

LEROBOT_SRC="${ROBOTWIN_LEROBOT_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}"
if [[ ! -d "${LEROBOT_SRC}/lerobot" ]]; then
  echo "Bundled LeRobot source does not exist: ${LEROBOT_SRC}" >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONNOUSERSITE=1
export ROBOTWIN_LEROBOT_SRC="${LEROBOT_SRC}"
export PYTHONPATH="${LEROBOT_SRC}:${PYTHONPATH:-}"
conda run --no-capture-output -n "${ROBOTWIN_LEROBOT_ENV:-lerobot-nero}" \
  "${ROBOTWIN_LEROBOT_PYTHON_NAME:-python}" -u tools/convert_robotwin_to_lerobot_v3.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --repo-id "${REPO_ID}" \
  --max-episodes "${EPISODES}" \
  --fps 30 \
  --front-image-size 800 1280 \
  --wrist-image-size 480 640 \
  --video-codec h264 \
  --action-mode next_state \
  --right-left-order

echo "LeRobot v3 dataset: ${OUTPUT_DIR}"
