#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_NAME="${1:?usage: convert_nero_task_lerobot_v3.sh TASK_NAME TASK_CONFIG [EPISODES]}"
TASK_CONFIG="${2:?usage: convert_nero_task_lerobot_v3.sh TASK_NAME TASK_CONFIG [EPISODES]}"
EPISODES="${3:-50}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/data/${TASK_NAME}/${TASK_CONFIG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/data/${TASK_NAME}_lerobot_v3}"
REPO_ID="${REPO_ID:-${TASK_NAME}_lerobot_v3}"

# Prefer the local official LeRobot checkout. The bundled Nero runtime tracks a
# newer Python syntax level and cannot be imported by the Python 3.10 `lerobot`
# environment used for offline conversion on this workstation.
OFFICIAL_LEROBOT_SRC="${ROOT}/../lerobot/src"
if [[ -z "${ROBOTWIN_LEROBOT_SRC:-}" && -d "${OFFICIAL_LEROBOT_SRC}" ]]; then
  export ROBOTWIN_LEROBOT_SRC="${OFFICIAL_LEROBOT_SRC}"
fi

cd "${ROOT}"
# The dedicated `lerobot` environment currently obtains h5py from qt's user
# site-packages. Keep simulator processes isolated, but allow that dependency
# for this offline conversion step.
unset PYTHONNOUSERSITE
conda run --no-capture-output -n "${ROBOTWIN_LEROBOT_ENV:-lerobot}" \
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
