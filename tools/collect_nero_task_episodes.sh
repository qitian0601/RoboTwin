#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_NAME="${1:?usage: collect_nero_task_episodes.sh TASK_NAME TASK_CONFIG [TARGET_EPISODES]}"
TASK_CONFIG="${2:?usage: collect_nero_task_episodes.sh TASK_NAME TASK_CONFIG [TARGET_EPISODES]}"
TARGET_EPISODES="${3:-50}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT}/data/${TASK_NAME}/${TASK_CONFIG}}"
SEED_START="${SEED_START:-}"
DATA_DIR="${SOURCE_ROOT}/data"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/collect_${TASK_NAME}_${TASK_CONFIG}_${RUN_ID}"

mkdir -p "${LOG_DIR}" "${DATA_DIR}"
cd "${ROOT}"
export PYTHONNOUSERSITE=1

count_episodes() {
  find "${DATA_DIR}" -maxdepth 1 -type f -name 'episode*.hdf5' 2>/dev/null | wc -l
}

echo "Task: ${TASK_NAME}"
echo "Config: ${TASK_CONFIG}"
echo "Target episodes: ${TARGET_EPISODES}"
echo "Source root: ${SOURCE_ROOT}"
if [[ -n "${SEED_START}" ]]; then
  echo "Initial seed: ${SEED_START}"
fi
echo "Log dir: ${LOG_DIR}"

while (( $(count_episodes) < TARGET_EPISODES )); do
  before="$(count_episodes)"
  episode_log="${LOG_DIR}/episode_$(printf '%03d' "${before}").log"
  echo "[$(date '+%F %T')] collecting ${before}/${TARGET_EPISODES} -> ${episode_log}"

  env \
    ROBOTWIN_COLLECTION_OUTPUT_ROOT="${SOURCE_ROOT}" \
    ROBOTWIN_COLLECTION_SEED_START="${SEED_START}" \
    conda run --no-capture-output -n "${ROBOTWIN_CONDA_ENV:-RoboTwin5090}" \
      "${ROBOTWIN_PYTHON_NAME:-python}" -u script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}" \
    2>&1 | tee "${episode_log}"

  after="$(count_episodes)"
  if (( after <= before )); then
    echo "ERROR: episode count did not increase (${before} -> ${after})" >&2
    exit 1
  fi
done

echo "Collection complete: $(count_episodes) episodes"
