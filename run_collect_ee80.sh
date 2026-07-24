#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

TASK_NAME="place_two_cubes_box"
TASK_CONFIG="demo_nero_two_cubes_ee80"
OUTPUT_DIR="data/${TASK_NAME}/${TASK_CONFIG}"
LOG_DIR="logs/collect_${TASK_CONFIG}_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/collect.log"

mkdir -p "$LOG_DIR"

echo "Collecting/resuming 80 episodes in: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"

conda run --no-capture-output -n "${ROBOTWIN_CONDA_ENV:-RoboTwin5090}" \
  python -u script/collect_data.py "$TASK_NAME" "$TASK_CONFIG" \
  2>&1 | tee "$LOG_FILE"

episode_count="$(find "${OUTPUT_DIR}/data" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)"
if [[ "$episode_count" -ne 80 ]]; then
  echo "ERROR: expected 80 HDF5 episodes, found ${episode_count}" >&2
  exit 1
fi

echo "Done: ${episode_count} episodes in ${OUTPUT_DIR}"
