#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

RUNS="${1:-200}"
TASK_NAME="${TASK_NAME:-place_two_cubes_box}"
TASK_CONFIG="${TASK_CONFIG:-demo_nero_two_cubes}"
DATA_DIR="data/${TASK_NAME}/${TASK_CONFIG}/data"
LOG_DIR="logs/collect_${TASK_NAME}_${TASK_CONFIG}_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

count_hdf5() {
  if [[ -d "$DATA_DIR" ]]; then
    find "$DATA_DIR" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l
  else
    echo 0
  fi
}

echo "Task: ${TASK_NAME} ${TASK_CONFIG}"
echo "Runs: ${RUNS}"
echo "Data dir: ${DATA_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Existing episodes: $(count_hdf5)"

for i in $(seq 1 "$RUNS"); do
  before_count="$(count_hdf5)"
  log_file="${LOG_DIR}/run_$(printf '%03d' "$i").log"

  echo
  echo "[$(date '+%F %T')] Run ${i}/${RUNS}, existing episodes: ${before_count}"
  echo "Logging to ${log_file}"

  conda run --no-capture-output -n "${ROBOTWIN_CONDA_ENV:-RoboTwin5090}" \
    python -u script/collect_data.py "$TASK_NAME" "$TASK_CONFIG" \
    2>&1 | tee "$log_file"

  after_count="$(count_hdf5)"
  echo "[$(date '+%F %T')] Run ${i}/${RUNS} finished, episodes: ${before_count} -> ${after_count}"

  if [[ "$after_count" -le "$before_count" ]]; then
    echo "ERROR: episode count did not increase. Stop to avoid repeated failed collection." >&2
    exit 1
  fi
done

echo
echo "Done. Final episodes: $(count_hdf5)"
