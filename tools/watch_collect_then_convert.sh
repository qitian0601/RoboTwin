#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INPUT_DIR="${INPUT_DIR:-$(pwd)/data/place_two_cubes_box/demo_nero_two_cubes}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/data/place_two_cubes_box_lerobot_v3}"
REPO_ID="${REPO_ID:-place_two_cubes_box_lerobot_v3}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
STABLE_CYCLES_REQUIRED="${STABLE_CYCLES_REQUIRED:-3}"
IDLE_CONVERT_CYCLES="${IDLE_CONVERT_CYCLES:-30}"
LOG_DIR="${LOG_DIR:-}"

if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$(find logs -maxdepth 1 -type d -name 'collect_place_two_cubes_box_demo_nero_two_cubes_*' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
fi

if [[ -z "$LOG_DIR" || ! -d "$LOG_DIR" ]]; then
  echo "ERROR: collect log dir not found: ${LOG_DIR}" >&2
  exit 1
fi

echo "Monitoring log dir: ${LOG_DIR}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Repo id: ${REPO_ID}"
echo "Stable cycles required: ${STABLE_CYCLES_REQUIRED}"
echo "Idle convert cycles: ${IDLE_CONVERT_CYCLES}"
echo "Sleep seconds: ${SLEEP_SECONDS}"

stable_cycles=0
idle_cycles=0
last_signature=""
last_activity_signature=""
finish_reason=""

while true; do
  hdf5_count="$(find "${INPUT_DIR}/data" -maxdepth 1 -type f -name 'episode*.hdf5' 2>/dev/null | wc -l)"
  log_count="$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'run_*.log' 2>/dev/null | wc -l)"
  latest_log="$(find "${LOG_DIR}" -maxdepth 1 -type f -name 'run_*.log' 2>/dev/null | sort -V | tail -1)"
  latest_mtime="none"
  latest_success=0
  latest_name="none"

  if [[ -n "$latest_log" ]]; then
    latest_name="$(basename "$latest_log")"
    latest_mtime="$(stat -c %Y "$latest_log")"
    if grep -q 'Successfully Saved Instructions' "$latest_log"; then
      latest_success=1
    fi
  fi

  cache_count="$(find "${INPUT_DIR}/.cache" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)"
  signature="${hdf5_count}:${log_count}:${latest_mtime}:${cache_count}:${latest_success}"
  activity_signature="${hdf5_count}:${log_count}:${latest_mtime}"

  echo "[$(date +"%F %T")] hdf5=${hdf5_count}, logs=${log_count}, cache=${cache_count}, latest=${latest_name}, success=${latest_success}, stable=${stable_cycles}/${STABLE_CYCLES_REQUIRED}, idle=${idle_cycles}/${IDLE_CONVERT_CYCLES}"

  if [[ "$latest_success" == "1" && "$cache_count" == "0" && "$signature" == "$last_signature" ]]; then
    stable_cycles=$((stable_cycles + 1))
  else
    stable_cycles=0
  fi
  last_signature="$signature"

  if [[ "$activity_signature" == "$last_activity_signature" ]]; then
    idle_cycles=$((idle_cycles + 1))
  else
    idle_cycles=0
  fi
  last_activity_signature="$activity_signature"

  if (( stable_cycles >= STABLE_CYCLES_REQUIRED )); then
    finish_reason="normal_finish"
    break
  fi

  if (( hdf5_count > 0 && idle_cycles >= IDLE_CONVERT_CYCLES )); then
    finish_reason="idle_timeout_partial_convert"
    break
  fi

  sleep "$SLEEP_SECONDS"
done

echo "[$(date +"%F %T")] Collection appears finished; reason=${finish_reason}; starting conversion."
INPUT_DIR="${INPUT_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
REPO_ID="${REPO_ID}" \
  tools/convert_nero_task_lerobot_v3.sh \
    place_two_cubes_box demo_nero_two_cubes

echo "[$(date +"%F %T")] Conversion finished: ${OUTPUT_DIR}"
