#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_NAME="pick_dual_bottles"
TASK_CONFIG="demo_nero_pick_dual_bottles_slow120"
EPISODES=120
RAW_DIR="${ROOT}/data/pick_dual_bottles/demo_nero_pick_dual_bottles_narrow120"
LEROBOT_DIR="${ROOT}/data/pick_dual_bottles_narrow120_lerobot_v3"
REPO_ID="pick_dual_bottles_narrow120_lerobot_v3"

if [[ -e "${LEROBOT_DIR}" ]]; then
  echo "ERROR: LeRobot output already exists: ${LEROBOT_DIR}" >&2
  exit 1
fi

echo "Narrow bottle collection pipeline"
echo "Raw output: ${RAW_DIR}"
echo "LeRobot output: ${LEROBOT_DIR}"

SOURCE_ROOT="${RAW_DIR}" \
SEED_START="${SEED_START:-5100000}" \
  "${ROOT}/tools/collect_nero_task_episodes.sh" \
  "${TASK_NAME}" "${TASK_CONFIG}" "${EPISODES}"

hdf5_count="$(find "${RAW_DIR}/data" -maxdepth 1 -type f -name 'episode*.hdf5' | wc -l)"
if [[ "${hdf5_count}" -ne "${EPISODES}" ]]; then
  echo "ERROR: expected ${EPISODES} raw episodes, found ${hdf5_count}; conversion skipped" >&2
  exit 1
fi

INPUT_DIR="${RAW_DIR}" \
OUTPUT_DIR="${LEROBOT_DIR}" \
REPO_ID="${REPO_ID}" \
  "${ROOT}/tools/convert_nero_task_lerobot_v3.sh" \
  "${TASK_NAME}" "${TASK_CONFIG}" "${EPISODES}"

echo "Pipeline complete"
echo "Raw dataset: ${RAW_DIR}"
echo "LeRobot dataset: ${LEROBOT_DIR}"
