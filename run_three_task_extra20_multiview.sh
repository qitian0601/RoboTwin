#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
EPISODES="${EPISODES:-20}"
TRAIN_EPISODES="${TRAIN_EPISODES:-16}"
RUN_TAG="${RUN_TAG:-extra20_20260724}"
DATASET_ROOT="${DATASET_ROOT:-${ROOT}/data/three_task_multiview_${RUN_TAG}}"

TASKS=(place_two_cubes_box pick_dual_bottles beat_block_hammer)
declare -A CONFIGS=(
  [place_two_cubes_box]=demo_nero_two_cubes
  [pick_dual_bottles]=demo_nero_pick_dual_bottles_c0
  [beat_block_hammer]=demo_nero_beat_block_hammer_c0
)
declare -A EXISTING_SOURCES=(
  [place_two_cubes_box]="${ROOT}/data/place_two_cubes_box/demo_nero_two_cubes"
  [pick_dual_bottles]="${ROOT}/data/pick_dual_bottles/demo_nero_pick_dual_bottles_c0"
  [beat_block_hammer]="${ROOT}/data/beat_block_hammer/demo_nero_beat_block_hammer_c0"
)

if (( TRAIN_EPISODES < 0 || TRAIN_EPISODES > EPISODES )); then
  echo "TRAIN_EPISODES must be in 0..EPISODES" >&2
  exit 2
fi

next_unused_seed() {
  local seed_file="$1"
  if [[ ! -s "${seed_file}" ]]; then
    echo 0
    return
  fi
  awk '{for (i = 1; i <= NF; i++) if ($i > max) max = $i} END {print max + 1}' \
    "${seed_file}"
}

mkdir -p "${DATASET_ROOT}/source" "${DATASET_ROOT}/lerobot_v3" \
  "${DATASET_ROOT}/multiview"

echo "Dataset root: ${DATASET_ROOT}"
echo "Per task: ${EPISODES} episodes (${TRAIN_EPISODES} train, $((EPISODES - TRAIN_EPISODES)) val)"
echo "Views: C0-C6 plus left/right wrist"

for task in "${TASKS[@]}"; do
  config="${CONFIGS[${task}]}"
  source_root="${DATASET_ROOT}/source/${task}"
  template_root="${DATASET_ROOT}/lerobot_v3/${task}"
  multiview_root="${DATASET_ROOT}/multiview/${task}"
  seed_start="$(next_unused_seed "${EXISTING_SOURCES[${task}]}/seed.txt")"

  echo
  echo "===== ${task}: collect ${EPISODES} new C0 expert episodes ====="
  SOURCE_ROOT="${source_root}" SEED_START="${seed_start}" \
    "${ROOT}/tools/collect_nero_task_episodes.sh" \
      "${task}" "${config}" "${EPISODES}"

  echo "===== ${task}: convert new episodes to LeRobot v3 ====="
  INPUT_DIR="${source_root}" \
  OUTPUT_DIR="${template_root}" \
  REPO_ID="${task}_${RUN_TAG}_lerobot_v3" \
    "${ROOT}/tools/convert_nero_task_lerobot_v3.sh" \
      "${task}" "${config}" "${EPISODES}"

  echo "===== ${task}: replay synchronized C0-C6 views ====="
  TEMPLATE_ROOT="${template_root}" \
  SOURCE_ROOT="${source_root}" \
  OUTPUT_ROOT="${multiview_root}" \
    "${ROOT}/tools/generate_nero_task_adapter_multiview.sh" \
      "${task}" "${config}" "${EPISODES}" "${TRAIN_EPISODES}"
done

echo
echo "All three task datasets completed: ${DATASET_ROOT}"
