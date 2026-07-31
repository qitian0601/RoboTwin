#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK_NAME="beat_block_hammer"
TASK_CONFIG="demo_nero_beat_block_hammer_slow120"
EPISODES=50
TRAIN_EPISODES=40
SOURCE_ROOT="${ROOT}/data/beat_block_hammer/adapter_source_slow50_hold3s_20260729"
TEMPLATE_ROOT="${ROOT}/data/beat_block_hammer_adapter_c0_lerobot_v3_50"
OUTPUT_ROOT="${ROOT}/data/beat_block_hammer_adapter_multiview_50"
CACHE_ROOT="${ROOT}/outputs/pi05_feature_adapter/hammer_adapter_multiview_cache"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
LEROBOT_SITE="${ROBOTWIN_PYARROW_SITE:-}"
LOCK_FILE="${ROOT}/outputs/.hammer_adapter_multiview_50.lock"

mkdir -p "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another hammer multi-view pipeline owns ${LOCK_FILE}" >&2
  exit 1
fi

if pgrep -af 'evaluate_pi05_camera_views.py|run_server.sh|deploy_policy.py|train_pi05_view_feature_adapter.py' >/dev/null; then
  echo "Refusing to start while PI0.5 evaluation/server/training is running." >&2
  pgrep -af 'evaluate_pi05_camera_views.py|run_server.sh|deploy_policy.py|train_pi05_view_feature_adapter.py' >&2
  exit 1
fi

cd "${ROOT}"
export PYTHONNOUSERSITE=1

source_episode_count() {
  if [[ ! -d "${SOURCE_ROOT}/data" ]]; then
    echo 0
    return
  fi
  find "${SOURCE_ROOT}/data" -maxdepth 1 -type f -name 'episode*.hdf5' 2>/dev/null | wc -l
}

echo "[paths] source=${SOURCE_ROOT}"
echo "[paths] template=${TEMPLATE_ROOT}"
echo "[paths] multiview=${OUTPUT_ROOT}"
echo "[paths] cache=${CACHE_ROOT}"

if (( $(source_episode_count) < EPISODES )); then
  echo "[stage 1/4] collecting ${EPISODES} canonical hammer episodes"
  env \
    ROBOTWIN_COLLECTION_OUTPUT_ROOT="${SOURCE_ROOT}" \
    ROBOTWIN_COLLECTION_EPISODE_NUM="${EPISODES}" \
    ROBOTWIN_COLLECTION_SEED_START="3000000" \
    "${ROBOTWIN_PYTHON}" -u script/collect_data.py "${TASK_NAME}" "${TASK_CONFIG}"
else
  echo "[stage 1/4] source already contains ${EPISODES} episodes"
fi

if (( $(source_episode_count) != EPISODES )); then
  echo "Expected exactly ${EPISODES} source episodes, found $(source_episode_count)" >&2
  exit 1
fi

"${ROBOTWIN_PYTHON}" script/prepare_hammer_adapter_instructions.py \
  --source-root "${SOURCE_ROOT}" \
  --episode-count "${EPISODES}"

if [[ ! -f "${TEMPLATE_ROOT}/meta/info.json" ]]; then
  if [[ -e "${TEMPLATE_ROOT}" ]]; then
    incomplete_template="${TEMPLATE_ROOT}.incomplete.$(date +%Y%m%d_%H%M%S)"
    echo "Preserving incomplete template as ${incomplete_template}"
    mv "${TEMPLATE_ROOT}" "${incomplete_template}"
  fi
  echo "[stage 2/4] converting canonical source to LeRobot v3"
  INPUT_DIR="${SOURCE_ROOT}" \
  OUTPUT_DIR="${TEMPLATE_ROOT}" \
  REPO_ID="beat_block_hammer_adapter_c0_lerobot_v3_50" \
    tools/convert_nero_task_lerobot_v3.sh \
      "${TASK_NAME}" "${TASK_CONFIG}" "${EPISODES}"
else
  echo "[stage 2/4] LeRobot template already exists"
fi

if ! rg -q '"total_episodes"[[:space:]]*:[[:space:]]*50' "${TEMPLATE_ROOT}/meta/info.json"; then
  echo "LeRobot template does not contain 50 episodes: ${TEMPLATE_ROOT}" >&2
  exit 1
fi

echo "[stage 3/4] replaying synchronized C0-C6 and wrist cameras"
env ${LEROBOT_SITE:+ROBOTWIN_PYARROW_SITE="${LEROBOT_SITE}"} \
  "${ROBOTWIN_PYTHON}" -u script/replay_adapter_multiview_dataset.py \
    --task-name "${TASK_NAME}" \
    --task-config "${TASK_CONFIG}" \
    --template-root "${TEMPLATE_ROOT}" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --episode-count "${EPISODES}" \
    --train-count "${TRAIN_EPISODES}" \
    --sample-hz 30 \
    --max-duration 20 \
    --success-hold-s 0.4 \
    --success-max-linear-velocity 0.08 \
    --success-max-angular-velocity 1.5

echo "[stage 4/4] exporting Adapter train/val cache"
"${ROBOTWIN_PYTHON}" script/export_pi05_adapter_cache.py \
  --dataset-root "${OUTPUT_ROOT}" \
  --split train \
  --output "${CACHE_ROOT}"
"${ROBOTWIN_PYTHON}" script/export_pi05_adapter_cache.py \
  --dataset-root "${OUTPUT_ROOT}" \
  --split val \
  --output "${CACHE_ROOT}"

echo "Hammer Adapter multi-view dataset complete: ${OUTPUT_ROOT}"
