#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_ROOT="${ADAPTER_MULTIVIEW_OUTPUT_ROOT:-${ROOT}/data/place_two_cubes_box_adapter_multiview_50}"
CACHE_ROOT="${ADAPTER_MULTIVIEW_CACHE_ROOT:-${ROOT}/outputs/pi05_feature_adapter/adapter_multiview_cache}"

cd "${ROOT}"
export PYTHONNOUSERSITE=1

"${ROBOTWIN_PYTHON:-python}" -u \
  script/replay_adapter_multiview_dataset.py \
  --template-root "${ROOT}/data/place_two_cubes_box_lerobot_v3" \
  --source-root "${ROOT}/data/place_two_cubes_box/demo_nero_two_cubes" \
  --output-root "${OUTPUT_ROOT}" \
  "$@"

if [[ " $* " != *" --prepare-only "* && " $* " != *" --validate-only "* ]]; then
  "${ROBOTWIN_PYTHON:-python}" \
    script/export_pi05_adapter_cache.py \
    --dataset-root "${OUTPUT_ROOT}" \
    --split train \
    --output "${CACHE_ROOT}"
  "${ROBOTWIN_PYTHON:-python}" \
    script/export_pi05_adapter_cache.py \
    --dataset-root "${OUTPUT_ROOT}" \
    --split val \
    --output "${CACHE_ROOT}"
fi
