#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_PATH="${ROBOTWIN_PI05_POLICY_PATH:?Set ROBOTWIN_PI05_POLICY_PATH to a pretrained_model directory}"
RUN_DIR="${ROBOTWIN_HAMMER_RUN_DIR:-${ROOT}/outputs/checkpoint_success_eval/hammer_trainrange_exec40_new1_10eps}"
SERVER_ADDRESS="${ROBOTWIN_PI05_SERVER_ADDRESS:-127.0.0.1:8081}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"

cd "${ROOT}"
"${ROBOTWIN_PYTHON}" -u script/evaluate_pi05_camera_views.py \
  --policy-path "${POLICY_PATH}" \
  --server-address "${SERVER_ADDRESS}" \
  --task-name beat_block_hammer \
  --task-config demo_nero_beat_block_hammer_slow120 \
  --episodes-per-view 10 \
  --seed 0 \
  --policy-inference-seed 123 \
  --instruction "Take the hammer in the right gripper and strike the block." \
  --hammer-arm-aware-instruction \
  --hammer-contact-success \
  --hammer-training-support-range \
  --view C0 \
  --actions-per-chunk 50 \
  --chunk-size-threshold 0.2 \
  --action-merge-new-weight 1.0 \
  --record-video \
  --video-stride 1 \
  --video-width 1280 \
  --video-height 800 \
  --video-crf 23 \
  --video-preset ultrafast \
  --run-dir "${RUN_DIR}"
