#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_SRC="${ROBOTWIN_PI05_SERVER_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}"
PYTHON="${ROBOTWIN_PI05_SERVER_PYTHON:-python}"
BASE_CHECKPOINT="${ROBOTWIN_DUAL_TASK_BASE_POLICY:-${ROOT}/../lerobot/outputs/train/three_task_sim/pickplace_hammer_hold3s/checkpoints/024000/pretrained_model}"
INDEX_ROOT="${ROOT}/outputs/pi05_feature_adapter/pickplace_hammer_balanced_cache"
TRAIN_INDEX="${INDEX_ROOT}/train_index.json"
VAL_INDEX="${INDEX_ROOT}/val_index.json"
RUN_NAME="${RUN_NAME:-pickplace_hammer_step024000_teacher_first_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS="${MAX_STEPS:-6000}"
RESUME="${RESUME:-0}"
OUTPUT="${ROOT}/outputs/pi05_feature_adapter/${RUN_NAME}"
LOG_DIR="${ROOT}/logs/feature_adapter"
LOG="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"

if [[ ! -f "${BASE_CHECKPOINT}/model.safetensors" ]]; then
  echo "Missing base checkpoint: ${BASE_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_INDEX}" || ! -f "${VAL_INDEX}" ]]; then
  echo "Missing balanced Adapter indexes below ${INDEX_ROOT}" >&2
  exit 1
fi
if [[ "${RESUME}" == "1" ]]; then
  if [[ ! -d "${OUTPUT}" ]] || ! compgen -G "${OUTPUT}/step_*/train_state.pt" >/dev/null; then
    echo "Cannot resume: no complete training checkpoint in ${OUTPUT}" >&2
    exit 1
  fi
elif [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT}" >&2
  exit 1
fi
if pgrep -af 'evaluate_pi05_camera_views.py|evaluate_pi05_three_tasks_camera_views.py|policy_server|deploy_policy.py|train_pi05_view_feature_adapter.py' >/dev/null; then
  echo "Refusing to start while PI0.5 evaluation/server/training is running:" >&2
  pgrep -af 'evaluate_pi05_camera_views.py|evaluate_pi05_three_tasks_camera_views.py|policy_server|deploy_policy.py|train_pi05_view_feature_adapter.py' >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${ROOT}"
RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi
nohup setsid env \
  PYTHONPATH="${RUNTIME_SRC}" \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  PYTHONNOUSERSITE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" -u script/train_pi05_view_feature_adapter.py \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --cache-index "${TRAIN_INDEX}" \
    --output "${OUTPUT}" \
    --shifted-views c1,c2,c3,c4,c5,c6 \
    --adapter-variant legacy \
    --batch-size 1 \
    --max-steps "${MAX_STEPS}" \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --grad-clip-norm 1.0 \
    --flow-loss-weight 0.05 \
    --velocity-loss-weight 1.0 \
    --global-feature-loss-weight 0.05 \
    --canonical-identity-loss-weight 0.2 \
    --canonical-velocity-loss-weight 1.0 \
    --residual-loss-weight 0.01 \
    --save-every 250 \
    --log-every 10 \
    --seed 42 \
    "${RESUME_ARGS[@]}" \
    >> "${LOG}" 2>&1 </dev/null &

PID=$!
echo "${PID}" > "${PID_FILE}"
cat <<EOF
PID=${PID}
RUN_NAME=${RUN_NAME}
LOG=${LOG}
OUTPUT=${OUTPUT}
MAX_STEPS=${MAX_STEPS}
RESUME=${RESUME}
TRAIN_INDEX=${TRAIN_INDEX}
VAL_INDEX=${VAL_INDEX}
EOF
