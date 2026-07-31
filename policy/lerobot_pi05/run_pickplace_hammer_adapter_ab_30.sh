#!/usr/bin/env bash
set -Eeuo pipefail

# Serialized, resumable 24k-base vs 4.5k-Adapter evaluation. Only one PI0.5
# server and one simulator client are allowed to own the GPU at any time.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ID="${ROBOTWIN_ADAPTER_AB_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${ROBOTWIN_ADAPTER_AB_OUTPUT_ROOT:-${ROOT}/outputs/pickplace_hammer_adapter_ab_30/${RUN_ID}}"
LOG_DIR="${ROBOTWIN_ADAPTER_AB_LOG_DIR:-${ROOT}/logs/pickplace_hammer_adapter_ab_30/${RUN_ID}}"
BASE_POLICY="${ROBOTWIN_ADAPTER_AB_BASE_POLICY:-${ROBOTWIN_DUAL_TASK_BASE_POLICY:-${ROOT}/../lerobot/outputs/train/three_task_sim/pickplace_hammer_hold3s/checkpoints/024000/pretrained_model}}"
ADAPTER_POLICY="${ROBOTWIN_ADAPTER_AB_ADAPTER_POLICY:-${ROOT}/outputs/pi05_feature_adapter/pickplace_hammer_offline_selection_20260730/selected_offline}"
EPISODES="${ROBOTWIN_ADAPTER_AB_EPISODES:-30}"
SCENARIO_SEED="${ROBOTWIN_ADAPTER_AB_SCENARIO_SEED:-0}"
POLICY_SEED="${ROBOTWIN_ADAPTER_AB_POLICY_SEED:-123}"
SERVER_HOST="${ROBOTWIN_PI05_HOST:-127.0.0.1}"
SERVER_PORT="${ROBOTWIN_PI05_PORT:-8081}"
SERVER_RUNNER="${ROOT}/policy/lerobot_pi05/run_server.sh"
EVALUATOR="${ROOT}/script/evaluate_pi05_three_tasks_camera_views.py"
SUMMARIZER="${ROOT}/script/summarize_pi05_adapter_ab_eval.py"
SERVER_PYTHON="${ROBOTWIN_PI05_SERVER_PYTHON:-python}"
CLIENT_PYTHON="${ROBOTWIN_PYTHON:-python}"
SERVER_SRC="${ROBOTWIN_PI05_SERVER_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}"
CLIENT_SRC="${ROBOTWIN_LEROBOT_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}"
LOCK_FILE="${ROBOTWIN_ADAPTER_AB_LOCK_FILE:-${ROOT}/logs/.pi05_gpu.lock}"

BASE_RUN="${OUTPUT_ROOT}/base_step024000"
ADAPTER_RUN="${OUTPUT_ROOT}/adapter_step004500"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}" "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "another PI0.5 evaluation owns ${LOCK_FILE}" >&2
  exit 73
fi

for path in "${BASE_POLICY}" "${ADAPTER_POLICY}"; do
  [[ -d "${path}" ]] || { echo "missing policy: ${path}" >&2; exit 66; }
done

SERVER_PID=""
SERVER_PGID=""
CLIENT_PID=""
CLIENT_PGID=""
HEARTBEAT_PID=""
HEARTBEAT_PGID=""

group_has_live_processes() {
  ps -eo pgid=,stat= | awk -v target="$1" \
    '$1 == target && $2 !~ /^Z/ { found = 1 } END { exit(found ? 0 : 1) }'
}

cleanup_group() {
  local pgid="$1" pid="$2"
  if [[ -n "${pgid}" ]] && group_has_live_processes "${pgid}"; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      group_has_live_processes "${pgid}" || break
      sleep 0.5
    done
    group_has_live_processes "${pgid}" && kill -KILL -- "-${pgid}" 2>/dev/null || true
  fi
  [[ -z "${pid}" ]] || wait "${pid}" 2>/dev/null || true
}

cleanup_server() {
  cleanup_group "${SERVER_PGID}" "${SERVER_PID}"
  SERVER_PID=""
  SERVER_PGID=""
}

cleanup_client() {
  cleanup_group "${CLIENT_PGID}" "${CLIENT_PID}"
  CLIENT_PID=""
  CLIENT_PGID=""
}

cleanup_all() {
  cleanup_client
  cleanup_server
  cleanup_group "${HEARTBEAT_PGID}" "${HEARTBEAT_PID}"
}
trap cleanup_all EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

set_status() {
  printf '%s\n' "$1" > "${LOG_DIR}/status"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export ROBOTWIN_PI05_SERVER_PYTHON="${SERVER_PYTHON}"
export ROBOTWIN_PI05_SERVER_SRC="${SERVER_SRC}"
export ROBOTWIN_PYTHON="${CLIENT_PYTHON}"
export ROBOTWIN_LEROBOT_SRC="${CLIENT_SRC}"
export ROBOTWIN_PI05_HOST="${SERVER_HOST}"
export ROBOTWIN_PI05_PORT="${SERVER_PORT}"
export PYTHONPATH="${ROOT}:${CLIENT_SRC}${ROBOTWIN_CUROBO_SRC:+:${ROBOTWIN_CUROBO_SRC}}:${PYTHONPATH:-}"

# Fail before loading the 9.35 GB policy or scanning expert seeds when the
# client/server transport dependencies are unavailable.
PYTHONPATH="${ROOT}/policy:${PYTHONPATH}" "${CLIENT_PYTHON}" -c \
  'import lerobot_pi05.deploy_policy; import grpc' >/dev/null

MANIFEST="${LOG_DIR}/manifest.env"
REQUESTED="${LOG_DIR}/.manifest.env.$$"
printf 'run_id=%s\noutput_root=%s\nbase_policy=%s\nadapter_policy=%s\nepisodes=%s\nscenario_seed=%s\npolicy_seed=%s\nhammer_profile=training_support_contact\nrecording=failures_only\n' \
  "${RUN_ID}" "${OUTPUT_ROOT}" "$(realpath "${BASE_POLICY}")" \
  "$(realpath "${ADAPTER_POLICY}")" "${EPISODES}" "${SCENARIO_SEED}" "${POLICY_SEED}" \
  > "${REQUESTED}"
if [[ -f "${MANIFEST}" ]]; then
  cmp -s "${REQUESTED}" "${MANIFEST}" || {
    echo "resume manifest mismatch: ${MANIFEST}" >&2
    rm -f "${REQUESTED}"
    exit 78
  }
  rm -f "${REQUESTED}"
else
  mv "${REQUESTED}" "${MANIFEST}"
fi

printf 'pid=%s\npgid=%s\n' "$$" "$(ps -o pgid= -p $$ | tr -d ' ')" > "${LOG_DIR}/supervisor.pid"
echo "$(date -Is) START_OR_RESUME output=${OUTPUT_ROOT}" | tee -a "${LOG_DIR}/supervisor.log"

setsid env HEARTBEAT_DIR="${LOG_DIR}" bash -c '
  while true; do
    printf "%s %s\n" "$(date -Is)" "$(cat "${HEARTBEAT_DIR}/status" 2>/dev/null || echo starting)" \
      > "${HEARTBEAT_DIR}/heartbeat"
    sleep 30
  done
' >/dev/null 2>&1 &
HEARTBEAT_PID=$!
HEARTBEAT_PGID="${HEARTBEAT_PID}"

wait_for_server() {
  for _ in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      return 1
    fi
    if (exec 3<>"/dev/tcp/${SERVER_HOST}/${SERVER_PORT}") 2>/dev/null; then
      exec 3>&-
      return 0
    fi
    sleep 1
  done
  return 1
}

start_server() {
  local label="$1" policy="$2"
  export ROBOTWIN_PI05_POLICY_PATH="${policy}"
  set_status "${label}: loading policy server"
  setsid "${SERVER_RUNNER}" >>"${LOG_DIR}/${label}_server.log" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID="${SERVER_PID}"
  printf 'pid=%s\npgid=%s\n' "${SERVER_PID}" "${SERVER_PGID}" > "${LOG_DIR}/${label}_server.pid"
  if ! wait_for_server; then
    echo "$(date -Is) ${label}: policy server failed" | tee -a "${LOG_DIR}/supervisor.log"
    cleanup_server
    return 1
  fi
}

run_eval() {
  local label="$1" policy="$2" run_dir="$3" seed_run="${4:-}"
  local seed_args=()
  [[ -z "${seed_run}" ]] || seed_args+=(--scenario-seeds-from-run "${seed_run}")
  mkdir -p "$(dirname "${run_dir}")"
  start_server "${label}" "${policy}"
  set_status "${label}: evaluating C0-C6"
  echo "$(date -Is) ${label}: evaluation start" | tee -a "${LOG_DIR}/supervisor.log"
  set +e
  setsid "${CLIENT_PYTHON}" -u "${EVALUATOR}" \
    --policy-path "${policy}" \
    --server-address "${SERVER_HOST}:${SERVER_PORT}" \
    --episodes-per-view "${EPISODES}" \
    --seed "${SCENARIO_SEED}" \
    --policy-inference-seed "${POLICY_SEED}" \
    --output-dir "$(dirname "${run_dir}")" \
    --run-name "$(basename "${run_dir}")" \
    --resume-existing \
    --task place_two_cubes_box \
    --task beat_block_hammer \
    --hammer-eval-profile training_support_contact \
    --record-video \
    --record-failures-only \
    --video-stride 3 \
    --video-width 640 \
    --video-height 400 \
    --video-crf 30 \
    --video-preset ultrafast \
    --no-continue-on-error \
    "${seed_args[@]}" \
    >>"${LOG_DIR}/${label}_client.log" 2>&1 &
  CLIENT_PID=$!
  CLIENT_PGID="${CLIENT_PID}"
  printf 'pid=%s\npgid=%s\n' "${CLIENT_PID}" "${CLIENT_PGID}" > "${LOG_DIR}/${label}_client.pid"
  wait "${CLIENT_PID}"
  local rc=$?
  set -e
  CLIENT_PID=""
  CLIENT_PGID=""
  cleanup_server
  echo "$(date -Is) ${label}: evaluation returncode=${rc}" | tee -a "${LOG_DIR}/supervisor.log"
  return "${rc}"
}

set_status "base_step024000: pending"
run_eval base_step024000 "${BASE_POLICY}" "${BASE_RUN}"
"${CLIENT_PYTHON}" "${SUMMARIZER}" \
  --base-run "${BASE_RUN}" --adapter-run "${BASE_RUN}" \
  --episodes-per-view "${EPISODES}" --output-dir "${OUTPUT_ROOT}" --validate-only

set_status "adapter_step004500: pending"
run_eval adapter_step004500 "${ADAPTER_POLICY}" "${ADAPTER_RUN}" "${BASE_RUN}"
"${CLIENT_PYTHON}" "${SUMMARIZER}" \
  --base-run "${BASE_RUN}" --adapter-run "${ADAPTER_RUN}" \
  --episodes-per-view "${EPISODES}" --output-dir "${OUTPUT_ROOT}"

set_status "complete"
echo "$(date -Is) COMPLETE report=${OUTPUT_ROOT}/REPORT.md" | tee -a "${LOG_DIR}/supervisor.log"
