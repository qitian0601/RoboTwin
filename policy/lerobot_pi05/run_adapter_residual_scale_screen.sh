#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCREEN_ROOT="${ROBOTWIN_ADAPTER_SCALE_SCREEN_ROOT:-${ROOT}/outputs/pi05_feature_adapter/pickplace_hammer_residual_scale_20260731}"
REFERENCE_RUN="${ROBOTWIN_ADAPTER_SCALE_REFERENCE_RUN:-${ROOT}/outputs/pickplace_hammer_adapter_ab_30/20260730_192417}"
LOG_DIR="${ROBOTWIN_ADAPTER_SCALE_LOG_DIR:-${ROOT}/logs/feature_adapter/residual_scale_screen_20260731}"
SERVER_RUNNER="${ROOT}/policy/lerobot_pi05/run_server.sh"
EVALUATOR="${ROOT}/script/evaluate_pi05_camera_views.py"
SUMMARIZER="${ROOT}/script/summarize_adapter_residual_scale_screen.py"
CLIENT_PYTHON="${ROBOTWIN_PYTHON:-python}"
SERVER_PYTHON="${ROBOTWIN_PI05_SERVER_PYTHON:-python}"
RUNTIME_SRC="${ROBOTWIN_PI05_SERVER_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}"
LOCK_FILE="${ROOT}/logs/.pi05_gpu.lock"
mkdir -p "${LOG_DIR}" "$(dirname "${LOCK_FILE}")"
exec 9>"${LOCK_FILE}"
flock -n 9 || { echo "another PI0.5 job owns ${LOCK_FILE}" >&2; exit 73; }

SERVER_PID=""
SERVER_PGID=""
cleanup_server() {
  if [[ -n "${SERVER_PGID}" ]]; then
    kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      ps -eo pgid=,stat= | awk -v p="${SERVER_PGID}" '$1==p && $2!~/^Z/{x=1} END{exit(x?0:1)}' || break
      sleep 0.5
    done
    kill -KILL -- "-${SERVER_PGID}" 2>/dev/null || true
  fi
  [[ -z "${SERVER_PID}" ]] || wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
  SERVER_PGID=""
}
trap cleanup_server EXIT INT TERM

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
export ROBOTWIN_PI05_SERVER_PYTHON="${SERVER_PYTHON}"
export ROBOTWIN_PI05_SERVER_SRC="${RUNTIME_SRC}"
export ROBOTWIN_LEROBOT_SRC="${RUNTIME_SRC}"
export ROBOTWIN_PI05_HOST=127.0.0.1 ROBOTWIN_PI05_PORT=8081
export PYTHONPATH="${ROOT}:${ROOT}/policy:${RUNTIME_SRC}${ROBOTWIN_CUROBO_SRC:+:${ROBOTWIN_CUROBO_SRC}}:${PYTHONPATH:-}"

wait_for_server() {
  for _ in $(seq 1 240); do
    kill -0 "${SERVER_PID}" 2>/dev/null || return 1
    (exec 3<>/dev/tcp/127.0.0.1/8081) 2>/dev/null && { exec 3>&-; return 0; }
    sleep 1
  done
  return 1
}

run_case() {
  local label="$1" policy="$2" task="$3" config="$4" view="$5" seeds="$6" start="$7"
  local run_dir="${SCREEN_ROOT}/${label}/${task}/${view}"
  local resume=()
  [[ -f "${run_dir}/results.json" ]] && resume+=(--resume-existing)
  local hammer=()
  if [[ "${task}" == "beat_block_hammer" ]]; then
    hammer+=(--hammer-arm-aware-instruction --hammer-contact-success --hammer-training-support-range)
    hammer+=(--actions-per-chunk 50 --chunk-size-threshold 0.2 --action-merge-new-weight 1.0)
  fi
  "${CLIENT_PYTHON}" -u "${EVALUATOR}" \
    --policy-path "${policy}" --server-address 127.0.0.1:8081 \
    --task-name "${task}" --task-config "${config}" --view "${view}" \
    --episodes-per-view 5 --scenario-seeds-file "${seeds}" \
    --scenario-episode-index "${start}" --policy-inference-seed 123 \
    --instruction "$( [[ "${task}" == "place_two_cubes_box" ]] && echo 'Put the yellow cube into the black box first, then put the green cube into the black box.' || echo 'Take the hammer in the right gripper and strike the block.' )" \
    --run-dir "${run_dir}" "${resume[@]}" "${hammer[@]}"
}

printf 'pid=%s\n' "$$" > "${LOG_DIR}/pid"
for label in alpha_0p10 alpha_0p25 alpha_0p50; do
  policy="${SCREEN_ROOT}/${label}"
  printf '%s %s loading\n' "$(date -Is)" "${label}" | tee "${LOG_DIR}/status"
  export ROBOTWIN_PI05_POLICY_PATH="${policy}"
  setsid "${SERVER_RUNNER}" >>"${LOG_DIR}/${label}_server.log" 2>&1 &
  SERVER_PID=$!
  SERVER_PGID="${SERVER_PID}"
  wait_for_server
  {
    run_case "${label}" "${policy}" place_two_cubes_box demo_nero_two_cubes C1 "${SCREEN_ROOT}/pickplace_seeds.json" 14
    run_case "${label}" "${policy}" place_two_cubes_box demo_nero_two_cubes C4 "${SCREEN_ROOT}/pickplace_seeds.json" 14
    run_case "${label}" "${policy}" beat_block_hammer demo_nero_beat_block_hammer_slow120 C3 "${SCREEN_ROOT}/hammer_seeds.json" 0
    run_case "${label}" "${policy}" beat_block_hammer demo_nero_beat_block_hammer_slow120 C5 "${SCREEN_ROOT}/hammer_seeds.json" 0
  } >>"${LOG_DIR}/${label}_client.log" 2>&1
  cleanup_server
  printf '%s %s complete\n' "$(date -Is)" "${label}" | tee "${LOG_DIR}/status"
done

"${CLIENT_PYTHON}" "${SUMMARIZER}" --screen-root "${SCREEN_ROOT}" --reference-run "${REFERENCE_RUN}"
printf '%s complete report=%s\n' "$(date -Is)" "${SCREEN_ROOT}/SCREEN_REPORT.md" | tee "${LOG_DIR}/status"
