#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/logs/adapter_multiview"
mkdir -p "${LOG_DIR}"

if [[ -f "${LOG_DIR}/latest.pid" ]]; then
  previous_pid="$(cat "${LOG_DIR}/latest.pid")"
  if kill -0 "${previous_pid}" 2>/dev/null; then
    echo "Collection is already running with PID ${previous_pid}" >&2
    exit 1
  fi
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="${LOG_DIR}/collection_${timestamp}.log"
nohup setsid "${ROOT}/run_replay_adapter_multiview_50.sh" "$@" \
  >"${log_path}" 2>&1 </dev/null &
pid=$!
echo "${pid}" >"${LOG_DIR}/latest.pid"
ln -sfn "$(basename "${log_path}")" "${LOG_DIR}/latest.log"

echo "Started adapter multiview collection"
echo "PID: ${pid}"
echo "Log: ${log_path}"
