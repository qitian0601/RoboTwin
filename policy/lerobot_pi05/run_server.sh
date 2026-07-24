#!/usr/bin/env bash
set -euo pipefail

# Run the PI0.5 server from a configurable Python and source tree.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"

SERVER_PYTHON="${ROBOTWIN_PI05_SERVER_PYTHON:-${ROBOTWIN_PYTHON:-python}}"
SERVER_SRC="${ROBOTWIN_PI05_SERVER_SRC:-${ROOT}/third_party/lerobot_nero_runtime/src}"

if [[ ! -x "${SERVER_PYTHON}" ]]; then
  echo "PI0.5 server Python is not executable: ${SERVER_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${SERVER_SRC}/lerobot" ]]; then
  echo "PI0.5 server source does not exist: ${SERVER_SRC}" >&2
  exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export PYTHONPATH="${SERVER_SRC}"

"${SERVER_PYTHON}" -m lerobot.async_inference.policy_server \
  --host="${ROBOTWIN_PI05_HOST:-0.0.0.0}" \
  --port="${ROBOTWIN_PI05_PORT:-8081}" \
  --fps="${ROBOTWIN_PI05_FPS:-30}" \
  --inference_latency="${ROBOTWIN_PI05_INFERENCE_LATENCY:-0.033333}" \
  --obs_queue_timeout="${ROBOTWIN_PI05_OBS_TIMEOUT:-300}" \
  --obs_similarity_atol="${ROBOTWIN_PI05_OBS_SIMILARITY_ATOL:-0.09}" \
  --async_rtc.enabled="${ROBOTWIN_PI05_ASYNC_RTC:-false}"
