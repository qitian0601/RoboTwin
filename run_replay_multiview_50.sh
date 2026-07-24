#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONNOUSERSITE=1

exec "${ROBOTWIN_PYTHON:-python}" -u \
  script/replay_multiview_dataset.py "$@"
