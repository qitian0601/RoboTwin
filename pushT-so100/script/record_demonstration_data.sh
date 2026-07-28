#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONNOUSERSITE=1

python src/env_human_ee.py \
  --repo_id ./data/nero-manipulation-dataset \
  --fps 10 \
  --move_speed 0.05 \
  --rot_speed 1.0
