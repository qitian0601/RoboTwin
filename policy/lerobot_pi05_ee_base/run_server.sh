#!/usr/bin/env bash
set -euo pipefail

# The EE and joint policies use the same Chenglong LeRobot policy server.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../lerobot_pi05/run_server.sh" "$@"
