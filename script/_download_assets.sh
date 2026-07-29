#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${ROBOTWIN_PYTHON:-python}"

cd "${ROOT}/assets"
"${PYTHON_BIN}" _download.py

unzip -o background_texture.zip
unzip -o embodiments.zip
unzip -o objects.zip
rm -f background_texture.zip embodiments.zip objects.zip

cd "${ROOT}"
echo "Configuring embodiment asset paths ..."
"${PYTHON_BIN}" script/update_embodiment_config_path.py
