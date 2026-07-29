#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${ROBOTWIN_PYTHON:-python}"
CUROBO_DIR="${ROBOTWIN_CUROBO_DIR:-${ROOT}/envs/curobo}"
cd "${ROOT}"

echo "Checking the user-selected PyTorch installation ..."
"${PYTHON_BIN}" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"RoboTwin requires Python 3.10, got {sys.version.split()[0]}")

try:
    import torch
    import torchvision
except ImportError as error:
    raise SystemExit(
        "Install a GPU-compatible torch and torchvision before running script/_install.sh"
    ) from error

print(f"Python {sys.version.split()[0]}")
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
print(f"torch CUDA runtime: {torch.version.cuda}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, capability: {torch.cuda.get_device_capability(0)}")
PY

echo "Installing RoboTwin Python dependencies without replacing PyTorch ..."
"${PYTHON_BIN}" -m pip install -r script/requirements.txt
"${PYTHON_BIN}" -m pip install warp-lang==1.12.0 setuptools==69.5.1

if [[ "${ROBOTWIN_REINSTALL_PYTORCH3D:-0}" == "1" ]] \
  || ! "${PYTHON_BIN}" -c 'import pytorch3d' >/dev/null 2>&1; then
  echo "Installing pytorch3d against the selected PyTorch ..."
  "${PYTHON_BIN}" -m pip install \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    --no-build-isolation
else
  echo "pytorch3d is already importable; set ROBOTWIN_REINSTALL_PYTORCH3D=1 to rebuild it."
fi

echo "Applying the SAPIEN URDF loader compatibility patch ..."
SAPIEN_LOCATION="$("${PYTHON_BIN}" -c 'from pathlib import Path; import sapien; print(Path(sapien.__file__).parent)')"
URDF_LOADER="${SAPIEN_LOCATION}/wrapper/urdf_loader.py"
if [[ ! -f "${URDF_LOADER}" ]]; then
  echo "SAPIEN URDF loader does not exist: ${URDF_LOADER}" >&2
  exit 1
fi
sed -i -E 's/(open\([^,]+, )"r"\)/\1"r", encoding="utf-8"\)/g' "${URDF_LOADER}"
sed -i -E 's/(urdf_file\[:-4\] \+ )"srdf"/\1".srdf"/g' "${URDF_LOADER}"

echo "Applying the MPLib screw-planner compatibility patch ..."
MPLIB_LOCATION="$("${PYTHON_BIN}" -c 'from pathlib import Path; import mplib; print(Path(mplib.__file__).parent)')"
PLANNER="${MPLIB_LOCATION}/planner.py"
if [[ ! -f "${PLANNER}" ]]; then
  echo "MPLib planner does not exist: ${PLANNER}" >&2
  exit 1
fi
sed -i -E \
  's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' \
  "${PLANNER}"

echo "Installing CuRobo v0.7.8 ..."
if [[ ! -e "${CUROBO_DIR}" ]]; then
  git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "${CUROBO_DIR}"
elif [[ ! -f "${CUROBO_DIR}/pyproject.toml" ]]; then
  echo "Existing CuRobo directory is incomplete: ${CUROBO_DIR}" >&2
  exit 1
fi
"${PYTHON_BIN}" -m pip install -e "${CUROBO_DIR}" --no-build-isolation

"${PYTHON_BIN}" - <<'PY'
import grpc
import curobo
import mplib
import pytorch3d
import sapien
import torch
import warp

print("Verified RoboTwin imports")
print(f"torch={torch.__version__}, curobo={getattr(curobo, '__version__', '0.7.8')}")
PY

echo "RoboTwin environment installation complete."
echo "Next: run 'bash script/_download_assets.sh' from this environment."
