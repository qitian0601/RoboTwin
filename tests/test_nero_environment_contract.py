from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_robotwin_requirements_do_not_override_gpu_pytorch() -> None:
    requirements = (ROOT / "script/requirements.txt").read_text(encoding="utf-8").splitlines()
    package_names = {
        line.split("==", 1)[0].strip().lower()
        for line in requirements
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "torch" not in package_names
    assert "torchvision" not in package_names


def test_lerobot_environment_contains_conversion_and_test_dependencies() -> None:
    environment = (ROOT / "environment/lerobot-nero.yml").read_text(encoding="utf-8")
    assert "h5py==" in environment
    assert "pytest" in environment
    assert "torch==" in environment
    assert "torchvision==" in environment


def test_converter_uses_bundled_runtime_and_documented_environment() -> None:
    converter = (ROOT / "tools/convert_nero_task_lerobot_v3.sh").read_text(encoding="utf-8")
    assert "third_party/lerobot_nero_runtime/src" in converter
    assert 'ROBOTWIN_LEROBOT_ENV:-lerobot-nero' in converter
    assert "export PYTHONNOUSERSITE=1" in converter
    assert "../lerobot/src" not in converter
    tool_scripts = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "tools").glob("*.sh")
    )
    assert 'ROBOTWIN_LEROBOT_ENV:-lerobot}' not in tool_scripts


def test_installer_requires_preselected_torch_without_installing_it() -> None:
    installer = (ROOT / "script/_install.sh").read_text(encoding="utf-8")
    assert "import torch" in installer
    assert "pip install torch" not in installer
    assert "script/requirements.txt" in installer


def test_asset_downloader_is_repo_relative_and_fail_fast() -> None:
    downloader = (ROOT / "script/_download_assets.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in downloader
    assert 'dirname "$0"' in downloader
    assert "script/update_embodiment_config_path.py" in downloader
