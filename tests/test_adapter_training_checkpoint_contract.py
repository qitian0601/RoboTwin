from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from script.train_pi05_view_feature_adapter import (
    make_deployment_checkpoint,
    newest_resume_checkpoint,
    save_training_checkpoint,
)


def _adapter_files(path: Path) -> None:
    (path / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")


def test_resume_uses_newest_complete_step_and_ignores_partial_directory(tmp_path: Path) -> None:
    older = tmp_path / "step_000010"
    newer = tmp_path / "step_000020"
    partial = tmp_path / "step_000030"
    for path in (older, newer, partial):
        path.mkdir()
    _adapter_files(older)
    _adapter_files(newer)
    torch.save({"step": 10}, older / "train_state.pt")
    torch.save({"step": 20}, newer / "train_state.pt")
    (partial / "adapter_model.safetensors").write_bytes(b"partial")

    step, checkpoint, state = newest_resume_checkpoint(tmp_path)

    assert step == 20
    assert checkpoint == newer
    assert state["step"] == 20


def test_resume_ignores_state_directory_step_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step_000020"
    checkpoint.mkdir()
    _adapter_files(checkpoint)
    torch.save({"step": 19}, checkpoint / "train_state.pt")
    assert newest_resume_checkpoint(tmp_path) is None


def test_resume_falls_back_from_corrupted_newest_state(tmp_path: Path) -> None:
    older = tmp_path / "step_000010"
    corrupted = tmp_path / "step_000020"
    older.mkdir()
    corrupted.mkdir()
    _adapter_files(older)
    _adapter_files(corrupted)
    torch.save({"step": 10}, older / "train_state.pt")
    (corrupted / "train_state.pt").write_bytes(b"not a torch checkpoint")

    step, checkpoint, state = newest_resume_checkpoint(tmp_path)

    assert (step, checkpoint, state["step"]) == (10, older, 10)


def test_training_checkpoint_directory_is_committed_atomically(tmp_path: Path) -> None:
    class FakePolicy:
        @staticmethod
        def save_feature_adapter(path: Path) -> None:
            path.mkdir()
            _adapter_files(path)

    class FakeOptimizer:
        @staticmethod
        def state_dict() -> dict:
            return {"optimizer": "state"}

    dataset = SimpleNamespace(rng=random.Random(7))
    generator = torch.Generator().manual_seed(7)

    checkpoint = save_training_checkpoint(
        FakePolicy(),
        FakeOptimizer(),
        dataset,
        generator,
        tmp_path,
        250,
        {"variant": "test"},
    )

    assert checkpoint == tmp_path / "step_000250"
    assert (checkpoint / "train_state.pt").is_file()
    assert not list(tmp_path.glob(".step_*.tmp"))


def test_deployment_is_atomic_versioned_and_never_overwritten(tmp_path: Path) -> None:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    output = tmp_path / "output"
    base.mkdir()
    adapter.mkdir()
    output.mkdir()
    (base / "config.json").write_text(
        json.dumps({"type": "pi05", "use_feature_adapter": False}), encoding="utf-8"
    )
    (base / "model.safetensors").write_bytes(b"frozen-model")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "train_state.pt").write_bytes(b"optimizer-must-not-deploy")
    config = SimpleNamespace(
        feature_adapter_variant="multi_scale_dynamic",
        feature_adapter_bottleneck_dim=256,
        feature_adapter_num_heads=8,
        feature_adapter_num_blocks=2,
        feature_adapter_ffn_expansion=4,
        feature_adapter_num_experts=4,
        feature_adapter_pose_enabled=False,
        feature_adapter_pose_dim=9,
        feature_adapter_flow_loss_weight=0.0,
        feature_adapter_velocity_loss_weight=1.0,
        feature_adapter_global_feature_loss_weight=0.05,
        feature_adapter_canonical_identity_loss_weight=0.2,
        feature_adapter_canonical_velocity_loss_weight=1.0,
        feature_adapter_residual_loss_weight=0.01,
    )

    deployment = make_deployment_checkpoint(base, adapter, output, config)

    deployed_config = json.loads((deployment / "config.json").read_text(encoding="utf-8"))
    assert deployed_config["feature_adapter_variant"] == "multi_scale_dynamic"
    assert (deployment / "model.safetensors").is_symlink()
    assert (deployment / "feature_adapter" / "adapter_model.safetensors").read_bytes() == b"adapter"
    assert not (deployment / "feature_adapter" / "train_state.pt").exists()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        make_deployment_checkpoint(base, adapter, output, config)
