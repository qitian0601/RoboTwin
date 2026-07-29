"""Regression tests for PI0.5 server/client action-space compatibility."""

import json
from pathlib import Path

import numpy as np

from policy.lerobot_pi05.deploy_policy import (
    _checkpoint_returns_absolute_actions,
    _policy_action_to_absolute_bus,
)


def _write_postprocessor(checkpoint: Path, *, absolute: bool) -> None:
    checkpoint.mkdir()
    steps = [
        {
            "registry_name": "unnormalizer_processor",
            "config": {},
        }
    ]
    if absolute:
        steps.append(
            {
                "registry_name": "absolute_actions_processor",
                "config": {"enabled": True},
            }
        )
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps({"name": "policy_postprocessor", "steps": steps}),
        encoding="utf-8",
    )


def test_old_export_keeps_relative_contract(tmp_path):
    checkpoint = tmp_path / "old"
    _write_postprocessor(checkpoint, absolute=False)
    assert not _checkpoint_returns_absolute_actions(checkpoint)


def test_new_export_detects_absolute_contract(tmp_path):
    checkpoint = tmp_path / "new"
    _write_postprocessor(checkpoint, absolute=True)
    assert _checkpoint_returns_absolute_actions(checkpoint)


def test_client_does_not_double_add_absolute_action():
    q = np.arange(16, dtype=np.float32)
    delta = np.full(16, 0.25, dtype=np.float32)
    server_absolute = q + delta
    # New checkpoint: the server already returned q + delta.
    client_target = _policy_action_to_absolute_bus(
        server_absolute,
        q,
        use_relative_actions=True,
        server_returns_absolute_actions=True,
    )
    assert np.allclose(client_target, q + delta)
    assert not np.allclose(client_target, q + server_absolute)


def test_old_relative_contract_adds_only_arm_deltas():
    q = np.arange(16, dtype=np.float32)
    prediction = np.full(16, 0.25, dtype=np.float32)
    client_target = _policy_action_to_absolute_bus(
        prediction,
        q,
        use_relative_actions=True,
        server_returns_absolute_actions=False,
    )
    assert np.allclose(client_target[:14], q[:14] + prediction[:14])
    assert np.allclose(client_target[14:16], prediction[14:16])


def test_disabled_absolute_processor_keeps_legacy_contract(tmp_path):
    checkpoint = tmp_path / "disabled"
    checkpoint.mkdir()
    (checkpoint / "policy_postprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "absolute_actions_processor",
                        "config": {"enabled": False},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert not _checkpoint_returns_absolute_actions(checkpoint)
