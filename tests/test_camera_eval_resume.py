from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import pytest

from script.camera_eval_resume import (
    load_or_create_results,
    write_results,
)


@dataclass(frozen=True)
class _ViewSpec:
    identifier: str
    kind: str
    value: float
    description: str


VIEW_SPECS = (
    _ViewSpec("C0", "canonical", 0.0, "canonical view"),
    _ViewSpec("C1", "yaw", -10.0, "yaw -10 deg"),
    _ViewSpec("C2", "yaw", 10.0, "yaw +10 deg"),
    _ViewSpec("C3", "pitch", -7.0, "pitch -7 deg"),
    _ViewSpec("C4", "pitch", 7.0, "pitch +7 deg"),
    _ViewSpec("C5", "translation", 0.10, "move forward 10 cm"),
    _ViewSpec("C6", "translation", -0.10, "move backward 10 cm"),
)


def _expected() -> dict:
    return {
        "task_name": "place_two_cubes_box",
        "task_config": "demo_nero_two_cubes",
        "policy_path": "/checkpoint/stable",
        "server_address": "127.0.0.1:8081",
        "instruction": "exact prompt",
        "episodes_per_view": 2,
        "max_steps": None,
        "shared_valid_seeds": [100000, 100001],
        "policy_inference_seed": 123,
        "adapter_c0_bypassed": True,
        "seed_search_index": 0,
        "scenario_episode_index": None,
        "scenario_sampling_range": {
            "kind": "observed_slow120_training_support",
            "left_x_m": [-0.25, -0.20],
            "right_x_m": [0.18, 0.24],
            "y_m": [-0.18, -0.15],
        },
        "success_criterion": {
            "kind": "any_hammer_block_contact",
            "hold_s": 0.0,
        },
        "control_config": {
            "fps": 30,
            "actions_per_chunk": 50,
            "chunk_size_threshold": 0.8,
            "action_merge_new_weight": 0.5,
            "max_policy_step_rad": 0.05,
            "max_gripper_step_m": 0.05,
            "max_executor_step_rad": 0.005,
            "max_executor_gripper_step_m": 0.004,
        },
        "recording_config": {
            "enabled": False,
            "video_stride": 3,
            "video_width": 640,
            "video_height": 400,
            "video_crf": 28,
            "video_preset": "ultrafast",
        },
        "instruction_by_arm": None,
        "trace_dir": None,
    }


def _partial_payload() -> dict:
    return {
        **_expected(),
        "resume_contract_version": 1,
        "views": [
            {
                "id": "C0",
                "kind": "canonical",
                "value": 0.0,
                "description": "canonical view",
                "successes": 1,
                "episodes": 1,
                "success_rate": 1.0,
                "complete": False,
                "head_camera": {
                    "position": [0.0, 0.0, 0.0],
                    "forward": [1.0, 0.0, 0.0],
                    "left": [0.0, 1.0, 0.0],
                },
                "episode_results": [
                    {
                        "seed": 100000,
                        "success": True,
                        "error": None,
                        "video": None,
                    }
                ],
            }
        ],
    }


def test_resume_keeps_partial_episode_checkpoint(tmp_path: Path) -> None:
    payload = _partial_payload()
    write_results(tmp_path, payload)

    resumed = load_or_create_results(
        tmp_path,
        _expected(),
        resume_existing=True,
        view_specs=VIEW_SPECS,
    )

    assert resumed["views"][0]["episode_results"] == payload["views"][0]["episode_results"]
    assert resumed["views"][0]["complete"] is False
    assert not list(tmp_path.glob(".*.tmp"))


def test_legacy_complete_view_is_upgraded_without_rerun(tmp_path: Path) -> None:
    payload = _partial_payload()
    payload.pop("seed_search_index")
    payload.pop("control_config")
    payload.pop("recording_config")
    payload.pop("resume_contract_version")
    view = payload["views"][0]
    view.pop("complete")
    view["episode_results"].append(
        {"seed": 100001, "success": False, "error": None, "video": None}
    )
    view["successes"] = 1
    view["episodes"] = 2
    view["success_rate"] = 0.5
    write_results(tmp_path, payload)

    resumed = load_or_create_results(
        tmp_path,
        _expected(),
        resume_existing=True,
        view_specs=VIEW_SPECS,
    )

    assert resumed["resume_contract_version"] == 1
    assert resumed["control_config"] == _expected()["control_config"]
    assert resumed["views"][0]["complete"] is True


def test_resume_rejects_policy_seed_or_scenario_seed_mismatch(tmp_path: Path) -> None:
    write_results(tmp_path, _partial_payload())

    wrong_policy_seed = copy.deepcopy(_expected())
    wrong_policy_seed["policy_inference_seed"] = 124
    with pytest.raises(ValueError, match="policy_inference_seed"):
        load_or_create_results(
            tmp_path, wrong_policy_seed, resume_existing=True, view_specs=VIEW_SPECS
        )

    wrong_scenario_seed = copy.deepcopy(_expected())
    wrong_scenario_seed["shared_valid_seeds"] = [100000, 100002]
    with pytest.raises(ValueError, match="shared_valid_seeds"):
        load_or_create_results(
            tmp_path, wrong_scenario_seed, resume_existing=True, view_specs=VIEW_SPECS
        )


def test_resume_rejects_hammer_range_or_success_criterion_mismatch(
    tmp_path: Path,
) -> None:
    write_results(tmp_path, _partial_payload())

    wrong_range = copy.deepcopy(_expected())
    wrong_range["scenario_sampling_range"] = None
    with pytest.raises(ValueError, match="scenario_sampling_range"):
        load_or_create_results(
            tmp_path, wrong_range, resume_existing=True, view_specs=VIEW_SPECS
        )

    wrong_success = copy.deepcopy(_expected())
    wrong_success["success_criterion"] = {"kind": "task_default"}
    with pytest.raises(ValueError, match="success_criterion"):
        load_or_create_results(
            tmp_path, wrong_success, resume_existing=True, view_specs=VIEW_SPECS
        )


def test_resume_rejects_hammer_prompt_or_trace_contract_mismatch(tmp_path: Path) -> None:
    write_results(tmp_path, _partial_payload())

    wrong_prompts = copy.deepcopy(_expected())
    wrong_prompts["instruction_by_arm"] = {
        "left": "left prompt",
        "right": "right prompt",
    }
    with pytest.raises(ValueError, match="instruction_by_arm"):
        load_or_create_results(
            tmp_path, wrong_prompts, resume_existing=True, view_specs=VIEW_SPECS
        )

    wrong_trace = copy.deepcopy(_expected())
    wrong_trace["trace_dir"] = "/different/trace"
    with pytest.raises(ValueError, match="trace_dir"):
        load_or_create_results(
            tmp_path, wrong_trace, resume_existing=True, view_specs=VIEW_SPECS
        )


def test_resume_rejects_non_prefix_or_inconsistent_episode_checkpoint(tmp_path: Path) -> None:
    payload = _partial_payload()
    payload["views"][0]["episode_results"][0]["seed"] = 100001
    write_results(tmp_path, payload)

    with pytest.raises(ValueError, match="seed prefix"):
        load_or_create_results(
            tmp_path, _expected(), resume_existing=True, view_specs=VIEW_SPECS
        )


def test_existing_results_require_explicit_resume(tmp_path: Path) -> None:
    write_results(tmp_path, _partial_payload())
    with pytest.raises(FileExistsError, match="--resume-existing"):
        load_or_create_results(
            tmp_path, _expected(), resume_existing=False, view_specs=VIEW_SPECS
        )
