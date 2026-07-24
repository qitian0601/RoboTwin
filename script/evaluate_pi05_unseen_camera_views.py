#!/usr/bin/env python3
"""Evaluate pickplace PI0.5 on camera poses not present in Adapter training."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_pi05_camera_views as base


ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "place_two_cubes_box"
TASK_CONFIG = "demo_nero_two_cubes"
INSTRUCTION = (
    "Use the right arm to place the yellow cube into the black box, then use the left arm "
    "to place the green cube into the black box."
)


@dataclass(frozen=True)
class UnseenViewSpec:
    identifier: str
    group: str
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    translation_m: float = 0.0
    description: str = ""


# Adapter training used only yaw +/-10 deg, pitch +/-7 deg and translation +/-10 cm.
# None of the poses below appeared in the Adapter training view set.
UNSEEN_VIEW_SPECS = (
    UnseenViewSpec("I_YM05", "interpolation", yaw_deg=-5.0, description="yaw -5 deg"),
    UnseenViewSpec("I_YP05", "interpolation", yaw_deg=5.0, description="yaw +5 deg"),
    UnseenViewSpec("I_PM03_5", "interpolation", pitch_deg=-3.5, description="pitch -3.5 deg"),
    UnseenViewSpec("I_PP03_5", "interpolation", pitch_deg=3.5, description="pitch +3.5 deg"),
    UnseenViewSpec(
        "I_TF05", "interpolation", translation_m=0.05, description="move forward 5 cm"
    ),
    UnseenViewSpec(
        "I_TB05", "interpolation", translation_m=-0.05, description="move backward 5 cm"
    ),
    UnseenViewSpec("E_YM15", "extrapolation", yaw_deg=-15.0, description="yaw -15 deg"),
    UnseenViewSpec("E_YP15", "extrapolation", yaw_deg=15.0, description="yaw +15 deg"),
    UnseenViewSpec(
        "E_PM10_5", "extrapolation", pitch_deg=-10.5, description="pitch -10.5 deg"
    ),
    UnseenViewSpec(
        "E_PP10_5", "extrapolation", pitch_deg=10.5, description="pitch +10.5 deg"
    ),
    UnseenViewSpec(
        "E_TF15", "extrapolation", translation_m=0.15, description="move forward 15 cm"
    ),
    UnseenViewSpec(
        "E_TB15", "extrapolation", translation_m=-0.15, description="move backward 15 cm"
    ),
    UnseenViewSpec(
        "C_YM10_PM07",
        "combination",
        yaw_deg=-10.0,
        pitch_deg=-7.0,
        description="yaw -10 deg + pitch -7 deg",
    ),
    UnseenViewSpec(
        "C_YM10_PP07",
        "combination",
        yaw_deg=-10.0,
        pitch_deg=7.0,
        description="yaw -10 deg + pitch +7 deg",
    ),
    UnseenViewSpec(
        "C_YP10_PM07",
        "combination",
        yaw_deg=10.0,
        pitch_deg=-7.0,
        description="yaw +10 deg + pitch -7 deg",
    ),
    UnseenViewSpec(
        "C_YP10_PP07",
        "combination",
        yaw_deg=10.0,
        pitch_deg=7.0,
        description="yaw +10 deg + pitch +7 deg",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--server-address", default="127.0.0.1:8081")
    parser.add_argument("--episodes-per-view", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-inference-seed", type=int, default=0)
    parser.add_argument("--scenario-seeds-file", type=Path)
    parser.add_argument(
        "--suite",
        action="append",
        choices=("interpolation", "extrapolation", "combination"),
        help="Run only selected suite(s). Repeat to select several; default is all.",
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=[spec.identifier for spec in UNSEEN_VIEW_SPECS],
        help="Run only selected view(s). Repeat to select several.",
    )
    parser.add_argument(
        "--feature-adapter-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Exact output directory; otherwise a timestamp is appended below --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "unseen_camera_view_eval",
    )
    return parser.parse_args()


def shift_head_camera(args: dict[str, Any], spec: UnseenViewSpec) -> dict[str, Any]:
    shifted = copy.deepcopy(args)
    camera = next(
        item
        for item in shifted["camera"]["static_camera_list"]
        if item["name"] == "head_camera"
    )
    position = np.asarray(camera["position"], dtype=np.float64)
    forward = base.normalize(camera["forward"])
    left = base.normalize(camera["left"])

    if spec.yaw_deg:
        forward = base.rotate(forward, np.array([0.0, 0.0, 1.0]), spec.yaw_deg)
        left = base.rotate(left, np.array([0.0, 0.0, 1.0]), spec.yaw_deg)
    if spec.pitch_deg:
        forward = base.rotate(forward, left, spec.pitch_deg)
    if spec.translation_m:
        position = position + forward * spec.translation_m

    camera["position"] = position.tolist()
    camera["forward"] = forward.tolist()
    camera["left"] = left.tolist()
    return shifted


def load_seed_pool(path: Path, count: int) -> list[int]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as seed_file:
        payload = json.load(seed_file)
    if isinstance(payload, dict):
        payload = payload.get("shared_valid_seeds")
    if not isinstance(payload, list):
        raise ValueError(f"No shared_valid_seeds list in {resolved}")
    seeds = [int(value) for value in payload]
    if len(seeds) < count:
        raise ValueError(f"Need {count} seeds, found only {len(seeds)} in {resolved}")
    selected = seeds[:count]
    if len(set(selected)) != len(selected):
        raise ValueError(f"Selected scenario seeds are not unique: {resolved}")
    return selected


def write_results(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, ensure_ascii=False, indent=2)
        result_file.write("\n")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "view",
                "suite",
                "description",
                "yaw_deg",
                "pitch_deg",
                "translation_m",
                "successes",
                "episodes",
                "success_rate",
            ),
        )
        writer.writeheader()
        for view in payload["views"]:
            writer.writerow(
                {
                    "view": view["id"],
                    "suite": view["suite"],
                    "description": view["description"],
                    "yaw_deg": view["yaw_deg"],
                    "pitch_deg": view["pitch_deg"],
                    "translation_m": view["translation_m"],
                    "successes": view["successes"],
                    "episodes": view["episodes"],
                    "success_rate": view["success_rate"],
                }
            )


def main() -> None:
    cli = parse_args()
    if cli.episodes_per_view <= 0:
        raise ValueError("--episodes-per-view must be positive")
    if cli.max_steps is not None and cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    policy_path = cli.policy_path.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"PI0.5 checkpoint directory does not exist: {policy_path}")

    selected_suites = set(cli.suite or ("interpolation", "extrapolation", "combination"))
    selected_views = set(cli.view or [spec.identifier for spec in UNSEEN_VIEW_SPECS])
    specs = [
        spec
        for spec in UNSEEN_VIEW_SPECS
        if spec.group in selected_suites and spec.identifier in selected_views
    ]
    if not specs:
        raise ValueError("No unseen views selected")

    task_args = base.load_task_args(TASK_NAME, TASK_CONFIG)
    task_args.update(
        {
            "policy_path": str(policy_path),
            "server_address": cli.server_address,
            "policy_device": "cuda",
            "actions_per_chunk": 50,
            "chunk_size_threshold": 0.8,
            "aggregate_fn_name": "average",
            "fps": 30,
            "gripper_max_width": 0.1,
            "direct_sim_control": True,
            "exact_policy_tracking": True,
            "max_policy_step_rad": 0.05,
            "max_gripper_step_m": 0.05,
            "max_executor_step_rad": 0.005,
            "max_executor_gripper_step_m": 0.004,
            "trace_enabled": False,
            "eval_video_stride": 3,
            "policy_inference_seed": cli.policy_inference_seed,
        }
    )

    if cli.scenario_seeds_file is not None:
        seeds = load_seed_pool(cli.scenario_seeds_file, cli.episodes_per_view)
    else:
        print(f"Collecting {cli.episodes_per_view} valid expert seeds from C0...")
        seeds = base.collect_valid_seeds(
            TASK_NAME, task_args, cli.episodes_per_view, cli.seed
        )

    output_dir = (
        cli.run_dir.expanduser().resolve()
        if cli.run_dir is not None
        else cli.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    payload: dict[str, Any] = {
        "task_name": TASK_NAME,
        "task_config": TASK_CONFIG,
        "policy_path": str(policy_path),
        "server_address": cli.server_address,
        "instruction": INSTRUCTION,
        "episodes_per_view": cli.episodes_per_view,
        "shared_valid_seeds": seeds,
        "policy_inference_seed": cli.policy_inference_seed,
        "feature_adapter_enabled": cli.feature_adapter_enabled,
        "max_steps": cli.max_steps,
        "views": [],
    }

    deploy = base.importlib.import_module("lerobot_pi05.deploy_policy")
    model_args = copy.deepcopy(task_args)
    model_args["feature_adapter_enabled"] = cli.feature_adapter_enabled
    model = deploy.get_model(model_args)
    print(f"Shared seeds: {seeds}")
    try:
        for spec in specs:
            view_args = shift_head_camera(task_args, spec)
            view_args["feature_adapter_enabled"] = cli.feature_adapter_enabled
            task = base.make_task(TASK_NAME)
            head_camera = next(
                item
                for item in view_args["camera"]["static_camera_list"]
                if item["name"] == "head_camera"
            )
            episode_results = []
            successes = 0
            print(f"\n[{spec.identifier}] {spec.group}: {spec.description}")
            for episode_index, scenario_seed in enumerate(seeds):
                success, error = base.run_episode(
                    task,
                    view_args,
                    model,
                    scenario_seed,
                    episode_index,
                    INSTRUCTION,
                    max_steps=cli.max_steps,
                )
                successes += int(success)
                episode_results.append(
                    {"seed": scenario_seed, "success": success, "error": error}
                )
                print(
                    f"[{spec.identifier}] {episode_index + 1}/{len(seeds)} "
                    f"seed={scenario_seed} {'success' if success else 'fail'}"
                )

            payload["views"].append(
                {
                    "id": spec.identifier,
                    "suite": spec.group,
                    "description": spec.description,
                    "yaw_deg": spec.yaw_deg,
                    "pitch_deg": spec.pitch_deg,
                    "translation_m": spec.translation_m,
                    "successes": successes,
                    "episodes": len(seeds),
                    "success_rate": successes / len(seeds),
                    "head_camera": {
                        "position": head_camera["position"],
                        "forward": head_camera["forward"],
                        "left": head_camera["left"],
                    },
                    "episode_results": episode_results,
                }
            )
            write_results(output_dir, payload)
            print(
                f"[{spec.identifier}] {successes}/{len(seeds)} "
                f"= {successes / len(seeds):.1%}"
            )
    finally:
        model.channel.close()

    write_results(output_dir, payload)
    print(f"\nSaved unseen-view results to {output_dir}")


if __name__ == "__main__":
    main()
