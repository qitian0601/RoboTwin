#!/usr/bin/env python3
"""Evaluate the RoboTwin PI0.5 policy under fixed third-person camera shifts."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
for path in (ROOT, ROOT / "policy"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from envs._GLOBAL_CONFIGS import CONFIGS_PATH
from envs.utils.create_actor import UnStableError
from script.camera_eval_resume import (
    existing_view,
    load_or_create_results,
    unused_video_path,
    write_results,
)
from script.hammer_task_prompts import (
    HAMMER_ARM_INSTRUCTIONS,
    hammer_instruction_for_block_x,
)


DEFAULT_INSTRUCTION = (
    "Use the right arm to place the yellow cube into the black box, then use the left arm "
    "to place the green cube into the black box."
)

@dataclass(frozen=True)
class ViewSpec:
    identifier: str
    kind: str
    value: float
    description: str


VIEW_SPECS = (
    ViewSpec("C0", "canonical", 0.0, "canonical view"),
    ViewSpec("C1", "yaw", -10.0, "yaw -10 deg"),
    ViewSpec("C2", "yaw", 10.0, "yaw +10 deg"),
    ViewSpec("C3", "pitch", -7.0, "pitch -7 deg"),
    ViewSpec("C4", "pitch", 7.0, "pitch +7 deg"),
    ViewSpec("C5", "translation", 0.10, "move forward 10 cm"),
    ViewSpec("C6", "translation", -0.10, "move backward 10 cm"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--server-address", default="127.0.0.1:8081")
    parser.add_argument("--task-name", default="place_two_cubes_box")
    parser.add_argument("--task-config", default="demo_nero_two_cubes")
    parser.add_argument("--episodes-per-view", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-inference-seed",
        type=int,
        help="Reset PI0.5 flow-noise RNG to this seed at the start of every episode.",
    )
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument(
        "--hammer-arm-aware-instruction",
        action="store_true",
        help=(
            "Select the canonical left/right Hammer training prompt from the block "
            "position separately for every episode."
        ),
    )
    parser.add_argument(
        "--hammer-contact-success",
        action="store_true",
        help=(
            "Evaluation-only Hammer success rule: require any hammer/block physical "
            "contact once, without functional-point alignment, velocity stability, "
            "or a sustained hold."
        ),
    )
    parser.add_argument(
        "--hammer-training-support-range",
        action="store_true",
        help=(
            "Restrict Hammer block sampling to the observed slow120 training "
            "support: left x [-0.25, -0.20], right x [0.18, 0.24], and "
            "world y [-0.18, -0.15]."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.8)
    parser.add_argument(
        "--action-merge-new-weight",
        type=float,
        default=0.5,
        help="Weight of the newly predicted action when overlapping chunks are merged.",
    )
    parser.add_argument("--max-policy-step-rad", type=float, default=0.05)
    parser.add_argument("--max-gripper-step-m", type=float, default=0.05)
    parser.add_argument("--max-executor-step-rad", type=float, default=0.005)
    parser.add_argument("--max-executor-gripper-step-m", type=float, default=0.004)
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help=(
            "Write per-replan policy outputs and converted robot targets as JSONL "
            "for inference diagnostics."
        ),
    )
    parser.add_argument(
        "--bottle-lift-success-height-m",
        type=float,
        help=(
            "Evaluation-only pick_dual_bottles success rule: both bottles must rise "
            "this many meters above their initial functional-point heights."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "camera_view_eval")
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=(
            "Write directly to this directory instead of creating a timestamped "
            "subdirectory below --output-dir. Intended for multi-task orchestration."
        ),
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help=(
            "Resume an interrupted --run-dir after strictly validating its policy, "
            "task, prompt, scenario seeds, policy seed, control settings, and camera "
            "contract. Completed episodes are never rerun or overwritten."
        ),
    )
    parser.add_argument("--view", choices=[spec.identifier for spec in VIEW_SPECS])
    parser.add_argument("--scenario-seed", type=int)
    parser.add_argument(
        "--scenario-episode-index",
        type=int,
        help=(
            "Use this task episode index for the first scenario instead of zero. "
            "This reproduces task logic that depends on episode parity."
        ),
    )
    parser.add_argument(
        "--scenario-seeds-file",
        type=Path,
        help=(
            "JSON file containing a seed list, or a previous results.json with "
            "shared_valid_seeds. This bypasses C0 seed collection."
        ),
    )
    parser.add_argument(
        "--adapter-on-c0",
        action="store_true",
        help="Apply a loaded Feature Adapter to C0 instead of its default canonical bypass.",
    )
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--video-stride",
        type=int,
        default=3,
        help="Record one evaluation-video frame every N policy steps (default: 3).",
    )
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=400)
    parser.add_argument("--video-crf", type=int, default=28)
    parser.add_argument("--video-preset", default="ultrafast")
    parser.add_argument(
        "--max-steps",
        type=int,
        help=(
            "Optional per-episode simulator-step cap for interface smoke tests. "
            "It is checked between action chunks and may overshoot by one replan stride. "
            "Omit for the task's normal step limit."
        ),
    )
    return parser.parse_args()


def normalize(vector: Any) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64)
    magnitude = np.linalg.norm(result)
    if magnitude == 0:
        raise ValueError("Camera vector must not be zero")
    return result / magnitude


def rotate(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = normalize(axis)
    theta = np.deg2rad(degrees)
    rotated = (
        vector * np.cos(theta)
        + np.cross(axis, vector) * np.sin(theta)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(theta))
    )
    return normalize(rotated)


def load_task_args(task_name: str, task_config: str) -> dict[str, Any]:
    config_path = ROOT / "task_config" / f"{task_config}.yml"
    with config_path.open(encoding="utf-8") as config_file:
        args = yaml.safe_load(config_file)

    with (Path(CONFIGS_PATH) / "_embodiment_config.yml").open(encoding="utf-8") as config_file:
        embodiment_types = yaml.safe_load(config_file)

    embodiment = args["embodiment"]
    if len(embodiment) != 3:
        raise ValueError("This evaluator expects the two-robot Nero embodiment")

    def robot_config(name: str) -> tuple[str, dict[str, Any]]:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"Embodiment {name} has no file_path")
        with (Path(robot_file) / "config.yml").open(encoding="utf-8") as config_file:
            return robot_file, yaml.safe_load(config_file)

    left_file, left_config = robot_config(embodiment[0])
    right_file, right_config = robot_config(embodiment[1])
    for side, config in (("left", left_config), ("right", right_config)):
        config.update(args.get("embodiment_overrides", {}).get(side, {}))

    args.update(
        {
            "task_name": task_name,
            "task_config": task_config,
            "left_robot_file": left_file,
            "right_robot_file": right_file,
            "left_embodiment_config": left_config,
            "right_embodiment_config": right_config,
            "embodiment_dis": embodiment[2],
            "dual_arm_embodied": False,
            "render_freq": 0,
            "eval_mode": True,
            "eval_video_log": False,
            "save_data": False,
            "save_freq": None,
        }
    )
    return args


def shift_head_camera(args: dict[str, Any], spec: ViewSpec) -> dict[str, Any]:
    shifted = copy.deepcopy(args)
    cameras = shifted["camera"]["static_camera_list"]
    camera = next((item for item in cameras if item["name"] == "head_camera"), None)
    if camera is None:
        raise KeyError("task config has no head_camera static camera")

    position = np.asarray(camera["position"], dtype=np.float64)
    forward = normalize(camera["forward"])
    left = normalize(camera["left"])

    if spec.kind == "canonical":
        pass
    elif spec.kind == "yaw":
        # World Z is the tabletop's vertical axis. Rotate the full camera basis together.
        forward = rotate(forward, np.array([0.0, 0.0, 1.0]), spec.value)
        left = rotate(left, np.array([0.0, 0.0, 1.0]), spec.value)
    elif spec.kind == "pitch":
        # Positive pitch follows the right-hand rule around the camera's local left axis.
        forward = rotate(forward, left, spec.value)
    elif spec.kind == "translation":
        # Positive values move toward the scene along the canonical viewing direction.
        position = position + forward * spec.value
    else:
        raise ValueError(f"Unsupported view shift kind: {spec.kind}")

    camera["position"] = position.tolist()
    camera["forward"] = forward.tolist()
    camera["left"] = left.tolist()
    return shifted


def make_task(task_name: str) -> Any:
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def close_task(task: Any, clear_cache: bool) -> None:
    if hasattr(task, "scene"):
        task.close_env(clear_cache=clear_cache)


def collect_valid_seeds(
    task_name: str,
    base_args: dict[str, Any],
    episodes: int,
    seed: int,
) -> list[int]:
    task = make_task(task_name)
    valid_seeds: list[int] = []
    candidate_seed = 100000 * (1 + seed)
    # Some tasks (notably beat_block_hammer) have sparse valid expert seeds;
    # its first known valid seed is 100066.  Always scan at least 100
    # candidates so one-episode smoke tests do not fail before reaching it.
    # Hammer expert planning is sparse enough that a 10-episode evaluation can
    # exhaust 500 candidates before finding ten reproducible scenes.
    max_attempts = max(episodes * 100, 100)

    for attempt_index in range(max_attempts):
        if len(valid_seeds) == episodes:
            break
        try:
            task.setup_demo(now_ep_num=attempt_index, seed=candidate_seed, is_test=True, **base_args)
            task.play_once()
            if task.plan_success and task.check_success():
                valid_seeds.append(candidate_seed)
        except UnStableError:
            pass
        except Exception as error:
            print(f"[seed-scan] seed={candidate_seed} skipped: {error}")
        finally:
            close_task(task, clear_cache=((attempt_index + 1) % base_args["clear_cache_freq"] == 0))
        candidate_seed += 1

    if len(valid_seeds) != episodes:
        raise RuntimeError(f"Found only {len(valid_seeds)}/{episodes} valid expert seeds")
    return valid_seeds


def run_episode(
    task: Any,
    args: dict[str, Any],
    model: Any,
    seed: int,
    episode_index: int,
    instruction: str,
    video_path: Path | None = None,
    video_fps: float = 10.0,
    video_width: int = 0,
    video_height: int = 0,
    video_crf: int = 23,
    video_preset: str = "medium",
    max_steps: int | None = None,
    hammer_arm_aware_instruction: bool = False,
) -> tuple[bool, str | None, str]:
    deploy = importlib.import_module("lerobot_pi05.deploy_policy")
    video_started = False
    episode_instruction = instruction
    try:
        task.setup_demo(now_ep_num=episode_index, seed=seed, is_test=True, **args)
        if hammer_arm_aware_instruction:
            block_x = float(task.block.get_pose().p[0])
            table_x = float(task.table_xy_bias[0])
            episode_instruction = hammer_instruction_for_block_x(block_x, table_x)
        task.set_instruction(instruction=episode_instruction)
        if video_path is not None:
            video_path.parent.mkdir(parents=True, exist_ok=True)
            observation = task.get_obs()
            frame = observation["observation"]["head_camera"]["rgb"]
            height, width = frame.shape[:2]
            ffmpeg_command = [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                "-framerate", str(video_fps), "-i", "-",
            ]
            if video_width > 0 and video_height > 0:
                ffmpeg_command.extend(
                    ["-vf", f"scale={video_width}:{video_height}:flags=fast_bilinear"]
                )
            ffmpeg_command.extend(
                [
                    "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                    "-preset", video_preset, "-crf", str(video_crf), str(video_path),
                ]
            )
            ffmpeg = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
            task.eval_video_path = str(video_path.parent)
            task._set_eval_video_ffmpeg(ffmpeg)
            video_started = True
        deploy.reset_model(model)
        episode_step_limit = task.step_lim
        if max_steps is not None:
            episode_step_limit = min(episode_step_limit, max_steps)
        while task.take_action_cnt < episode_step_limit:
            deploy.eval(task, model, task.get_obs())
            if task.eval_success:
                return True, None, episode_instruction
        if max_steps is not None and max_steps < task.step_lim:
            return False, f"smoke_step_limit_reached:{max_steps}", episode_instruction
        return False, None, episode_instruction
    except Exception as error:
        return False, f"{type(error).__name__}: {error}", episode_instruction
    finally:
        if video_started:
            task._del_eval_video_ffmpeg()
        close_task(task, clear_cache=((episode_index + 1) % args["clear_cache_freq"] == 0))


def load_scenario_seeds(path: Path, expected_count: int) -> list[int]:
    resolved = path.expanduser().resolve()
    with resolved.open(encoding="utf-8") as seed_file:
        payload = json.load(seed_file)
    if isinstance(payload, dict):
        payload = payload.get("shared_valid_seeds")
    if not isinstance(payload, list):
        raise ValueError(
            f"Seed file must contain a JSON list or shared_valid_seeds: {resolved}"
        )
    seeds = [int(value) for value in payload]
    if len(seeds) != expected_count:
        raise ValueError(
            f"Expected {expected_count} scenario seeds, got {len(seeds)} from {resolved}"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Scenario seeds must be unique: {resolved}")
    return seeds


def main() -> None:
    cli = parse_args()
    if cli.episodes_per_view <= 0:
        raise ValueError("--episodes-per-view must be positive")
    if cli.video_stride <= 0:
        raise ValueError("--video-stride must be positive")
    if cli.video_width < 0 or cli.video_height < 0:
        raise ValueError("--video-width and --video-height must be non-negative")
    if cli.max_steps is not None and cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if cli.scenario_episode_index is not None and cli.scenario_episode_index < 0:
        raise ValueError("--scenario-episode-index must be non-negative")
    if not 0.0 <= cli.action_merge_new_weight <= 1.0:
        raise ValueError("--action-merge-new-weight must be in [0, 1]")
    if cli.hammer_arm_aware_instruction and cli.task_name != "beat_block_hammer":
        raise ValueError(
            "--hammer-arm-aware-instruction requires --task-name beat_block_hammer"
        )
    if cli.hammer_contact_success and cli.task_name != "beat_block_hammer":
        raise ValueError("--hammer-contact-success requires --task-name beat_block_hammer")
    if cli.hammer_training_support_range and cli.task_name != "beat_block_hammer":
        raise ValueError(
            "--hammer-training-support-range requires --task-name beat_block_hammer"
        )
    if (
        cli.bottle_lift_success_height_m is not None
        and cli.bottle_lift_success_height_m <= 0
    ):
        raise ValueError("--bottle-lift-success-height-m must be positive")
    if (
        cli.bottle_lift_success_height_m is not None
        and cli.task_name != "pick_dual_bottles"
    ):
        raise ValueError("--bottle-lift-success-height-m requires pick_dual_bottles")
    if cli.resume_existing and cli.run_dir is None:
        raise ValueError("--resume-existing requires --run-dir")

    policy_path = cli.policy_path.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"PI0.5 checkpoint directory does not exist: {policy_path}")

    base_args = load_task_args(cli.task_name, cli.task_config)
    base_args["policy_path"] = str(policy_path)
    base_args["server_address"] = cli.server_address
    base_args["policy_device"] = "cuda"
    base_args["actions_per_chunk"] = cli.actions_per_chunk
    base_args["chunk_size_threshold"] = cli.chunk_size_threshold
    base_args["aggregate_fn_name"] = "average"
    base_args["action_merge_new_weight"] = cli.action_merge_new_weight
    base_args["fps"] = cli.fps
    base_args["gripper_max_width"] = 0.1
    base_args["direct_sim_control"] = True
    base_args["exact_policy_tracking"] = True
    base_args["max_policy_step_rad"] = cli.max_policy_step_rad
    base_args["max_gripper_step_m"] = cli.max_gripper_step_m
    base_args["max_executor_step_rad"] = cli.max_executor_step_rad
    base_args["max_executor_gripper_step_m"] = cli.max_executor_gripper_step_m
    base_args["trace_enabled"] = cli.trace_dir is not None
    if cli.trace_dir is not None:
        base_args["trace_dir"] = str(cli.trace_dir.expanduser().resolve())
    base_args["policy_inference_seed"] = cli.policy_inference_seed
    base_args["eval_video_stride"] = cli.video_stride
    if cli.bottle_lift_success_height_m is not None:
        base_args["eval_lift_success_height_m"] = cli.bottle_lift_success_height_m
        base_args["success_hold_s"] = 0.0
    if cli.hammer_contact_success:
        base_args["success_require_alignment"] = False
        base_args["success_require_stability"] = False
        base_args["success_hold_s"] = 0.0
    hammer_sampling_range = None
    if cli.hammer_training_support_range:
        base_args["block_left_xlim_offset"] = [-0.25, -0.20]
        base_args["block_right_xlim_offset"] = [0.18, 0.24]
        # The slow120 table bias is y=-0.30, giving world y [-0.18, -0.15].
        base_args["block_ylim_offset"] = [0.12, 0.15]
        table_x, table_y = base_args["table_xy_bias"]
        hammer_sampling_range = {
            "kind": "observed_slow120_training_support",
            "left_x_m": [table_x - 0.25, table_x - 0.20],
            "right_x_m": [table_x + 0.18, table_x + 0.24],
            "y_m": [table_y + 0.12, table_y + 0.15],
        }

    if cli.scenario_seed is not None and cli.scenario_seeds_file is not None:
        raise ValueError("Use only one of --scenario-seed and --scenario-seeds-file")
    if cli.run_dir is not None:
        output_dir = cli.run_dir.expanduser().resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = cli.output_dir.expanduser().resolve() / timestamp

    existing_payload: dict[str, Any] | None = None
    existing_results = output_dir / "results.json"
    if cli.resume_existing and existing_results.is_file():
        with existing_results.open(encoding="utf-8") as result_file:
            candidate = json.load(result_file)
        if not isinstance(candidate, dict):
            raise ValueError(f"Existing result is not a JSON object: {existing_results}")
        existing_payload = candidate

    if cli.scenario_seed is not None:
        valid_seeds = [cli.scenario_seed]
    elif cli.scenario_seeds_file is not None:
        valid_seeds = load_scenario_seeds(
            cli.scenario_seeds_file, cli.episodes_per_view
        )
    elif existing_payload is not None:
        stored_seeds = existing_payload.get("shared_valid_seeds")
        if not isinstance(stored_seeds, list):
            raise ValueError(
                f"Existing result has no shared_valid_seeds list: {existing_results}"
            )
        valid_seeds = [int(seed) for seed in stored_seeds]
        if len(valid_seeds) != cli.episodes_per_view:
            raise ValueError(
                f"Existing result has {len(valid_seeds)} seeds, requested "
                f"{cli.episodes_per_view}"
            )
    else:
        print(f"Collecting {cli.episodes_per_view} shared valid seeds from C0...")
        valid_seeds = collect_valid_seeds(
            cli.task_name, base_args, cli.episodes_per_view, cli.seed
        )
    print(f"Shared seeds: {valid_seeds}")
    expected_payload = {
        "task_name": cli.task_name,
        "task_config": cli.task_config,
        "policy_path": str(policy_path),
        "server_address": cli.server_address,
        "instruction": cli.instruction,
        "instruction_by_arm": (
            HAMMER_ARM_INSTRUCTIONS if cli.hammer_arm_aware_instruction else None
        ),
        "episodes_per_view": cli.episodes_per_view,
        "max_steps": cli.max_steps,
        "shared_valid_seeds": valid_seeds,
        "scenario_episode_index": cli.scenario_episode_index,
        "policy_inference_seed": cli.policy_inference_seed,
        "adapter_c0_bypassed": not cli.adapter_on_c0,
        "success_criterion": (
            {
                "kind": "both_bottles_lifted",
                "height_m": cli.bottle_lift_success_height_m,
                "hold_s": 0.0,
            }
            if cli.bottle_lift_success_height_m is not None
            else (
                {
                    "kind": "any_hammer_block_contact",
                    "hold_s": 0.0,
                }
                if cli.hammer_contact_success
                else {"kind": "task_default"}
            )
        ),
        "seed_search_index": cli.seed,
        "scenario_sampling_range": hammer_sampling_range,
        "control_config": {
            "fps": cli.fps,
            "actions_per_chunk": cli.actions_per_chunk,
            "chunk_size_threshold": cli.chunk_size_threshold,
            "action_merge_new_weight": cli.action_merge_new_weight,
            "max_policy_step_rad": cli.max_policy_step_rad,
            "max_gripper_step_m": cli.max_gripper_step_m,
            "max_executor_step_rad": cli.max_executor_step_rad,
            "max_executor_gripper_step_m": cli.max_executor_gripper_step_m,
        },
        "recording_config": {
            "enabled": cli.record_video,
            "video_stride": cli.video_stride,
            "video_width": cli.video_width,
            "video_height": cli.video_height,
            "video_crf": cli.video_crf,
            "video_preset": cli.video_preset,
        },
        "trace_dir": (
            str(cli.trace_dir.expanduser().resolve())
            if cli.trace_dir is not None
            else None
        ),
    }
    payload = load_or_create_results(
        output_dir,
        expected_payload,
        resume_existing=cli.resume_existing,
        view_specs=VIEW_SPECS,
    )
    write_results(output_dir, payload)

    deploy = importlib.import_module("lerobot_pi05.deploy_policy")
    selected_specs = tuple(spec for spec in VIEW_SPECS if cli.view in (None, spec.identifier))
    for spec in selected_specs:
        view_result = existing_view(payload, spec.identifier)
        if view_result is not None and view_result["complete"]:
            print(
                f"\n[{spec.identifier}] resume: already complete "
                f"({view_result['successes']}/{view_result['episodes']}), skipping"
            )
            continue

        view_args = shift_head_camera(base_args, spec)
        # The evaluator knows the active camera configuration. Preserve the
        # original C0 policy exactly; apply the Adapter only to shifted views.
        view_args["feature_adapter_enabled"] = spec.kind != "canonical" or cli.adapter_on_c0
        model = deploy.get_model(view_args)
        head_camera = next(
            item for item in view_args["camera"]["static_camera_list"] if item["name"] == "head_camera"
        )
        task = make_task(cli.task_name)
        if view_result is None:
            view_result = {
                "id": spec.identifier,
                "kind": spec.kind,
                "value": spec.value,
                "description": spec.description,
                "successes": 0,
                "episodes": 0,
                "success_rate": None,
                "complete": False,
                "head_camera": {
                    "position": head_camera["position"],
                    "forward": head_camera["forward"],
                    "left": head_camera["left"],
                },
                "episode_results": [],
            }
            payload["views"].append(view_result)
            write_results(output_dir, payload)
        else:
            requested_camera = {
                "position": head_camera["position"],
                "forward": head_camera["forward"],
                "left": head_camera["left"],
            }
            for field, requested in requested_camera.items():
                existing = view_result.get("head_camera", {}).get(field)
                if existing is None or not np.allclose(existing, requested):
                    raise ValueError(
                        f"Resume camera contract mismatch for {spec.identifier}.{field}"
                    )

        start_episode = len(view_result["episode_results"])
        successes = int(view_result["successes"])
        print(f"\n[{spec.identifier}] {spec.description}")
        if start_episode:
            print(
                f"[{spec.identifier}] resume: {start_episode}/{len(valid_seeds)} "
                "episodes already checkpointed"
            )
        for episode_index in range(start_episode, len(valid_seeds)):
            scenario_seed = valid_seeds[episode_index]
            task_episode_index = (
                episode_index
                if cli.scenario_episode_index is None
                else cli.scenario_episode_index + episode_index
            )
            video_path = None
            if cli.record_video:
                video_path = unused_video_path(
                    output_dir / f"{spec.identifier}_seed_{scenario_seed}.mp4"
                )
            success, error, episode_instruction = run_episode(
                task,
                view_args,
                model,
                scenario_seed,
                task_episode_index,
                cli.instruction,
                video_path,
                video_fps=cli.fps / cli.video_stride,
                video_width=cli.video_width,
                video_height=cli.video_height,
                video_crf=cli.video_crf,
                video_preset=cli.video_preset,
                max_steps=cli.max_steps,
                hammer_arm_aware_instruction=cli.hammer_arm_aware_instruction,
            )
            successes += int(success)
            view_result["episode_results"].append(
                {
                    "seed": scenario_seed,
                    "scenario_episode_index": task_episode_index,
                    "instruction": episode_instruction,
                    "success": success,
                    "error": error,
                    "video": str(video_path) if video_path is not None else None,
                }
            )
            view_result["successes"] = successes
            view_result["episodes"] = len(view_result["episode_results"])
            view_result["success_rate"] = successes / view_result["episodes"]
            view_result["complete"] = view_result["episodes"] == len(valid_seeds)
            write_results(output_dir, payload)
            print(
                f"[{spec.identifier}] {episode_index + 1}/{len(valid_seeds)} "
                f"seed={scenario_seed} {'success' if success else 'fail'}"
            )

        success_rate = successes / len(valid_seeds)
        view_result["complete"] = True
        write_results(output_dir, payload)
        print(f"[{spec.identifier}] {successes}/{len(valid_seeds)} = {success_rate:.1%}")
        model.channel.close()

    write_results(output_dir, payload)
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
