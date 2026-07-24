#!/usr/bin/env python3
"""Run one PI0.5 checkpoint on three Nero tasks under camera views C0-C6."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SINGLE_TASK_EVALUATOR = ROOT / "script" / "evaluate_pi05_camera_views.py"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    config: str
    instruction: str


TASK_SPECS = (
    TaskSpec(
        "place_two_cubes_box",
        "demo_nero_two_cubes",
        (
            "Use the right arm to place the yellow cube into the black box, then use "
            "the left arm to place the green cube into the black box."
        ),
    ),
    TaskSpec(
        "pick_dual_bottles",
        "demo_nero_pick_dual_bottles_c0",
        "Pick up both bottles, one with each arm.",
    ),
    TaskSpec(
        "beat_block_hammer",
        "demo_nero_beat_block_hammer_c0",
        "Pick up the hammer and strike the block.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--server-address", default="127.0.0.1:8081")
    parser.add_argument("--episodes-per-view", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-inference-seed", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "three_task_camera_view_eval",
    )
    parser.add_argument("--run-name")
    parser.add_argument(
        "--scenario-seeds-from-run",
        type=Path,
        help=(
            "Reuse each task's shared_valid_seeds from a previous three-task run. "
            "This keeps policy/Adapter comparisons on identical scenarios."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=[spec.name for spec in TASK_SPECS],
        help="Evaluate only selected task(s). Repeat the option to select several.",
    )
    parser.add_argument("--view", choices=[f"C{index}" for index in range(7)])
    parser.add_argument("--adapter-on-c0", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-stride", type=int, default=3)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=400)
    parser.add_argument("--video-crf", type=int, default=28)
    parser.add_argument("--video-preset", default="ultrafast")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Cap each episode for a load/interface smoke test; omit for formal evaluation.",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continue with later tasks if one task process exits with an error.",
    )
    return parser.parse_args()


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as manifest_file:
        json.dump(payload, manifest_file, ensure_ascii=False, indent=2)
        manifest_file.write("\n")


def write_aggregate_summary(run_dir: Path, task_results: list[dict]) -> None:
    rows = []
    for task_result in task_results:
        task_name = task_result["task_name"]
        for view in task_result.get("views", []):
            rows.append(
                {
                    "task": task_name,
                    "view": view["id"],
                    "description": view["description"],
                    "successes": view["successes"],
                    "episodes": view["episodes"],
                    "success_rate": view["success_rate"],
                }
            )

    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                "task",
                "view",
                "description",
                "successes",
                "episodes",
                "success_rate",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    total_successes = sum(int(row["successes"]) for row in rows)
    total_episodes = sum(int(row["episodes"]) for row in rows)
    macro_rate = (
        sum(float(row["success_rate"]) for row in rows) / len(rows) if rows else None
    )
    aggregate = {
        "rows": len(rows),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "micro_success_rate": total_successes / total_episodes if total_episodes else None,
        "macro_task_view_success_rate": macro_rate,
    }
    write_manifest(run_dir / "aggregate.json", aggregate)


def main() -> None:
    cli = parse_args()
    policy_path = cli.policy_path.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"PI0.5 checkpoint directory does not exist: {policy_path}")
    if cli.episodes_per_view <= 0:
        raise ValueError("--episodes-per-view must be positive")
    if cli.max_steps is not None and cli.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    selected_names = set(cli.task or [spec.name for spec in TASK_SPECS])
    selected_specs = [spec for spec in TASK_SPECS if spec.name in selected_names]
    run_name = cli.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cli.output_dir.expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "policy_path": str(policy_path),
        "server_address": cli.server_address,
        "episodes_per_view": cli.episodes_per_view,
        "seed": cli.seed,
        "policy_inference_seed": cli.policy_inference_seed,
        "view": cli.view,
        "adapter_on_c0": cli.adapter_on_c0,
        "record_video": cli.record_video,
        "max_steps": cli.max_steps,
        "scenario_seeds_from_run": (
            str(cli.scenario_seeds_from_run.expanduser().resolve())
            if cli.scenario_seeds_from_run is not None
            else None
        ),
        "tasks": [spec.name for spec in selected_specs],
        "task_runs": [],
    }
    write_manifest(run_dir / "run_manifest.json", manifest)

    task_results: list[dict] = []
    for task_spec in selected_specs:
        task_dir = run_dir / task_spec.name
        command = [
            sys.executable,
            str(SINGLE_TASK_EVALUATOR),
            "--policy-path",
            str(policy_path),
            "--server-address",
            cli.server_address,
            "--task-name",
            task_spec.name,
            "--task-config",
            task_spec.config,
            "--instruction",
            task_spec.instruction,
            "--episodes-per-view",
            str(cli.episodes_per_view),
            "--seed",
            str(cli.seed),
            "--run-dir",
            str(task_dir),
            "--video-stride",
            str(cli.video_stride),
            "--video-width",
            str(cli.video_width),
            "--video-height",
            str(cli.video_height),
            "--video-crf",
            str(cli.video_crf),
            "--video-preset",
            cli.video_preset,
        ]
        if cli.policy_inference_seed is not None:
            command.extend(
                ["--policy-inference-seed", str(cli.policy_inference_seed)]
            )
        if cli.view is not None:
            command.extend(["--view", cli.view])
        if cli.scenario_seeds_from_run is not None:
            previous_results = (
                cli.scenario_seeds_from_run.expanduser().resolve()
                / task_spec.name
                / "results.json"
            )
            if not previous_results.is_file():
                raise FileNotFoundError(
                    f"Previous task results do not exist: {previous_results}"
                )
            command.extend(["--scenario-seeds-file", str(previous_results)])
        if cli.adapter_on_c0:
            command.append("--adapter-on-c0")
        if cli.record_video:
            command.append("--record-video")
        if cli.max_steps is not None:
            command.extend(["--max-steps", str(cli.max_steps)])

        print(f"\n=== {task_spec.name}: starting ===", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        task_record = {
            "task_name": task_spec.name,
            "task_config": task_spec.config,
            "instruction": task_spec.instruction,
            "output_dir": str(task_dir),
            "returncode": completed.returncode,
        }
        manifest["task_runs"].append(task_record)
        write_manifest(run_dir / "run_manifest.json", manifest)

        results_path = task_dir / "results.json"
        if completed.returncode == 0 and results_path.is_file():
            with results_path.open(encoding="utf-8") as result_file:
                task_results.append(json.load(result_file))
        elif not cli.continue_on_error:
            raise subprocess.CalledProcessError(completed.returncode, command)

        write_aggregate_summary(run_dir, task_results)
        print(
            f"=== {task_spec.name}: returncode={completed.returncode} ===",
            flush=True,
        )

    write_aggregate_summary(run_dir, task_results)
    failed_tasks = [item["task_name"] for item in manifest["task_runs"] if item["returncode"]]
    print(f"\nSaved three-task results to {run_dir}")
    if failed_tasks:
        print(f"Tasks with process errors: {failed_tasks}")


if __name__ == "__main__":
    main()
