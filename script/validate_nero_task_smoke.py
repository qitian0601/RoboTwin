"""Run one Nero expert episode without saving data and print step-level status."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.load(file.read(), Loader=yaml.FullLoader)


def build_args(task_name: str, task_config: str, seed: int, overrides: dict) -> dict:
    args = load_yaml(ROOT / "task_config" / f"{task_config}.yml")
    args.update(overrides)
    args["task_name"] = task_name

    embodiment = args["embodiment"]
    embodiment_index = load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    args["left_robot_file"] = embodiment_index[embodiment[0]]["file_path"]
    args["right_robot_file"] = embodiment_index[embodiment[1]]["file_path"]
    args["embodiment_dis"] = embodiment[2]
    args["dual_arm_embodied"] = False

    for side in ("left", "right"):
        robot_file = Path(args[f"{side}_robot_file"])
        robot_config = load_yaml(robot_file / "config.yml")
        robot_config.update(args.get("embodiment_overrides", {}).get(side, {}))
        args[f"{side}_embodiment_config"] = robot_config

    args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    args["task_config"] = task_config
    args["save_path"] = str(ROOT / "data" / task_name / task_config)
    args["need_plan"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    args["now_ep_num"] = 0
    args["seed"] = seed
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override-json", default="{}")
    parser.add_argument("--save-debug-dir", type=Path)
    cli = parser.parse_args()

    module = importlib.import_module(f"envs.{cli.task_name}")
    task = getattr(module, cli.task_name)()
    try:
        task.setup_demo(
            **build_args(
                cli.task_name,
                cli.task_config,
                cli.seed,
                json.loads(cli.override_json),
            )
        )
        original_move = task.move
        move_index = 0

        def logged_move(*actions):
            nonlocal move_index
            move_index += 1
            result = original_move(*actions)
            print(
                f"move={move_index} result={result} "
                f"plan_success={task.plan_success} success={task.check_success()}"
            )
            for actor_attr in ("bell", "hammer", "bottle1", "bottle2"):
                actor = getattr(task, actor_attr, None)
                if actor is None:
                    continue
                contacts = task.get_gripper_actor_contact_position(actor.get_name())
                print(
                    f"  actor={actor_attr} pose={actor.get_pose().p.tolist()} "
                    f"gripper_contact_points={len(contacts)} "
                    f"contact_samples={[point.tolist() for point in contacts[:3]]}"
                )
            print(
                f"  grippers_closed left={task.is_left_gripper_close()} "
                f"right={task.is_right_gripper_close()}"
            )
            print(
                f"  ee left={task.robot.get_left_ee_pose()[:3]} "
                f"right={task.robot.get_right_ee_pose()[:3]}"
            )
            if hasattr(task, "bell"):
                print(f"  bell_contact_0={task.bell.get_contact_point(0)[:3]}")
            if hasattr(task, "block"):
                print(
                    f"  block_pose_p={task.block.get_pose().p.tolist()} "
                    f"block_pose_q={task.block.get_pose().q.tolist()} "
                    f"block_functional_1="
                    f"{task.block.get_functional_point(1, 'pose').p.tolist()}"
                )
            if hasattr(task, "hammer"):
                print(
                    f"  hammer_functional_0="
                    f"{task.hammer.get_functional_point(0, 'pose').p.tolist()}"
                )
            if cli.save_debug_dir is not None:
                cli.save_debug_dir.mkdir(parents=True, exist_ok=True)
                task.save_camera_rgb(
                    str(cli.save_debug_dir / f"{cli.task_name}_move{move_index}.png")
                )
            return result

        task.move = logged_move
        info = task.play_once()
        print(
            f"FINAL task={cli.task_name} seed={cli.seed} "
            f"plan_success={task.plan_success} success={task.check_success()} info={info}"
        )
    finally:
        task.close_env()


if __name__ == "__main__":
    main()
