from pathlib import Path
import argparse
import importlib
import os
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402
from envs.utils import ArmTag  # noqa: E402


TASK_NAME = "place_two_cubes_box"
TASK_CONFIG = "demo_nero_two_cubes"


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _get_embodiment_config(robot_file):
    return _load_yaml(ROOT / robot_file / "config.yml")


def _apply_embodiment_overrides(args):
    overrides = args.get("embodiment_overrides", {})
    for side, config_key in (
        ("left", "left_embodiment_config"),
        ("right", "right_embodiment_config"),
    ):
        side_overrides = overrides.get(side, {})
        if side_overrides:
            args[config_key].update(side_overrides)


def _build_args(seed, render_freq):
    args = _load_yaml(ROOT / "task_config" / f"{TASK_CONFIG}.yml")
    args["task_name"] = TASK_NAME
    embodiment_type = args["embodiment"]
    embodiment_config = _load_yaml(Path(CONFIGS_PATH) / "_embodiment_config.yml")
    args["left_robot_file"] = embodiment_config[embodiment_type[0]]["file_path"]
    args["right_robot_file"] = embodiment_config[embodiment_type[1]]["file_path"]
    args["embodiment_dis"] = embodiment_type[2]
    args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])
    _apply_embodiment_overrides(args)
    args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    args["task_config"] = TASK_CONFIG
    args["save_path"] = str(ROOT / "data" / TASK_NAME / TASK_CONFIG)
    args["need_plan"] = True
    args["render_freq"] = render_freq
    args["save_data"] = False
    args["now_ep_num"] = 0
    args["seed"] = seed
    return args


def _class_decorator(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def _last_status(task):
    left = task.left_joint_path[-1]["status"] if task.left_joint_path else "NA"
    right = task.right_joint_path[-1]["status"] if task.right_joint_path else "NA"
    return f"plan_success={task.plan_success} left_last={left} right_last={right}"


def _run_step(task, label, action_tuple):
    left_before = len(task.left_joint_path)
    right_before = len(task.right_joint_path)
    ok = task.move(action_tuple)
    print(f"{label}: move_return={ok} {_last_status(task)}")
    _print_new_paths(task, left_before, right_before)
    return ok


def _print_path_delta(side, idx, path):
    if path.get("status") != "Success":
        print(f"  {side}_path[{idx}]: {path.get('status')}")
        return
    pos = path["position"]
    start = pos[0]
    end = pos[-1]
    delta = end - start
    print(f"  {side}_path[{idx}] start:", [round(float(v), 4) for v in start[:7]])
    print(f"  {side}_path[{idx}] end:  ", [round(float(v), 4) for v in end[:7]])
    print(f"  {side}_path[{idx}] delta:", [round(float(v), 4) for v in delta[:7]])


def _print_new_paths(task, left_before, right_before):
    for idx in range(left_before, len(task.left_joint_path)):
        _print_path_delta("left", idx, task.left_joint_path[idx])
    for idx in range(right_before, len(task.right_joint_path)):
        _print_path_delta("right", idx, task.right_joint_path[idx])


def _print_positions(task):
    print("  right_cube:", [round(float(v), 6) for v in task.right_cube.get_pose().p])
    print("  left_cube:", [round(float(v), 6) for v in task.left_cube.get_pose().p])
    print("  right_target:", [round(float(v), 6) for v in task.right_target_pose.p])
    print("  left_target:", [round(float(v), 6) for v in task.left_target_pose.p])
    print("  right_ee:", [round(float(v), 6) for v in task.robot.get_right_ee_pose()])
    print("  left_ee:", [round(float(v), 6) for v in task.robot.get_left_ee_pose()])
    print("  check_success:", task.check_success())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render-freq", type=int, default=0)
    args = parser.parse_args()

    task = _class_decorator(TASK_NAME)
    try:
        task.setup_demo(**_build_args(args.seed, args.render_freq))
        _print_positions(task)

        task.play_once()
        _print_positions(task)
        print(_last_status(task))
        _print_new_paths(task, 0, 0)
    finally:
        task.close_env()


if __name__ == "__main__":
    main()
