from pathlib import Path
import importlib
import os
import sys

import numpy as np
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


def _build_args():
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
    for side, config_key in (("left", "left_embodiment_config"), ("right", "right_embodiment_config")):
        args[config_key].update(args.get("embodiment_overrides", {}).get(side, {}))
    args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    args["task_config"] = TASK_CONFIG
    args["save_path"] = str(ROOT / "data" / TASK_NAME / TASK_CONFIG)
    args["need_plan"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    args["now_ep_num"] = 0
    args["seed"] = 0
    return args


def main():
    module = importlib.import_module(f"envs.{TASK_NAME}")
    task = getattr(module, TASK_NAME)()
    right = ArmTag("right")
    try:
        task.setup_demo(**_build_args())
        print("initial right_cube:", [round(float(v), 4) for v in task.right_cube.get_pose().p])
        print("box_center:", [round(float(v), 4) for v in task.box_center])
        print("box x range:", [round(-task.box_half_xy[0], 4), round(task.box_half_xy[0], 4)])

        ok = task.move(task.grasp_actor(task.right_cube, arm_tag=right, pre_grasp_dis=0.08, grasp_dis=-0.015))
        print("right_grasp:", ok, task.plan_success)
        task._move_by_chunks(right, z=0.22, max_step=0.035)
        print("right_lift:", task.plan_success)
        print("lifted right_cube:", [round(float(v), 4) for v in task.right_cube.get_pose().p])
        print("right_ee:", [round(float(v), 4) for v in task.robot.get_right_ee_pose()[:3]])

        for x in np.linspace(-0.12, 0.06, 10):
            target_xy = np.array([float(x), task.box_center[1]])
            target_high_pos = np.array(task.robot.get_right_ee_pose()[:3], dtype=float)
            target_high_pos[:2] += target_xy - task.right_cube.get_pose().p[:2]
            high_path, low_path = task._plan_right_cross_place_paths(target_high_pos)
            status = "Y" if high_path is not None else "."
            lower = "low" if low_path is not None else "high"
            print(f"x={x:+.3f} {status} {lower}")
    finally:
        task.close_env()


if __name__ == "__main__":
    main()
