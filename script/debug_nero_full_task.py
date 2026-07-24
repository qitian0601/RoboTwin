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


TASK_NAME = "beat_block_hammer"
TASK_CONFIG = "demo_nero_smoke"


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _get_embodiment_config(robot_file):
    return _load_yaml(ROOT / robot_file / "config.yml")


def _build_args(seed):
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
    args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    args["task_config"] = TASK_CONFIG
    args["save_path"] = str(ROOT / "data" / TASK_NAME / TASK_CONFIG)
    args["need_plan"] = True
    args["render_freq"] = 0
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
    ok = task.move(action_tuple)
    print(f"{label}: move_return={ok} {_last_status(task)}")
    return ok


def _print_grasp_state(task, arm_tag):
    entity = task.robot.right_entity if arm_tag == "right" else task.robot.left_entity
    active = entity.get_active_joints()
    names = [joint.get_name() for joint in active]
    qpos = entity.get_qpos()
    grippers = task.robot.right_gripper if arm_tag == "right" else task.robot.left_gripper
    print("  gripper_links:", task.robot.gripper_name)
    for joint, scale, bias in grippers:
        name = joint.get_name()
        idx = names.index(name)
        target = joint.get_drive_target()[0] if hasattr(joint, "get_drive_target") else joint.drive_target
        print(f"  {name}: qpos={float(qpos[idx]):.6f} drive_target={float(target):.6f} scale={scale} bias={bias}")
    contacts = []
    hammer_name = task.hammer.get_name()
    for contact in task.scene.get_contacts():
        n0 = contact.bodies[0].entity.name
        n1 = contact.bodies[1].entity.name
        if hammer_name in (n0, n1):
            other = n1 if n0 == hammer_name else n0
            contacts.append(other)
    print("  hammer_contacts:", sorted(set(contacts)))
    print("  gripper_hammer_contact_points:", len(task.get_gripper_actor_contact_position(hammer_name)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--gripper-pos", type=float, default=0.0)
    parser.add_argument("--pre-grasp-dis", type=float, default=0.12)
    parser.add_argument("--grasp-dis", type=float, default=0.01)
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--gripper-stiffness", type=float, default=None)
    parser.add_argument("--gripper-damping", type=float, default=None)
    parser.add_argument("--gripper-bias", type=float, default=None)
    parser.add_argument("--render-freq", type=int, default=0)
    parser.add_argument("--place-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    args = parser.parse_args()
    for seed in range(args.seeds):
        task = _class_decorator(TASK_NAME)
        try:
            setup_args = _build_args(seed)
            setup_args["render_freq"] = args.render_freq
            if args.friction is not None:
                setup_args["static_friction"] = args.friction
                setup_args["dynamic_friction"] = args.friction
            if args.gripper_stiffness is not None:
                setup_args["left_embodiment_config"]["gripper_stiffness"] = args.gripper_stiffness
                setup_args["right_embodiment_config"]["gripper_stiffness"] = args.gripper_stiffness
            if args.gripper_damping is not None:
                setup_args["left_embodiment_config"]["gripper_damping"] = args.gripper_damping
                setup_args["right_embodiment_config"]["gripper_damping"] = args.gripper_damping
            if args.gripper_bias is not None:
                setup_args["left_embodiment_config"]["gripper_bias"] = args.gripper_bias
                setup_args["right_embodiment_config"]["gripper_bias"] = args.gripper_bias
            task.setup_demo(**setup_args)
            block_pose = task.block.get_functional_point(0, "pose").p
            arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
            print(f"\nseed={seed} arm_tag={arm_tag} block={block_pose.tolist()}")

            if not _run_step(
                task,
                "grasp",
                task.grasp_actor(
                    task.hammer,
                    arm_tag=arm_tag,
                    pre_grasp_dis=args.pre_grasp_dis,
                    grasp_dis=args.grasp_dis,
                    gripper_pos=args.gripper_pos,
                ),
            ):
                continue
            _print_grasp_state(task, arm_tag)
            if not _run_step(task, "lift", task.move_by_displacement(arm_tag, z=0.07, move_axis="arm")):
                continue
            place_target = task.block.get_functional_point(1, "pose")
            place_target.p += args.place_offset
            if not _run_step(
                task,
                "place",
                task.place_actor(
                    task.hammer,
                    target_pose=place_target,
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.06,
                    dis=0,
                    is_open=False,
                ),
            ):
                continue

            print("check_success:", task.check_success())
            hammer = task.hammer.get_functional_point(0, "pose").p
            target = task.block.get_functional_point(1, "pose").p
            print("hammer_fp0:", hammer.tolist())
            print("block_fp1:", target.tolist())
        except Exception as exc:
            print(f"seed={seed} exception {type(exc).__name__}: {exc}")
        finally:
            task.close_env()


if __name__ == "__main__":
    main()
