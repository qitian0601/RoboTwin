from pathlib import Path
import argparse
import importlib
import os
import sys

import numpy as np
import torch
import transforms3d as t3d
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from curobo.types.math import Pose as CuroboPose  # noqa: E402
from curobo.types.robot import JointState  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # noqa: E402
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig  # noqa: E402
from envs._GLOBAL_CONFIGS import CONFIGS_PATH  # noqa: E402
from envs.utils import ArmTag  # noqa: E402


TASK_NAME = "beat_block_hammer"
TASK_CONFIG = "demo_nero_smoke"
CUROBO_YML = ROOT / "assets" / "embodiments" / "nero" / "curobo.yml"


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def _get_embodiment_config(robot_file):
    return _load_yaml(ROOT / robot_file / "config.yml")


def _build_args(robot_quat=None):
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
    if robot_quat is not None:
        for key in ("left_embodiment_config", "right_embodiment_config"):
            poses = args[key]["robot_pose"]
            idx = 0 if key == "left_embodiment_config" or len(poses) == 1 else 1
            poses[idx] = poses[idx][:3] + robot_quat
    args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    args["task_config"] = TASK_CONFIG
    args["save_path"] = str(ROOT / "data" / TASK_NAME / TASK_CONFIG)
    args["need_plan"] = True
    args["render_freq"] = 0
    args["save_data"] = False
    return args


def _class_decorator(task_name):
    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def _world_to_base(base_pose, target_pose):
    base_p, base_q = np.array(base_pose[:3]), np.array(base_pose[3:])
    target_p, target_q = np.array(target_pose[:3]), np.array(target_pose[3:])
    w_R_b = t3d.quaternions.quat2mat(base_q)
    w_R_t = t3d.quaternions.quat2mat(target_q)
    return (w_R_b.T @ (target_p - base_p)).tolist() + t3d.quaternions.mat2quat(w_R_b.T @ w_R_t).tolist()


def _candidate_base_target(task, arm_tag, candidate_idx):
    actor = task.hammer
    contact_matrix = actor.get_contact_point(0, "matrix")
    contact_pose_matrix = contact_matrix @ np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    rotation = contact_pose_matrix[:3, :3]
    position = contact_pose_matrix[:3, 3] + rotation @ np.array([-0.24, 0, 0]).T
    raw_pose = list(position) + list(t3d.quaternions.mat2quat(rotation))
    target_list = task.robot.create_target_pose_list(raw_pose, actor.get_contact_point(0, "list"), arm_tag)
    pose = target_list[candidate_idx]
    endlink_pose = task.robot._trans_from_gripper_to_endlink(pose, arm_tag=arm_tag)
    origin = task.robot.right_entity_origion_pose if arm_tag == "right" else task.robot.left_entity_origion_pose
    base_pose = origin.p.tolist() + origin.q.tolist()
    return pose, _world_to_base(base_pose, endlink_pose.p.tolist() + endlink_pose.q.tolist())


def _advance_grasp_pose(pre_pose, pre_dis=0.12, target_dis=0.01):
    pose = np.array(pre_pose, dtype=float)
    direction_mat = t3d.quaternions.quat2mat(pose[-4:])
    pose[:3] += [pre_dis - target_dis, 0, 0] @ np.linalg.inv(direction_mat)
    return pose.tolist()


def _base_target_from_gripper_pose(task, arm_tag, gripper_pose, gripper_bias=None):
    if gripper_bias is None:
        endlink_pose = task.robot._trans_from_gripper_to_endlink(gripper_pose, arm_tag=arm_tag)
    else:
        old_left = task.robot.left_gripper_bias
        old_right = task.robot.right_gripper_bias
        task.robot.left_gripper_bias = gripper_bias
        task.robot.right_gripper_bias = gripper_bias
        endlink_pose = task.robot._trans_from_gripper_to_endlink(gripper_pose, arm_tag=arm_tag)
        task.robot.left_gripper_bias = old_left
        task.robot.right_gripper_bias = old_right
    origin = task.robot.right_entity_origion_pose if arm_tag == "right" else task.robot.left_entity_origion_pose
    base_pose = origin.p.tolist() + origin.q.tolist()
    return _world_to_base(base_pose, endlink_pose.p.tolist() + endlink_pose.q.tolist())


def _make_motion_gen(world_config, self_collision_check=True):
    cfg = MotionGenConfig.load_from_robot_config(
        str(CUROBO_YML),
        world_config,
        interpolation_dt=1 / 250,
        num_ik_seeds=64,
        num_graph_seeds=4,
        num_trajopt_seeds=4,
        num_trajopt_noisy_seeds=4,
        self_collision_check=self_collision_check,
        self_collision_opt=self_collision_check,
        use_cuda_graph=False,
    )
    mg = MotionGen(cfg)
    mg.warmup()
    return mg


def _make_ik_solver():
    cfg = IKSolverConfig.load_from_robot_config(
        str(CUROBO_YML),
        None,
        num_seeds=128,
        position_threshold=0.01,
        rotation_threshold=0.08,
        self_collision_check=False,
        self_collision_opt=False,
        use_cuda_graph=False,
        collision_checker_type=None,
    )
    return IKSolver(cfg)


def _ik_success(ik_solver, base_target):
    result = ik_solver.solve_single(CuroboPose.from_list([float(v) for v in base_target]))
    return bool(result.success.item())


def _plan(motion_gen, base_target, start_names, start_values, check_start_validity=True):
    goal = CuroboPose.from_list([float(v) for v in base_target])
    start = JointState.from_position(
        torch.tensor(start_values, device="cuda", dtype=torch.float32).reshape(1, -1),
        joint_names=start_names,
    )
    result = motion_gen.plan_single(
        start,
        goal,
        MotionGenPlanConfig(max_attempts=20, timeout=10.0, check_start_validity=check_start_validity),
    )
    status = getattr(result, "status", None)
    success = bool(result.success.item())
    position = None
    if success:
        position = np.array(result.interpolated_plan.position.to("cpu"))
    return success, str(status), position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-quat", type=float, nargs=4, default=[0.707, 0, 0, 0.707])
    parser.add_argument("--candidate", type=int, default=3)
    parser.add_argument("--phase", choices=["pre", "grasp"], default="pre")
    parser.add_argument("--target-dis", type=float, default=0.01)
    parser.add_argument("--gripper-bias", type=float, default=None)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    task = _class_decorator(TASK_NAME)
    setup_args = _build_args(robot_quat=args.robot_quat)
    task.setup_demo(now_ep_num=0, seed=0, **setup_args)
    block_pose = task.block.get_functional_point(0, "pose").p
    arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
    pre_gripper_pose, pre_base_target = _candidate_base_target(task, arm_tag, args.candidate)
    if args.sweep:
        ik_solver = _make_ik_solver()
        print("arm_tag:", arm_tag)
        print("candidate:", args.candidate)
        print("sweep target_dis x gripper_bias; values show grasp IK success")
        target_dis_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
        bias_values = [0.04, 0.06, 0.08, 0.0958, 0.11, 0.12, 0.14]
        for target_dis in target_dis_values:
            row = []
            for bias in bias_values:
                grasp_pose = _advance_grasp_pose(pre_gripper_pose, target_dis=target_dis)
                base_target = _base_target_from_gripper_pose(task, arm_tag, grasp_pose, gripper_bias=bias)
                row.append("Y" if _ik_success(ik_solver, base_target) else ".")
            print(f"target_dis={target_dis:.4f}:", " ".join(row), "  biases=", bias_values)
        task.close_env()
        return
    if args.phase == "pre":
        gripper_pose = pre_gripper_pose
        base_target = pre_base_target
    else:
        pre_plan = task.robot.right_plan_path(pre_gripper_pose) if arm_tag == "right" else task.robot.left_plan_path(pre_gripper_pose)
        print("pre_plan_status:", pre_plan.get("status"))
        if pre_plan.get("status") != "Success":
            print("Cannot test grasp because pre plan failed.")
            task.close_env()
            return
        grasp_pose = _advance_grasp_pose(pre_gripper_pose, target_dis=args.target_dis)
        gripper_pose = grasp_pose
        base_target = _base_target_from_gripper_pose(task, arm_tag, grasp_pose, gripper_bias=args.gripper_bias)

    print("arm_tag:", arm_tag)
    print("candidate:", args.candidate)
    print("phase:", args.phase)
    print("gripper_pose_world:", [round(float(v), 6) for v in gripper_pose])
    print("base_target_endlink:", [round(float(v), 6) for v in base_target])

    active_names = [joint.get_name() for joint in (task.robot.right_entity if arm_tag == "right" else task.robot.left_entity).get_active_joints()]
    qpos = (task.robot.right_entity if arm_tag == "right" else task.robot.left_entity).get_qpos()
    arm_names = task.robot.right_arm_joints_name if arm_tag == "right" else task.robot.left_arm_joints_name
    if args.phase == "grasp":
        start7 = [float(v) for v in pre_plan["position"][-1]]
    else:
        start7 = [float(qpos[active_names.index(name)]) for name in arm_names]
    start9 = [float(qpos[active_names.index(name)]) for name in load_yaml(str(CUROBO_YML))["robot_cfg"]["kinematics"]["cspace"]["joint_names"]]

    origin = task.robot.right_entity_origion_pose if arm_tag == "right" else task.robot.left_entity_origion_pose
    table_world = {
        "cuboid": {
            "table": {
                "dims": [0.7, 2, 0.04],
                "pose": [origin.p[1], 0.0, 0.74 - origin.p[2], 1, 0, 0, 0.0],
            }
        }
    }

    variants = [
        ("table_self_on_start7_check_start_on", table_world, True, arm_names, start7, True),
        ("table_self_on_start7_check_start_off", table_world, True, arm_names, start7, False),
        ("table_self_off_start7_check_start_on", table_world, False, arm_names, start7, True),
        ("table_self_off_start7_check_start_off", table_world, False, arm_names, start7, False),
        ("table_self_on_start9_check_start_on", table_world, True,
         load_yaml(str(CUROBO_YML))["robot_cfg"]["kinematics"]["cspace"]["joint_names"], start9, True),
    ]
    for name, world_config, self_collision, start_names, start_values, check_start in variants:
        try:
            mg = _make_motion_gen(world_config, self_collision_check=self_collision)
            success, status, _ = _plan(mg, base_target, start_names, start_values, check_start_validity=check_start)
            print(f"{name}: {'Success' if success else 'Failure'} status={status}")
        except Exception as exc:
            print(f"{name}: Exception {type(exc).__name__}: {exc}")

    task.close_env()


if __name__ == "__main__":
    main()
