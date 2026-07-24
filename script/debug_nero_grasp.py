from copy import deepcopy
import argparse
import importlib
import os
from pathlib import Path
import sys

import numpy as np
import transforms3d as t3d
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from envs._GLOBAL_CONFIGS import CONFIGS_PATH, GRASP_DIRECTION_DIC  # noqa: E402
from envs.utils import ArmTag, cal_quat_dis  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose as CuroboPose  # noqa: E402
from curobo.util_file import load_yaml  # noqa: E402
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig  # noqa: E402


def _world_to_base(base_pose, target_pose):
    base_p, base_q = np.array(base_pose[:3]), np.array(base_pose[3:])
    target_p, target_q = np.array(target_pose[:3]), np.array(target_pose[3:])
    rel_p = target_p - base_p
    w_R_b = t3d.quaternions.quat2mat(base_q)
    w_R_t = t3d.quaternions.quat2mat(target_q)
    return (w_R_b.T @ rel_p).tolist() + t3d.quaternions.mat2quat(w_R_b.T @ w_R_t).tolist()


TASK_NAME = "beat_block_hammer"
TASK_CONFIG = "demo_nero_smoke"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)


def _make_ik_solver():
    tensor_args = TensorDeviceType()
    cfg = IKSolverConfig.load_from_robot_config(
        load_yaml(str(ROOT / "assets" / "embodiments" / "nero" / "curobo.yml")),
        None,
        tensor_args=tensor_args,
        num_seeds=128,
        position_threshold=0.01,
        rotation_threshold=0.08,
        self_collision_check=False,
        self_collision_opt=False,
        use_cuda_graph=False,
        collision_checker_type=None,
    )
    return IKSolver(cfg), tensor_args


def _ik_status(ik_solver, tensor_args, base_target):
    pose = CuroboPose.from_list([float(v) for v in base_target], tensor_args=tensor_args)
    result = ik_solver.solve_single(pose)
    return "IKSuccess" if _as_bool(result.success) else "IKFail"


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

    def get_embodiment_file(name):
        return embodiment_config[name]["file_path"]

    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
    args["embodiment_dis"] = embodiment_type[2]
    args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = _get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = _get_embodiment_config(args["right_robot_file"])
    if robot_quat is not None:
        left_pose = args["left_embodiment_config"]["robot_pose"][0]
        right_pose_list = args["right_embodiment_config"]["robot_pose"]
        right_pose = right_pose_list[0 if len(right_pose_list) == 1 else 1]
        args["left_embodiment_config"]["robot_pose"][0] = left_pose[:3] + robot_quat
        args["right_embodiment_config"]["robot_pose"][0 if len(right_pose_list) == 1 else 1] = right_pose[:3] + robot_quat
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


def _candidate_grasp_pose(task, actor, arm_tag, contact_point_id, pre_dis):
    contact_matrix = actor.get_contact_point(contact_point_id, "matrix")
    if contact_matrix is None:
        return None
    global_contact_pose_matrix = contact_matrix @ np.array(
        [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    )
    rotation = global_contact_pose_matrix[:3, :3]
    position = global_contact_pose_matrix[:3, 3] + rotation @ np.array([-0.12 - pre_dis, 0, 0]).T
    quat = t3d.quaternions.mat2quat(rotation)
    return list(position) + list(quat)


def _grasp_from_pre(pre_pose, pre_dis, target_dis):
    pose = np.array(deepcopy(pre_pose))
    direction_mat = t3d.quaternions.quat2mat(pose[-4:])
    pose[:3] += [pre_dis - target_dis, 0, 0] @ np.linalg.inv(direction_mat)
    return pose.tolist()


def _plan_pair(task, arm_tag, pre_pose, grasp_pose):
    plan_func = task.robot.left_plan_path if arm_tag == "left" else task.robot.right_plan_path
    pre_path = plan_func(pre_pose)
    if pre_path.get("status") != "Success":
        return "Fail", None, "skip"
    grasp_path = plan_func(grasp_pose, last_qpos=pre_path["position"][-1])
    return pre_path.get("status"), len(pre_path.get("position", [])), grasp_path.get("status")


def _debug_choose_best(task, actor, arm_tag, contact_id, raw_pose, ik_solver, tensor_args):
    center_pose = actor.get_contact_point(contact_id, "list")
    target_list = task.robot.create_target_pose_list(raw_pose, center_pose, arm_tag)
    plan_multi = task.robot.left_plan_multi_path if arm_tag == "left" else task.robot.right_plan_multi_path
    result = plan_multi(target_list)
    statuses = [str(v) for v in result.get("status", [])]
    print("  rotate candidates:", len(target_list))
    print("  batch statuses:", statuses)
    for idx, pose in enumerate(target_list):
        trans_pose = task.robot._trans_from_gripper_to_endlink(pose, arm_tag=arm_tag)
        if arm_tag == "left":
            base_pose = task.robot.left_entity_origion_pose.p.tolist() + task.robot.left_entity_origion_pose.q.tolist()
        else:
            base_pose = task.robot.right_entity_origion_pose.p.tolist() + task.robot.right_entity_origion_pose.q.tolist()
        base_target = _world_to_base(base_pose, trans_pose.p.tolist() + trans_pose.q.tolist())
        ik_status = _ik_status(ik_solver, tensor_args, base_target)
        print(f"    rot[{idx}] status={statuses[idx] if idx < len(statuses) else 'NA'} {ik_status} pose:",
              [round(float(v), 6) for v in pose])
        print("      planner_base_target:", [round(float(v), 6) for v in base_target])
    if "Success" not in statuses:
        return None
    best_idx = statuses.index("Success")
    return target_list[best_idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-quat", type=float, nargs=4, default=None)
    parsed = parser.parse_args()

    args = _build_args(robot_quat=parsed.robot_quat)
    task = _class_decorator(TASK_NAME)
    task.setup_demo(now_ep_num=0, seed=0, **args)
    ik_solver, tensor_args = _make_ik_solver()

    block_pose = task.block.get_functional_point(0, "pose").p
    arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")
    actor = task.hammer
    pre_dis = 0.12
    target_dis = 0.01
    pref_direction = task.robot.get_grasp_perfect_direction(arm_tag)
    print("task:", TASK_NAME)
    print("seed:", 0)
    print("arm_tag:", arm_tag)
    print("block_pose:", [round(float(v), 6) for v in block_pose])
    print("hammer_pose:", [round(float(v), 6) for v in actor.get_pose().p], [round(float(v), 6) for v in actor.get_pose().q])
    print("pref_direction:", pref_direction, GRASP_DIRECTION_DIC[pref_direction])
    print("left_ee_pose:", [round(float(v), 6) for v in task.robot.get_left_ee_pose()])
    print("right_ee_pose:", [round(float(v), 6) for v in task.robot.get_right_ee_pose()])
    print("contact_points:", len(actor.config["contact_points_pose"]))

    best = []
    for contact_id, _ in actor.iter_contact_points():
        raw_pre_pose = _candidate_grasp_pose(task, actor, arm_tag, contact_id, pre_dis)
        if raw_pre_pose is None:
            print(f"[{contact_id}] raw_pre_pose=None")
            continue
        print(f"[{contact_id}] raw_pre_pose:", [round(float(v), 6) for v in raw_pre_pose])
        pre_pose = _debug_choose_best(task, actor, arm_tag, contact_id, raw_pre_pose, ik_solver, tensor_args)
        if pre_pose is None:
            print(f"[{contact_id}] choose_best_pose=None")
            continue
        grasp_pose = _grasp_from_pre(pre_pose, pre_dis, target_dis)
        quat_top = GRASP_DIRECTION_DIC["top_down_little_left" if arm_tag == "right" else "top_down_little_right"]
        dis_top = cal_quat_dis(grasp_pose[-4:], quat_top)
        dis_side = cal_quat_dis(grasp_pose[-4:], GRASP_DIRECTION_DIC[pref_direction])
        weighted = 0.7 * dis_top + 0.3 * dis_side
        pre_status, pre_steps, grasp_status = _plan_pair(task, arm_tag, pre_pose, grasp_pose)
        best.append((weighted, contact_id, pre_status, grasp_status))
        print(f"[{contact_id}] weighted={weighted:.4f} top={dis_top:.4f} side={dis_side:.4f}")
        print("  contact:", [round(float(v), 6) for v in actor.get_contact_point(contact_id, "list")])
        print("  pre_pose:", [round(float(v), 6) for v in pre_pose])
        print("  grasp_pose:", [round(float(v), 6) for v in grasp_pose])
        print(f"  plan: pre={pre_status} pre_steps={pre_steps} grasp={grasp_status}")

    print("\nranked:")
    for item in sorted(best)[:10]:
        print(f"  contact={item[1]} weighted={item[0]:.4f} pre={item[2]} grasp={item[3]}")
    task.close_env()


if __name__ == "__main__":
    main()
