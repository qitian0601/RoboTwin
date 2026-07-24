from pathlib import Path
import argparse
import itertools

import numpy as np
import sapien.core as sapien
import torch
import transforms3d as t3d
import yaml

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


ROOT = Path(__file__).resolve().parents[1]
NERO_DIR = ROOT / "assets" / "embodiments" / "nero"
CONFIG_YML = NERO_DIR / "config.yml"
CUROBO_YML = NERO_DIR / "curobo.yml"
URDF = NERO_DIR / "nero_with_gripper_description.urdf"


def _load_config():
    with CONFIG_YML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _round_matrix(matrix):
    rounded = []
    for row in matrix:
        rounded.append([0 if abs(float(v)) < 1e-6 else round(float(v), 6) for v in row])
    return rounded


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if torch.is_tensor(value):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    return bool(value)


def _make_solver(args):
    tensor_args = TensorDeviceType()
    ik_cfg = IKSolverConfig.load_from_robot_config(
        load_yaml(str(CUROBO_YML)),
        None,
        tensor_args=tensor_args,
        num_seeds=args.num_seeds,
        position_threshold=args.position_threshold,
        rotation_threshold=args.rotation_threshold,
        self_collision_check=False,
        self_collision_opt=False,
        use_cuda_graph=False,
        collision_checker_type=None,
    )
    return IKSolver(ik_cfg), tensor_args


def _solve_target_link_pose(solver, tensor_args, xyz, target_quat, delta_matrix):
    target_rot = t3d.quaternions.quat2mat(target_quat)
    # This mirrors Robot._trans_from_gripper_to_endlink().
    endlink_rot = target_rot @ np.linalg.inv(delta_matrix)
    endlink_quat = t3d.quaternions.mat2quat(endlink_rot)
    pose = CuroboPose.from_list(
        [float(xyz[0]), float(xyz[1]), float(xyz[2])] + [float(v) for v in endlink_quat],
        tensor_args=tensor_args,
    )
    result = solver.solve_single(pose)
    if not _as_bool(result.success):
        return None, endlink_quat
    solution = result.solution.detach().cpu().reshape(-1, result.solution.shape[-1]).numpy()[0]
    return solution, endlink_quat


def _load_sapien_robot(q, joint_names):
    engine = sapien.Engine()
    scene = engine.create_scene()
    scene.set_timestep(1 / 250)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(str(URDF))
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    active_joints = robot.get_active_joints()
    active_names = [joint.get_name() for joint in active_joints]
    qpos = robot.get_qpos()
    for src_idx, name in enumerate(joint_names):
        if name in active_names:
            qpos[active_names.index(name)] = q[src_idx]
    robot.set_qpos(qpos)

    return scene, robot, active_names, qpos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xyz", type=float, nargs=3, default=[0.35, 0.65, 0.16])
    parser.add_argument("--quat", type=float, nargs=4, default=[1, 0, 0, 0])
    parser.add_argument("--scan", action="store_true", default=True)
    parser.add_argument("--xlim", type=float, nargs=2, default=[0.0, 0.7])
    parser.add_argument("--ylim", type=float, nargs=2, default=[0.0, 0.9])
    parser.add_argument("--zlim", type=float, nargs=2, default=[0.0, 0.5])
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--num-seeds", type=int, default=128)
    parser.add_argument("--position-threshold", type=float, default=0.005)
    parser.add_argument("--rotation-threshold", type=float, default=0.03)
    args = parser.parse_args()

    config = _load_config()
    delta_matrix = np.array(config["delta_matrix"], dtype=float)
    target_quat = np.array(args.quat, dtype=float)
    target_rot = t3d.quaternions.quat2mat(target_quat)

    solver, tensor_args = _make_solver(args)
    q, endlink_quat = _solve_target_link_pose(solver, tensor_args, args.xyz, target_quat, delta_matrix)
    if q is None:
        if not args.scan:
            raise RuntimeError("CuRobo IK failed for the Step 6 target. Try another --xyz from Step 5 successes.")
        xs = np.arange(args.xlim[0], args.xlim[1] + args.step * 0.5, args.step)
        ys = np.arange(args.ylim[0], args.ylim[1] + args.step * 0.5, args.step)
        zs = np.arange(args.zlim[0], args.zlim[1] + args.step * 0.5, args.step)
        tried = 0
        for xyz in itertools.product(xs, ys, zs):
            tried += 1
            q, endlink_quat = _solve_target_link_pose(solver, tensor_args, xyz, target_quat, delta_matrix)
            if q is not None:
                args.xyz = list(xyz)
                print(f"Default target failed; found reachable Step 6 target after {tried} tries.")
                break
        if q is None:
            raise RuntimeError("No reachable Step 6 target found in scan range.")

    scene, robot, active_names, qpos = _load_sapien_robot(q, solver.joint_names)
    ee_joint = robot.find_joint_by_name(config["ee_joints"][0])
    if ee_joint is None:
        raise RuntimeError(f"Cannot find ee joint {config['ee_joints'][0]}")

    joint_pose = ee_joint.global_pose
    w_R_joint = t3d.quaternions.quat2mat(joint_pose.q)
    global_trans_matrix = w_R_joint.T @ target_rot @ delta_matrix.T

    print("Step 6 target xyz in CuRobo/base_link frame:", [round(float(v), 6) for v in args.xyz])
    print("Step 6 target quaternion[wxyz]:", [round(float(v), 6) for v in target_quat])
    print("Planner endlink quaternion after delta^-1[wxyz]:", [round(float(v), 6) for v in endlink_quat])
    print("CuRobo joint_names:", solver.joint_names)
    print("SAPIEN active joint names:", active_names)
    print("IK solution q:")
    print([round(float(v), 6) for v in q])
    print("SAPIEN ee_joint global position:")
    print([round(float(v), 6) for v in joint_pose.p])
    print("SAPIEN ee_joint global quaternion[wxyz]:")
    print([round(float(v), 6) for v in joint_pose.q])
    print("w_R_joint:")
    for row in _round_matrix(w_R_joint):
        print(" ", row)
    print("delta_matrix:")
    for row in _round_matrix(delta_matrix):
        print(" ", row)
    print("computed global_trans_matrix:")
    for row in _round_matrix(global_trans_matrix):
        print(" ", row)

    verified_rot = w_R_joint @ global_trans_matrix @ delta_matrix
    print("verified R_joint @ global_trans @ delta:")
    for row in _round_matrix(verified_rot):
        print(" ", row)
    print("verified quaternion[wxyz]:", [round(float(v), 6) for v in t3d.quaternions.mat2quat(verified_rot)])


if __name__ == "__main__":
    main()
