#!/usr/bin/env python3
"""Replay a LeRobot URDF-derived EE trajectory through the PI0.5 simulation IK path."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sapien
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "policy"))

# RoboTwin5090 does not install pyarrow, but the compatible pushT-pi05 Python 3.10
# environment already has it for LeRobot dataset inspection.
PYARROW_SITE = Path(os.environ.get("ROBOTWIN_PYARROW_SITE", ""))
if PYARROW_SITE.is_dir():
    sys.path.append(str(PYARROW_SITE))
import pyarrow.parquet as pq  # noqa: E402

from lerobot_pi05_ee_base.deploy_policy import (  # noqa: E402
    NeroCuroboIK,
    _world_control_target_to_base_ik,
)
from script.replay_multiview_dataset import load_task_args  # noqa: E402


DEFAULT_DATASET = ROOT / "data/place_two_cubes_box_lerobot_v3_eef"
DEFAULT_SOURCE = ROOT / "data/place_two_cubes_box/demo_nero_two_cubes"
DEFAULT_CUROBO = ROOT / "assets/embodiments/nero/curobo.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--control-mode", choices=("ee_ik", "joint"), default="ee_ik")
    parser.add_argument("--task-config", default="demo_nero_two_cubes")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--gripper-max-width", type=float, default=0.1)
    parser.add_argument("--curobo-config", type=Path, default=DEFAULT_CUROBO)
    parser.add_argument("--curobo-num-seeds", type=int, default=128)
    parser.add_argument("--curobo-return-seeds", type=int, default=8)
    parser.add_argument("--curobo-position-threshold", type=float, default=0.005)
    parser.add_argument("--curobo-rotation-threshold", type=float, default=0.03)
    parser.add_argument(
        "--align-cubes-to-demo-grasp",
        action="store_true",
        help=(
            "Align each cube's x/y to the first fully closed gripper pose in the "
            "recorded episode. This compensates for task RNG/config drift when "
            "validating a recorded trajectory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/urdf_eef_replay/<timestamp>_episode<id>",
    )
    return parser.parse_args()


def read_episode(dataset: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    files = sorted((dataset / "data").glob("*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset / 'data'}")
    tables = [
        pq.read_table(path, columns=["episode_index", "frame_index", "action.eef_pose", "action"])
        for path in files
    ]
    rows: list[tuple[int, list[float], list[float]]] = []
    for table in tables:
        episode_ids = table["episode_index"].to_numpy()
        frame_ids = table["frame_index"].to_numpy()
        poses = table["action.eef_pose"]
        actions = table["action"]
        for index in np.flatnonzero(episode_ids == episode):
            rows.append((int(frame_ids[index]), poses[index].as_py(), actions[index].as_py()))
    if not rows:
        raise ValueError(f"Episode {episode} does not exist in {dataset}")
    rows.sort(key=lambda row: row[0])
    expected = list(range(len(rows)))
    if [row[0] for row in rows] != expected:
        raise ValueError(f"Episode {episode} frame indices are not contiguous")
    return (
        np.asarray([row[1] for row in rows], dtype=np.float32),
        np.asarray([row[2] for row in rows], dtype=np.float32),
    )


def pose7_to_pose6(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    quaternion_wxyz = pose[3:7]
    rotation = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]])
    return np.concatenate((pose[:3], rotation.as_rotvec())).astype(np.float32)


def pose_error(target: np.ndarray, actual_pose7: np.ndarray) -> tuple[float, float]:
    actual = pose7_to_pose6(actual_pose7)
    position_error = float(np.linalg.norm(actual[:3] - target[:3]))
    rotation_error = float(
        (Rotation.from_rotvec(actual[3:]).inv() * Rotation.from_rotvec(target[3:])).magnitude()
    )
    return position_error, rotation_error


def actor_pose(actor) -> np.ndarray:
    pose = actor.get_pose()
    return np.concatenate((np.asarray(pose.p), np.asarray(pose.q))).astype(np.float64)


def first_closed_grasp(
    ee_actions: np.ndarray,
    joint_actions: np.ndarray,
    pose_start: int,
    gripper_index: int,
    threshold_m: float = 0.005,
) -> tuple[int, np.ndarray]:
    closed = np.flatnonzero(joint_actions[:, gripper_index] <= threshold_m)
    if not closed.size:
        raise ValueError(
            f"No gripper value <= {threshold_m} m at action index {gripper_index}"
        )
    frame = int(closed[0])
    return frame, np.asarray(ee_actions[frame, pose_start : pose_start + 3], dtype=np.float64)


def align_cube_xy(actor, grasp_position: np.ndarray) -> np.ndarray:
    pose = actor_pose(actor)
    pose[:2] = grasp_position[:2]
    actor.actor.set_pose(sapien.Pose(pose[:3], pose[3:]))
    return actor_pose(actor)


def start_video(path: Path, width: int = 1280, height: int = 800, fps: int = 30):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


def main() -> None:
    cli = parse_args()
    dataset = cli.dataset.expanduser().resolve()
    source_root = cli.source_root.expanduser().resolve()
    output_dir = cli.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            ROOT
            / "outputs/urdf_eef_replay"
            / (
                f"{timestamp}_episode{cli.episode}_{cli.control_mode}"
                f"{'_aligned' if cli.align_cubes_to_demo_grasp else ''}"
            )
        )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    ee_actions, joint_actions = read_episode(dataset, cli.episode)
    seeds = [int(value) for value in (source_root / "seed.txt").read_text().split()]
    if cli.episode >= len(seeds):
        raise ValueError(f"No source seed for episode {cli.episode}")
    seed = seeds[cli.episode]

    args = load_task_args(cli.task_config)
    args.update(
        {
            "save_path": str(source_root),
            "now_ep_num": cli.episode,
            "seed": seed,
            "is_test": True,
            "eval_mode": True,
            "eval_video_save_dir": str(output_dir),
            "eval_video_fps": cli.fps,
        }
    )
    task_module = importlib.import_module("envs.place_two_cubes_box")
    task = task_module.place_two_cubes_box()
    video = None
    records: list[dict] = []
    try:
        task.setup_demo(**args)
        generated_right_cube = actor_pose(task.right_cube)
        generated_left_cube = actor_pose(task.left_cube)
        right_close_frame, right_grasp_position = first_closed_grasp(
            ee_actions, joint_actions, pose_start=0, gripper_index=14
        )
        left_close_frame, left_grasp_position = first_closed_grasp(
            ee_actions, joint_actions, pose_start=7, gripper_index=15
        )
        if cli.align_cubes_to_demo_grasp:
            align_cube_xy(task.right_cube, right_grasp_position)
            align_cube_xy(task.left_cube, left_grasp_position)
        initial_right_cube = actor_pose(task.right_cube)
        initial_left_cube = actor_pose(task.left_cube)
        ik = None
        if cli.control_mode == "ee_ik":
            ik = NeroCuroboIK(
                cli.curobo_config,
                cli.curobo_num_seeds,
                cli.curobo_return_seeds,
                cli.curobo_position_threshold,
                cli.curobo_rotation_threshold,
                "gripper_tcp",
            )
        video = start_video(output_dir / f"episode{cli.episode}.mp4", fps=cli.fps)
        task._set_eval_video_ffmpeg(video)

        for frame, (pose_action, joint_action) in enumerate(zip(ee_actions, joint_actions, strict=True)):
            right_target = pose7_to_pose6(pose_action[:7])
            left_target = pose7_to_pose6(pose_action[7:14])
            original_right_joints = np.asarray(joint_action[:7], dtype=np.float32)
            original_left_joints = np.asarray(joint_action[7:14], dtype=np.float32)
            if cli.control_mode == "ee_ik":
                right_base_target = _world_control_target_to_base_ik(task.robot, right_target, "right")
                left_base_target = _world_control_target_to_base_ik(task.robot, left_target, "left")
                right_current = np.asarray(
                    task.robot.get_right_arm_real_jointState()[:7], dtype=np.float32
                )
                left_current = np.asarray(
                    task.robot.get_left_arm_real_jointState()[:7], dtype=np.float32
                )
                right_solution = ik.solve(right_base_target, right_current, "right")
                left_solution = ik.solve(left_base_target, left_current, "left")
            else:
                right_solution = original_right_joints
                left_solution = original_left_joints

            right_gripper_m = float(joint_action[14])
            left_gripper_m = float(joint_action[15])
            sim_action = np.concatenate(
                (
                    left_solution,
                    [np.clip(left_gripper_m / cli.gripper_max_width, 0.0, 1.0)],
                    right_solution,
                    [np.clip(right_gripper_m / cli.gripper_max_width, 0.0, 1.0)],
                )
            )
            task.take_policy_action(sim_action, fps=cli.fps, exact_policy_tracking=True)

            right_position_error, right_rotation_error = pose_error(
                right_target, np.asarray(task.robot.get_right_ee_pose())
            )
            left_position_error, left_rotation_error = pose_error(
                left_target, np.asarray(task.robot.get_left_ee_pose())
            )
            records.append(
                {
                    "frame": frame,
                    "right_gripper_m": right_gripper_m,
                    "left_gripper_m": left_gripper_m,
                    "right_target_x": float(right_target[0]),
                    "right_target_y": float(right_target[1]),
                    "right_target_z": float(right_target[2]),
                    "left_target_x": float(left_target[0]),
                    "left_target_y": float(left_target[1]),
                    "left_target_z": float(left_target[2]),
                    "right_position_error_m": right_position_error,
                    "right_rotation_error_rad": right_rotation_error,
                    "left_position_error_m": left_position_error,
                    "left_rotation_error_rad": left_rotation_error,
                    "right_solution_vs_dataset_joint_norm": float(
                        np.linalg.norm(right_solution - original_right_joints)
                    ),
                    "left_solution_vs_dataset_joint_norm": float(
                        np.linalg.norm(left_solution - original_left_joints)
                    ),
                    "right_cube_x": float(task.right_cube.get_pose().p[0]),
                    "right_cube_y": float(task.right_cube.get_pose().p[1]),
                    "right_cube_z": float(task.right_cube.get_pose().p[2]),
                    "left_cube_x": float(task.left_cube.get_pose().p[0]),
                    "left_cube_y": float(task.left_cube.get_pose().p[1]),
                    "left_cube_z": float(task.left_cube.get_pose().p[2]),
                    "success": bool(task.check_success()),
                }
            )
            if frame % 25 == 0 or frame + 1 == len(ee_actions):
                print(
                    f"frame={frame + 1}/{len(ee_actions)} "
                    f"right_cube_z={records[-1]['right_cube_z']:.4f} "
                    f"left_cube_z={records[-1]['left_cube_z']:.4f}"
                )

        task._del_eval_video_ffmpeg()
        video = None

        right_z = np.asarray([record["right_cube_z"] for record in records])
        left_z = np.asarray([record["left_cube_z"] for record in records])
        right_position_error = np.asarray([record["right_position_error_m"] for record in records])
        left_position_error = np.asarray([record["left_position_error_m"] for record in records])
        summary = {
            "episode": cli.episode,
            "control_mode": cli.control_mode,
            "aligned_cubes_to_demo_grasp": cli.align_cubes_to_demo_grasp,
            "source_seed": seed,
            "frames": len(records),
            "final_success": bool(task.check_success()),
            "right_cube_generated_pose": generated_right_cube.tolist(),
            "right_demo_close_frame": right_close_frame,
            "right_demo_grasp_position": right_grasp_position.tolist(),
            "right_cube_initial_pose": initial_right_cube.tolist(),
            "right_cube_final_pose": actor_pose(task.right_cube).tolist(),
            "right_cube_max_lift_m": float(right_z.max() - initial_right_cube[2]),
            "left_cube_generated_pose": generated_left_cube.tolist(),
            "left_demo_close_frame": left_close_frame,
            "left_demo_grasp_position": left_grasp_position.tolist(),
            "left_cube_initial_pose": initial_left_cube.tolist(),
            "left_cube_final_pose": actor_pose(task.left_cube).tolist(),
            "left_cube_max_lift_m": float(left_z.max() - initial_left_cube[2]),
            "right_position_error_mean_m": float(right_position_error.mean()),
            "right_position_error_p90_m": float(np.quantile(right_position_error, 0.9)),
            "right_position_error_max_m": float(right_position_error.max()),
            "left_position_error_mean_m": float(left_position_error.mean()),
            "left_position_error_p90_m": float(np.quantile(left_position_error, 0.9)),
            "left_position_error_max_m": float(left_position_error.max()),
            "ik_position_threshold_m": cli.curobo_position_threshold,
            "ik_rotation_threshold_rad": cli.curobo_rotation_threshold,
            "right_solution_vs_dataset_joint_norm_mean": float(
                np.mean([record["right_solution_vs_dataset_joint_norm"] for record in records])
            ),
            "right_solution_vs_dataset_joint_norm_max": float(
                np.max([record["right_solution_vs_dataset_joint_norm"] for record in records])
            ),
            "left_solution_vs_dataset_joint_norm_mean": float(
                np.mean([record["left_solution_vs_dataset_joint_norm"] for record in records])
            ),
            "left_solution_vs_dataset_joint_norm_max": float(
                np.max([record["left_solution_vs_dataset_joint_norm"] for record in records])
            ),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (output_dir / "trace.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Output: {output_dir}")
    finally:
        if video is not None:
            task._del_eval_video_ffmpeg()
        if hasattr(task, "scene"):
            task.close_env(clear_cache=True)


if __name__ == "__main__":
    main()
