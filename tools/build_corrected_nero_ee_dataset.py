#!/usr/bin/env python3
"""Build a training-ready Nero EE dataset with image-aligned observations.

The source LeRobot dataset keeps the successful joint-policy timing convention:
each row's action is the next joint drive target. Observations are replaced with
the actual RoboTwin control-frame poses stored alongside the same source image
in the original HDF5 episode. Actions are the FK of the existing joint action,
with gripper widths copied from that same action row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYARROW_SITE = Path(os.environ.get("ROBOTWIN_PYARROW_SITE", ""))
if PYARROW_SITE.is_dir():
    sys.path.append(str(PYARROW_SITE))
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from tools.add_nero_eef_pose_to_lerobot import (  # noqa: E402
    STAT_NAMES,
    UrdfChain,
    copy_dataset,
    derive_poses,
    feature_stats,
)


DEFAULT_INPUT = ROOT / "data/place_two_cubes_box_lerobot_v3"
DEFAULT_OUTPUT = ROOT / "data/place_two_cubes_box_lerobot_v3_ee_corrected"
DEFAULT_HDF5 = ROOT / "data/place_two_cubes_box/demo_nero_two_cubes/data"
DEFAULT_URDF = ROOT / "assets/embodiments/nero/nero_with_gripper_description.urdf"

EE_NAMES = [
    f"{arm}_{component}"
    for arm in ("right", "left")
    for component in ("x", "y", "z", "rx", "ry", "rz", "gripper")
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-hdf5-dir", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--gripper-max-width", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fixed_list_array(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]
    )


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    array = fixed_list_array(values)
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, array)
    return table.append_column(name, array)


def pose7_pair_to_ee14(poses: np.ndarray, grippers: np.ndarray) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float64)
    grippers = np.asarray(grippers, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 14:
        raise ValueError(f"Expected dual-arm poses with shape (N, 14), got {poses.shape}")
    if grippers.shape != (len(poses), 2):
        raise ValueError(f"Expected grippers with shape ({len(poses)}, 2), got {grippers.shape}")

    result = np.empty((len(poses), 14), dtype=np.float32)
    for pose_offset, output_offset, gripper_index in ((0, 0, 0), (7, 7, 1)):
        result[:, output_offset : output_offset + 3] = poses[
            :, pose_offset : pose_offset + 3
        ]
        quaternion_wxyz = poses[:, pose_offset + 3 : pose_offset + 7]
        quaternion_xyzw = quaternion_wxyz[:, [1, 2, 3, 0]]
        result[:, output_offset + 3 : output_offset + 6] = Rotation.from_quat(
            quaternion_xyzw
        ).as_rotvec()
        result[:, output_offset + 6] = grippers[:, gripper_index]
    return result


def actual_hdf5_observations(
    episode_ids: np.ndarray,
    frame_ids: np.ndarray,
    hdf5_dir: Path,
    gripper_max_width: float,
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.empty((len(episode_ids), 14), dtype=np.float64)
    grippers = np.empty((len(episode_ids), 2), dtype=np.float64)
    for episode in np.unique(episode_ids):
        indices = np.flatnonzero(episode_ids == episode)
        frames = frame_ids[indices].astype(np.int64)
        path = hdf5_dir / f"episode{int(episode)}.hdf5"
        if not path.is_file():
            raise FileNotFoundError(f"Missing source HDF5 episode: {path}")
        with h5py.File(path) as h5:
            frame_count = len(h5["/endpose/right_endpose"])
            if len(frames) and (frames.min() < 0 or frames.max() >= frame_count):
                raise IndexError(
                    f"Episode {episode} frame range {frames.min()}..{frames.max()} "
                    f"does not fit HDF5 length {frame_count}"
                )
            poses[indices] = np.concatenate(
                (h5["/endpose/right_endpose"][:], h5["/endpose/left_endpose"][:]), axis=1
            )[frames]
            grippers[indices, 0] = h5["/endpose/right_gripper"][:][frames]
            grippers[indices, 1] = h5["/endpose/left_gripper"][:][frames]
    grippers *= gripper_max_width
    return poses, grippers


def pose_errors(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position_errors = []
    rotation_errors = []
    for offset in (0, 7):
        position_errors.append(
            np.linalg.norm(first[:, offset : offset + 3] - second[:, offset : offset + 3], axis=1)
        )
        first_q = first[:, offset + 3 : offset + 7][:, [1, 2, 3, 0]]
        second_q = second[:, offset + 3 : offset + 7][:, [1, 2, 3, 0]]
        rotation_errors.append(
            (Rotation.from_quat(first_q).inv() * Rotation.from_quat(second_q)).magnitude()
        )
    return np.concatenate(position_errors), np.concatenate(rotation_errors)


def quantile_report(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def update_info(output_dir: Path) -> None:
    path = output_dir / "meta/info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    feature = {"dtype": "float32", "shape": [14], "names": EE_NAMES}
    info["features"]["observation.state"] = feature
    info["features"]["action"] = feature
    info["features"].pop("observation.eef_pose", None)
    info["features"].pop("action.eef_pose", None)
    info["eef_pose_convention"] = {
        "reference_frame": "robotwin_world_control",
        "rotation": "rotation_vector_radians",
        "layout": "right_xyz_rotvec_gripper_m,left_xyz_rotvec_gripper_m",
        "observation_source": "same-frame RoboTwin HDF5 endpose feedback",
        "action_source": "URDF FK of the existing next-state joint drive target",
        "gripper_alignment": "observation uses same frame; action uses the same row as action pose",
    }
    path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_global_stats(output_dir: Path, observations: np.ndarray, actions: np.ndarray) -> None:
    path = output_dir / "meta/stats.json"
    stats = json.loads(path.read_text(encoding="utf-8"))
    stats["observation.state"] = feature_stats(observations)
    stats["action"] = feature_stats(actions)
    stats.pop("observation.eef_pose", None)
    stats.pop("action.eef_pose", None)
    path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_episode_stats(
    output_dir: Path,
    episode_ids: np.ndarray,
    observations: np.ndarray,
    actions: np.ndarray,
) -> None:
    paths = sorted((output_dir / "meta/episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata parquet under {output_dir / 'meta/episodes'}")
    for path in paths:
        table = pq.read_table(path)
        stats_columns: dict[str, list] = {}
        table_episode_ids = table["episode_index"].to_pylist()
        for feature_name, values in (
            ("observation.state", observations),
            ("action", actions),
        ):
            per_episode = []
            for episode in table_episode_ids:
                selected = values[episode_ids == episode]
                if not len(selected):
                    raise ValueError(f"Episode metadata references missing episode {episode}")
                per_episode.append(feature_stats(selected))
            for stat_name in STAT_NAMES:
                stats_columns[f"stats/{feature_name}/{stat_name}"] = [
                    item[stat_name] for item in per_episode
                ]
        for name, values in stats_columns.items():
            column = pa.array(values)
            if name in table.column_names:
                table = table.set_column(table.schema.get_field_index(name), name, column)
            else:
                table = table.append_column(name, column)
        pq.write_table(table, path)


def main() -> None:
    cli = parse_args()
    input_dir = cli.input_dir.expanduser().resolve()
    output_dir = cli.output_dir.expanduser().resolve()
    hdf5_dir = cli.source_hdf5_dir.expanduser().resolve()
    urdf = cli.urdf.expanduser().resolve()
    if cli.gripper_max_width <= 0:
        raise ValueError("--gripper-max-width must be positive")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")

    copy_dataset(input_dir, output_dir, cli.overwrite)
    chain = UrdfChain(urdf)
    output_tables = sorted((output_dir / "data").glob("*/*.parquet"))
    if not output_tables:
        raise FileNotFoundError(f"No parquet data under {output_dir / 'data'}")

    all_episode_ids = []
    all_observations = []
    all_actions = []
    old_position_errors = []
    old_rotation_errors = []
    for path in output_tables:
        table = pq.read_table(path)
        required = {"episode_index", "frame_index", "observation.state", "action"}
        missing = sorted(required - set(table.column_names))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        episode_ids = table["episode_index"].to_numpy().astype(np.int64)
        frame_ids = table["frame_index"].to_numpy().astype(np.int64)
        joint_states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        joint_actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if joint_states.shape != (len(table), 16) or joint_actions.shape != (len(table), 16):
            raise ValueError(
                f"Expected 16D joint state/action in {path}, got "
                f"{joint_states.shape} and {joint_actions.shape}"
            )

        actual_poses, actual_grippers = actual_hdf5_observations(
            episode_ids, frame_ids, hdf5_dir, cli.gripper_max_width
        )
        observation_ee = pose7_pair_to_ee14(actual_poses, actual_grippers)
        action_poses = derive_poses(joint_actions, chain)
        action_ee = pose7_pair_to_ee14(action_poses, joint_actions[:, 14:16])
        command_observation_poses = derive_poses(joint_states, chain)
        position_error, rotation_error = pose_errors(command_observation_poses, actual_poses)

        table = replace_column(table, "observation.state", observation_ee)
        table = replace_column(table, "action", action_ee)
        removable = [
            name
            for name in ("observation.eef_pose", "action.eef_pose")
            if name in table.column_names
        ]
        if removable:
            table = table.drop_columns(removable)
        # Avoid retaining stale Hugging Face schema metadata for the former 16D columns.
        table = table.replace_schema_metadata(None)
        pq.write_table(table, path)

        all_episode_ids.append(episode_ids)
        all_observations.append(observation_ee)
        all_actions.append(action_ee)
        old_position_errors.append(position_error)
        old_rotation_errors.append(rotation_error)

    episode_ids = np.concatenate(all_episode_ids)
    observations = np.concatenate(all_observations)
    actions = np.concatenate(all_actions)
    position_errors = np.concatenate(old_position_errors)
    rotation_errors = np.concatenate(old_rotation_errors)
    if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(actions)):
        raise ValueError("Corrected dataset contains non-finite state/action values")

    update_info(output_dir)
    update_global_stats(output_dir, observations, actions)
    update_episode_stats(output_dir, episode_ids, observations, actions)
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_hdf5_dir": str(hdf5_dir),
        "rows": int(len(observations)),
        "episodes": int(len(np.unique(episode_ids))),
        "observation_source": "same-frame HDF5 endpose feedback",
        "action_source": "FK of existing next-state joint action",
        "action_gripper_source": "same existing action row as the FK pose",
        "replaced_observation_command_fk_vs_actual_position_error_m": quantile_report(
            position_errors
        ),
        "replaced_observation_command_fk_vs_actual_rotation_error_rad": quantile_report(
            rotation_errors
        ),
    }
    report_path = output_dir / "meta/corrected_ee_build_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote corrected EE dataset: {output_dir}")


if __name__ == "__main__":
    main()
