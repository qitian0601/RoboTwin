#!/usr/bin/env python
"""Add URDF-derived Nero end-effector poses to a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation


POSE_NAMES = [
    f"{arm}_eef_{component}"
    for arm in ("right", "left")
    for component in ("x", "y", "z", "qw", "qx", "qy", "qz")
]
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}
GLOBAL_TRANS = np.diag([1.0, -1.0, -1.0])
DELTA = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--urdf",
        type=Path,
        default=root / "assets/embodiments/nero/nero_with_gripper_description.urdf",
    )
    parser.add_argument(
        "--source-hdf5-dir",
        type=Path,
        default=None,
        help="Optional RoboTwin HDF5 directory used only to validate the FK result.",
    )
    parser.add_argument(
        "--validation-tolerance",
        type=float,
        default=None,
        help="Fail when HDF5 comparison exceeds this value; by default comparison is report-only.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def transform(xyz=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    result = np.eye(4)
    result[:3, 3] = xyz
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(default if value is None else [float(v) for v in value.split()], dtype=np.float64)


class UrdfChain:
    def __init__(self, urdf_path: Path, target_link: str = "gripper_tcp") -> None:
        root = ET.parse(urdf_path).getroot()
        joints_by_child = {}
        for joint in root.findall("joint"):
            joints_by_child[joint.find("child").attrib["link"]] = joint

        chain = []
        link = target_link
        while link in joints_by_child:
            joint = joints_by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        self.joints = list(reversed(chain))

        movable = [j.attrib["name"] for j in self.joints if j.attrib["type"] in ("revolute", "continuous")]
        expected = [f"joint{i}" for i in range(1, 8)]
        if movable != expected:
            raise ValueError(f"Unexpected Nero URDF chain: {movable}, expected {expected}")

    def forward(self, q: np.ndarray) -> np.ndarray:
        values = {f"joint{i + 1}": float(q[i]) for i in range(7)}
        result = np.eye(4)
        for joint in self.joints:
            origin = joint.find("origin")
            xyz = parse_vector(origin.get("xyz") if origin is not None else None, (0, 0, 0))
            rpy = parse_vector(origin.get("rpy") if origin is not None else None, (0, 0, 0))
            result = result @ transform(xyz, Rotation.from_euler("xyz", rpy).as_matrix())
            if joint.attrib["type"] in ("revolute", "continuous"):
                axis_node = joint.find("axis")
                axis = parse_vector(axis_node.get("xyz") if axis_node is not None else None, (1, 0, 0))
                result = result @ transform(rotation=Rotation.from_rotvec(axis * values[joint.attrib["name"]]).as_matrix())
        return result


def root_transform(arm: str) -> np.ndarray:
    # These are the effective poses from demo_nero_two_cubes.yml after the 0.7 m arm separation.
    x = 0.35 if arm == "right" else -0.35
    quat_wxyz = np.array([0.707107, 0.0, 0.0, -0.707107])
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    return transform((x, -0.65, 0.74), Rotation.from_quat(quat_xyzw).as_matrix())


def robotwin_eef_pose(chain: UrdfChain, q: np.ndarray, arm: str) -> np.ndarray:
    tcp = root_transform(arm) @ chain.forward(q)
    # SAPIEN Joint.global_pose uses a frame rotated by GLOBAL_TRANS from the URDF child link.
    # Robot._trans_endpose then applies GLOBAL_TRANS and DELTA to that joint frame.
    joint_rotation = tcp[:3, :3] @ GLOBAL_TRANS
    rotation = joint_rotation @ GLOBAL_TRANS @ DELTA
    # Mirrors Robot._trans_endpose(is_endpose=False): gripper_bias(0.08) - 0.12.
    position = tcp[:3, 3] + rotation @ np.array([-0.04, 0.0, 0.0])
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return np.concatenate((position, quat_wxyz))


def derive_poses(states: np.ndarray, chain: UrdfChain) -> np.ndarray:
    result = np.empty((len(states), 14), dtype=np.float32)
    for i, state in enumerate(states):
        result[i, :7] = robotwin_eef_pose(chain, state[:7], "right")
        result[i, 7:] = robotwin_eef_pose(chain, state[7:14], "left")
    return result


def feature_stats(values: np.ndarray) -> dict[str, list]:
    stats = {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(),
        "count": [len(values)],
    }
    for name, quantile in QUANTILES.items():
        stats[name] = np.quantile(values, quantile, axis=0).tolist()
    return stats


def copy_dataset(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for source in input_dir.rglob("*"):
        relative = source.relative_to(input_dir)
        destination = output_dir / relative
        if source.is_dir():
            destination.mkdir(exist_ok=True)
        elif "videos" in relative.parts:
            os.link(source, destination)
        else:
            shutil.copy2(source, destination)


def replace_parquet_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    array = pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])
    if name in table.column_names:
        return table.set_column(table.schema.get_field_index(name), name, array)
    return table.append_column(name, array)


def validate_against_hdf5(
    table: pa.Table, poses: np.ndarray, hdf5_dir: Path, tolerance: float | None
) -> tuple[float, float]:
    max_position_error = 0.0
    max_quaternion_error = 0.0
    episodes = table["episode_index"].to_numpy()
    frames = table["frame_index"].to_numpy()
    for episode in np.unique(episodes):
        indices = np.flatnonzero(episodes == episode)
        path = hdf5_dir / f"episode{episode}.hdf5"
        if not path.exists():
            raise FileNotFoundError(f"Missing validation source: {path}")
        with h5py.File(path) as h5:
            expected = np.concatenate(
                (h5["/endpose/right_endpose"][:], h5["/endpose/left_endpose"][:]), axis=1
            )[frames[indices]]
        actual = poses[indices]
        max_position_error = max(
            max_position_error,
            float(np.max(np.abs(actual[:, [0, 1, 2, 7, 8, 9]] - expected[:, [0, 1, 2, 7, 8, 9]]))),
        )
        for offset in (3, 10):
            dot = np.abs(np.sum(actual[:, offset : offset + 4] * expected[:, offset : offset + 4], axis=1))
            max_quaternion_error = max(max_quaternion_error, float(np.max(1.0 - dot)))
    if tolerance is not None and (max_position_error > tolerance or max_quaternion_error > tolerance):
        raise ValueError(
            f"FK validation failed: position={max_position_error:.3g}, quaternion={max_quaternion_error:.3g}, "
            f"tolerance={tolerance:.3g}"
        )
    return max_position_error, max_quaternion_error


def update_episode_stats(output_dir: Path, table: pa.Table, observation: np.ndarray, action: np.ndarray) -> None:
    path = output_dir / "meta/episodes/chunk-000/file-000.parquet"
    episodes_table = pq.read_table(path)
    episode_ids = table["episode_index"].to_numpy()
    stat_columns: dict[str, list] = {}
    for key, values in (("observation.eef_pose", observation), ("action.eef_pose", action)):
        per_episode = [feature_stats(values[episode_ids == episode]) for episode in episodes_table["episode_index"].to_pylist()]
        for stat in STAT_NAMES:
            stat_columns[f"stats/{key}/{stat}"] = [item[stat] for item in per_episode]
    for name, values in stat_columns.items():
        column = pa.array(values)
        if name in episodes_table.column_names:
            episodes_table = episodes_table.set_column(
                episodes_table.schema.get_field_index(name), name, column
            )
        else:
            episodes_table = episodes_table.append_column(name, column)
    pq.write_table(episodes_table, path)


def update_json_metadata(output_dir: Path, observation: np.ndarray, action: np.ndarray) -> None:
    info_path = output_dir / "meta/info.json"
    with info_path.open() as f:
        info = json.load(f)
    pose_feature = {"dtype": "float32", "shape": [14], "names": POSE_NAMES}
    info["features"]["observation.eef_pose"] = pose_feature
    info["features"]["action.eef_pose"] = pose_feature
    info["eef_pose_convention"] = {
        "reference_frame": "world",
        "layout": "right_xyz_qw_qx_qy_qz,left_xyz_qw_qx_qy_qz",
        "source": "Nero URDF FK with RoboTwin end-effector control-frame transforms",
    }
    with info_path.open("w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    stats_path = output_dir / "meta/stats.json"
    with stats_path.open() as f:
        stats = json.load(f)
    stats["observation.eef_pose"] = feature_stats(observation)
    stats["action.eef_pose"] = feature_stats(action)
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("input-dir and output-dir must differ")
    copy_dataset(args.input_dir, args.output_dir, args.overwrite)

    data_path = args.output_dir / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    chain = UrdfChain(args.urdf)
    observation_pose = derive_poses(states, chain)
    action_pose = derive_poses(actions, chain)

    if args.source_hdf5_dir is not None:
        pos_error, quat_error = validate_against_hdf5(
            table, observation_pose, args.source_hdf5_dir, args.validation_tolerance
        )
        print(f"FK validation: max position error={pos_error:.3g} m, quaternion error={quat_error:.3g}")

    table = replace_parquet_column(table, "observation.eef_pose", observation_pose)
    table = replace_parquet_column(table, "action.eef_pose", action_pose)
    pq.write_table(table, data_path)
    update_episode_stats(args.output_dir, table, observation_pose, action_pose)
    update_json_metadata(args.output_dir, observation_pose, action_pose)
    print(f"Wrote {len(table)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
