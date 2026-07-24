#!/usr/bin/env python
"""Convert RoboTwin HDF5 episodes to LeRobot v3 format.

This converter targets datasets shaped like:

    data/episode0.hdf5
    instructions/episode0.json

and writes a LeRobot v3 dataset with:

    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/observation.images.*/chunk-000/file-*.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np


LEROBOT_SRC = Path(
    os.environ.get(
        "ROBOTWIN_LEROBOT_SRC",
        Path(__file__).resolve().parents[1] / "third_party/lerobot_nero_runtime/src",
    )
)
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


BUS_TABLE_ORDER = [
    "right_nero_joint_1",
    "right_nero_joint_2",
    "right_nero_joint_3",
    "right_nero_joint_4",
    "right_nero_joint_5",
    "right_nero_joint_6",
    "right_nero_joint_7",
    "left_nero_joint_1",
    "left_nero_joint_2",
    "left_nero_joint_3",
    "left_nero_joint_4",
    "left_nero_joint_5",
    "left_nero_joint_6",
    "left_nero_joint_7",
    "right_gripper_width",
    "left_gripper_width",
]

CAMERAS = {
    "observation.images.front": {
        "robotwin_key": "head_camera",
        "metadata_key": "front",
        "default_size": (800, 1280),
    },
    "observation.images.left_wrist": {
        "robotwin_key": "left_camera",
        "metadata_key": "left_wrist",
        "default_size": (480, 640),
    },
    "observation.images.right_wrist": {
        "robotwin_key": "right_camera",
        "metadata_key": "right_wrist",
        "default_size": (480, 640),
    },
}

CAMERA_METADATA = {
    "front": {
        "type": "intelrealsense",
        "serial_number_or_name": "324422301659",
        "width": 1280,
        "height": 800,
        "fps": 30,
    },
    "left_wrist": {
        "type": "intelrealsense",
        "serial_number_or_name": "244222077114",
        "width": 640,
        "height": 480,
        "fps": 30,
    },
    "right_wrist": {
        "type": "intelrealsense",
        "serial_number_or_name": "244222070153",
        "width": 640,
        "height": 480,
        "fps": 30,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="RoboTwin task config directory, e.g. data/place_two_cubes_box/demo_nero_two_cubes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the LeRobot v3 dataset will be written.",
    )
    parser.add_argument("--repo-id", default="place_two_cubes_box_lerobot")
    parser.add_argument("--robot-type", default="nero_dual")
    parser.add_argument("--task", default=None, help="Override task prompt for all frames.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--gripper-max-width",
        type=float,
        default=0.1,
        help=(
            "Physical gripper width in meters for RoboTwin's normalized open value 1.0. "
            "The normalized closed value 0.0 maps to 0 meters."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=None,
        help="Optional legacy override: resize all camera frames to this LeRobot feature shape.",
    )
    parser.add_argument(
        "--front-image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(800, 1280),
        help="LeRobot feature shape for observation.images.front.",
    )
    parser.add_argument(
        "--wrist-image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(480, 640),
        help="LeRobot feature shape for both wrist cameras.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Convert only the first N episodes. Useful for a quick smoke test.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete output-dir before conversion.",
    )
    parser.add_argument(
        "--video-codec",
        default="libsvtav1",
        help="Encoder used by LeRobot. Use h264 if AV1 encoder is unavailable.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("next_state", "same_state"),
        default="next_state",
        help=(
            "next_state uses qpos[t] as observation.state and qpos[t+1] as action, dropping the final "
            "frame. same_state writes qpos[t] to both fields."
        ),
    )
    parser.add_argument(
        "--right-left-order",
        action="store_true",
        default=True,
        help="Reorder RoboTwin left+right vector to bus_table right+left+right_gripper+left_gripper order.",
    )
    return parser.parse_args()


def episode_index(path: Path) -> int:
    return int(path.stem.replace("episode", ""))


def load_task(input_dir: Path, ep_idx: int, fallback: str) -> str:
    instr_path = input_dir / "instructions" / f"episode{ep_idx}.json"
    if not instr_path.exists():
        return fallback

    with instr_path.open() as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("seen", "instructions", "unseen"):
            values = data.get(key)
            if values:
                return str(values[0])
    return fallback


def decode_rgb(raw: bytes, image_size_hw: tuple[int, int]) -> np.ndarray:
    # RoboTwin stores these JPEG bytes via cv2.imencode from arrays that are used as RGB elsewhere.
    # Decoding with PIL swaps the effective red/blue channels, so keep OpenCV's channel order here.
    array = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError("Failed to decode image bytes from HDF5.")
    height, width = image_size_hw
    if array.shape[:2] != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_AREA)
    return array


def camera_image_sizes(args: argparse.Namespace) -> dict[str, tuple[int, int]]:
    if args.image_size is not None:
        size = tuple(args.image_size)
        return {key: size for key in CAMERAS}

    return {
        "observation.images.front": tuple(args.front_image_size),
        "observation.images.left_wrist": tuple(args.wrist_image_size),
        "observation.images.right_wrist": tuple(args.wrist_image_size),
    }


def robotwin_to_bus_order(h5: h5py.File, gripper_max_width: float) -> np.ndarray:
    """Return right7 + left7 + right/left gripper widths in meters."""
    left_arm = h5["/joint_action/left_arm"][:]
    right_arm = h5["/joint_action/right_arm"][:]

    def to_width(dataset_key: str) -> np.ndarray:
        normalized = np.asarray(h5[dataset_key][:], dtype=np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ValueError(f"{dataset_key} contains non-finite values.")
        if normalized.min() < -1e-6 or normalized.max() > 1.0 + 1e-6:
            raise ValueError(
                f"{dataset_key} must be normalized to 0..1, got "
                f"{normalized.min():.6f}..{normalized.max():.6f}."
            )
        return np.clip(normalized, 0.0, 1.0) * gripper_max_width

    left_gripper = to_width("/joint_action/left_gripper")
    right_gripper = to_width("/joint_action/right_gripper")
    vector = np.concatenate(
        [
            right_arm,
            left_arm,
            right_gripper[:, None],
            left_gripper[:, None],
        ],
        axis=1,
    )
    return vector.astype(np.float32)


def create_dataset(args: argparse.Namespace) -> LeRobotDataset:
    image_sizes = camera_image_sizes(args)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": BUS_TABLE_ORDER,
        },
        "action": {
            "dtype": "float32",
            "shape": (16,),
            "names": BUS_TABLE_ORDER,
        },
    }
    for key in CAMERAS:
        height, width = image_sizes[key]
        features[key] = {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }

    if args.output_dir.exists() and not args.keep_existing:
        shutil.rmtree(args.output_dir)

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        root=args.output_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=True,
        vcodec=args.video_codec,
    )


def write_camera_metadata(output_dir: Path, fps: int) -> None:
    info_path = output_dir / "meta" / "info.json"
    if not info_path.exists():
        return

    with info_path.open() as f:
        info = json.load(f)

    cameras = json.loads(json.dumps(CAMERA_METADATA))
    for camera in cameras.values():
        camera["fps"] = fps

    info["cameras"] = cameras
    info["camera_features"] = {
        config["metadata_key"]: lerobot_key for lerobot_key, config in CAMERAS.items()
    }

    with info_path.open("w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def convert(args: argparse.Namespace) -> None:
    if args.gripper_max_width <= 0:
        raise ValueError("--gripper-max-width must be greater than zero.")

    data_dir = args.input_dir / "data"
    hdf5_files = sorted(data_dir.glob("episode*.hdf5"), key=episode_index)
    if args.max_episodes is not None:
        hdf5_files = hdf5_files[: args.max_episodes]
    if not hdf5_files:
        raise FileNotFoundError(f"No episode*.hdf5 files found in {data_dir}")

    image_sizes = camera_image_sizes(args)
    dataset = create_dataset(args)
    default_task = args.task or args.input_dir.parent.name.replace("_", " ")

    for count, hdf5_path in enumerate(hdf5_files, start=1):
        ep_idx = episode_index(hdf5_path)
        task = args.task or load_task(args.input_dir, ep_idx, default_task)
        print(f"[{count}/{len(hdf5_files)}] episode{ep_idx}: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as h5:
            qpos = robotwin_to_bus_order(h5, args.gripper_max_width)
            if args.action_mode == "next_state":
                state = qpos[:-1]
                action = qpos[1:]
            else:
                state = qpos
                action = qpos.copy()
            episode_len = action.shape[0]

            for frame_idx in range(episode_len):
                frame = {
                    "observation.state": state[frame_idx],
                    "action": action[frame_idx],
                    "task": task,
                }

                for lerobot_key, config in CAMERAS.items():
                    raw = h5[f"/observation/{config['robotwin_key']}/rgb"][frame_idx]
                    frame[lerobot_key] = decode_rgb(raw, image_sizes[lerobot_key])

                dataset.add_frame(frame)

        dataset.save_episode()

    if hasattr(dataset.meta, "_close_writer"):
        dataset.meta._close_writer()
    if hasattr(dataset, "_close_writer"):
        dataset._close_writer()

    write_camera_metadata(args.output_dir, args.fps)

    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    convert(parse_args())
