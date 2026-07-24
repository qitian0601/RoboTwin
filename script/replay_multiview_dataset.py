#!/usr/bin/env python3
"""Replay saved RoboTwin expert paths into a synchronized multi-view dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import sapien
import yaml

from envs._GLOBAL_CONFIGS import CONFIGS_PATH


TASK_NAME = "place_two_cubes_box"
TASK_CONFIG = "demo_nero_two_cubes"
DEFAULT_SOURCE = ROOT / "data/place_two_cubes_box/demo_nero_two_cubes"
DEFAULT_OUTPUT = ROOT / "data/place_two_cubes_box_multiview_50"
PHYSICS_HZ = 250
DEFAULT_SAMPLE_HZ = 10
DEFAULT_GRIPPER_MAX_WIDTH_M = 0.1
DEFAULT_CAMERA_SHIFTS = {
    "c0": 0.0,
    "c1": 15.0,
    "c2": -15.0,
    "c3": 25.0,
    "c4": -25.0,
}
CAMERA_SOURCE_NAMES = {
    "c0": "head_camera",
    "c1": "multiview_c1",
    "c2": "multiview_c2",
    "c3": "multiview_c3",
    "c4": "multiview_c4",
    "c5": "multiview_c5",
    "c6": "multiview_c6",
    "left_wrist": "left_camera",
    "right_wrist": "right_camera",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def instruction_text(source_root: Path, episode_id: int) -> str:
    data = load_json(source_root / "instructions" / f"episode{episode_id}.json")
    for key in ("seen", "unseen"):
        values = data.get(key, [])
        if values:
            return str(values[0]).strip()
    raise ValueError(f"Episode {episode_id} has no language instruction")


def discover_source_episodes(source_root: Path) -> list[int]:
    data_ids = {
        int(path.stem.removeprefix("episode"))
        for path in (source_root / "data").glob("episode*.hdf5")
    }
    path_ids = {
        int(path.stem.removeprefix("episode"))
        for path in (source_root / "_traj_data").glob("episode*.pkl")
    }
    instruction_ids = {
        int(path.stem.removeprefix("episode"))
        for path in (source_root / "instructions").glob("episode*.json")
    }
    available = sorted(data_ids & path_ids & instruction_ids)
    if not available:
        raise FileNotFoundError(f"No complete source episodes found under {source_root}")
    return available


def build_manifest(
    source_root: Path,
    camera_shifts: dict[str, float],
    total: int = 50,
) -> dict[str, Any]:
    available = discover_source_episodes(source_root)
    grouped: dict[str, list[int]] = {}
    for episode_id in available:
        grouped.setdefault(instruction_text(source_root, episode_id), []).append(episode_id)

    if len(grouped) != 2 or total != 50:
        raise ValueError(
            f"Expected this dataset's two instruction groups and total=50, got {len(grouped)} groups and total={total}"
        )

    groups = sorted(grouped.items(), key=lambda item: min(item[1]))
    selected_groups: list[tuple[str, list[int]]] = []
    anchors = {0, 25, 52}
    for text, ids in groups:
        ids = sorted(ids)
        if len(ids) < 25:
            raise ValueError(f"Instruction group has only {len(ids)} episodes; 25 are required")
        selected = ids[:25]
        required = [episode_id for episode_id in ids if episode_id in anchors]
        for episode_id in required:
            if episode_id not in selected:
                replace_at = next(
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index] not in anchors
                )
                selected[replace_at] = episode_id
                selected.sort()
        selected_groups.append((text, selected))

    first_text, first_ids = selected_groups[0]
    second_text, second_ids = selected_groups[1]
    split_sources = {
        "train": first_ids[:20] + second_ids[:20],
        "val": first_ids[20:23] + second_ids[20:22],
        "test": first_ids[23:25] + second_ids[22:25],
    }

    records: list[dict[str, Any]] = []
    dataset_index = 0
    for split in ("train", "val", "test"):
        for source_episode in sorted(split_sources[split]):
            text = instruction_text(source_root, source_episode)
            records.append(
                {
                    "dataset_index": dataset_index,
                    "source_episode": source_episode,
                    "split": split,
                    "instruction": text,
                    "instruction_id": 0 if text == first_text else 1,
                    "views": ["c0", "c1", "c2", "left_wrist", "right_wrist"]
                    if split != "test"
                    else ["c0", "c1", "c2", "c3", "c4", "left_wrist", "right_wrist"],
                }
            )
            dataset_index += 1

    selected_ids = {record["source_episode"] for record in records}
    return {
        "format_version": "robotwin_multiview_v1",
        "task": TASK_NAME,
        "task_config": TASK_CONFIG,
        "source_root": str(source_root.resolve()),
        "physics_hz": PHYSICS_HZ,
        "sample_hz": DEFAULT_SAMPLE_HZ,
        "camera_pitch_degrees": camera_shifts,
        "split_counts": {split: len(ids) for split, ids in split_sources.items()},
        "instruction_counts": {
            text: sum(record["instruction"] == text for record in records)
            for text, _ in selected_groups
        },
        "reserve_source_episodes": sorted(set(available) - selected_ids),
        "episodes": records,
    }


def load_task_args(config_name: str, task_name: str = TASK_NAME) -> dict[str, Any]:
    with (ROOT / "task_config" / f"{config_name}.yml").open("r", encoding="utf-8") as handle:
        args = yaml.load(handle, Loader=yaml.FullLoader)

    with (Path(CONFIGS_PATH) / "_embodiment_config.yml").open("r", encoding="utf-8") as handle:
        embodiment_types = yaml.load(handle, Loader=yaml.FullLoader)

    embodiment = args["embodiment"]
    if len(embodiment) != 3:
        raise ValueError("The multi-view replay currently expects the two-robot Nero embodiment")

    def embodiment_file(name: str) -> str:
        path = embodiment_types[name]["file_path"]
        if path is None:
            raise ValueError(f"Embodiment {name} has no file_path")
        return path

    def embodiment_config(robot_file: str) -> dict[str, Any]:
        with (Path(robot_file) / "config.yml").open("r", encoding="utf-8") as handle:
            return yaml.load(handle, Loader=yaml.FullLoader)

    args["left_robot_file"] = embodiment_file(embodiment[0])
    args["right_robot_file"] = embodiment_file(embodiment[1])
    args["embodiment_dis"] = embodiment[2]
    args["dual_arm_embodied"] = False
    args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])

    for side, key in (("left", "left_embodiment_config"), ("right", "right_embodiment_config")):
        args[key].update(args.get("embodiment_overrides", {}).get(side, {}))

    args.update(
        {
            "task_name": task_name,
            "task_config": config_name,
            "embodiment_name": f"{embodiment[0]}+{embodiment[1]}",
            "need_plan": False,
            "render_freq": 0,
            "save_data": False,
            "save_freq": None,
        }
    )
    return args


TASK_OBJECT_ATTRIBUTES: dict[str, dict[str, str]] = {
    "place_two_cubes_box": {
        "yellow_cube": "right_cube",
        "green_cube": "left_cube",
    },
    "beat_block_hammer": {
        "hammer": "hammer",
        "block": "block",
    },
    "click_bell": {
        "bell": "bell",
    },
    "pick_dual_bottles": {
        "left_bottle": "bottle1",
        "right_bottle": "bottle2",
    },
}


def task_object_actors(task_name: str, task: Any) -> dict[str, Any]:
    """Return task objects recorded as auxiliary state in the multi-view HDF5."""
    mapping = TASK_OBJECT_ATTRIBUTES.get(task_name, {})
    result = {}
    for output_name, attribute_name in mapping.items():
        actor = getattr(task, attribute_name, None)
        if actor is None:
            raise AttributeError(
                f"Task {task_name!r} has no actor attribute {attribute_name!r}"
            )
        result[output_name] = actor
    return result


def rotate_about_axis(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    theta = np.deg2rad(degrees)
    rotated = (
        vector * np.cos(theta)
        + np.cross(axis, vector) * np.sin(theta)
        + axis * np.dot(axis, vector) * (1.0 - np.cos(theta))
    )
    return rotated / np.linalg.norm(rotated)


def configure_cameras(
    args: dict[str, Any],
    split: str,
    camera_shifts: dict[str, float],
) -> list[str]:
    camera = args["camera"]
    original = copy.deepcopy(camera["static_camera_list"][0])
    position = np.asarray(original["position"], dtype=np.float64)
    forward = np.asarray(original["forward"], dtype=np.float64)
    left = np.asarray(original["left"], dtype=np.float64)
    left /= np.linalg.norm(left)

    labels = ["c0", "c1", "c2"] if split != "test" else ["c0", "c1", "c2", "c3", "c4"]
    static_cameras = []
    for label in labels:
        info = copy.deepcopy(original)
        info["name"] = CAMERA_SOURCE_NAMES[label]
        info["position"] = position.tolist()
        info["forward"] = rotate_about_axis(forward, left, camera_shifts[label]).tolist()
        info["left"] = left.tolist()
        if label != "c0":
            info["preserve_task_rng"] = True
        static_cameras.append(info)
    camera["static_camera_list"] = static_cameras

    base_distortion = copy.deepcopy(camera.get("rgb_distortion", {}).get("head_camera", {}))
    if base_distortion:
        camera["rgb_distortion"] = {
            CAMERA_SOURCE_NAMES[label]: copy.deepcopy(base_distortion)
            for label in labels
        }
    return labels


class StreamingVideoWriter:
    def __init__(self, path: Path, fps: int, frame: np.ndarray, crf: int):
        self.path = path
        self.temporary = path.with_suffix(".part.mp4")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self.shape = (height, width, 3)
        self.frames = 0
        self.process = subprocess.Popen(
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
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-threads",
                "2",
                str(self.temporary),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        if frame.shape != self.shape:
            raise ValueError(f"Video frame shape changed from {self.shape} to {frame.shape}")
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(frame.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait()
        error = self.process.stderr.read().decode("utf-8", errors="replace") if self.process.stderr else ""
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {self.path}: {error.strip()}")
        self.temporary.replace(self.path)

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.temporary.unlink(missing_ok=True)


def scalar(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def pose_vector(pose: Any) -> np.ndarray:
    return np.asarray(list(pose.p) + list(pose.q), dtype=np.float32)


def dynamic_component(actor: Any) -> Any:
    entity = actor.actor if hasattr(actor, "actor") else actor
    for component in entity.get_components():
        if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
            return component
    return None


class EpisodeRecorder:
    def __init__(
        self,
        task: Any,
        episode_dir: Path,
        record: dict[str, Any],
        camera_labels: list[str],
        sample_hz: int,
        max_duration: float,
        gripper_max_width_m: float,
        crf: int,
        camera_shifts: dict[str, Any],
        task_name: str = TASK_NAME,
        object_actors: dict[str, Any] | None = None,
    ):
        if not 0 < sample_hz <= PHYSICS_HZ:
            raise ValueError(f"sample_hz must be in (0, {PHYSICS_HZ}], got {sample_hz}")
        self.task = task
        self.episode_dir = episode_dir
        self.record = record
        self.sample_hz = sample_hz
        self.max_step = int(round(max_duration * PHYSICS_HZ)) if max_duration > 0 else None
        self.gripper_max_width_m = gripper_max_width_m
        self.crf = crf
        self.camera_shifts = camera_shifts
        self.task_name = task_name
        self.object_actors = object_actors or {}
        self.labels = camera_labels + ["left_wrist", "right_wrist"]
        self.writers: dict[str, StreamingVideoWriter] = {}
        self.samples: dict[str, list[np.ndarray | float | int | bool]] = {}
        self.last_step = -1

    def on_step(self, sim_step: int) -> None:
        sample_index = len(self.samples.get("timestamp", []))
        target_step = round(sample_index * PHYSICS_HZ / self.sample_hz)
        if sim_step >= target_step:
            self.capture(sim_step)

    def append(self, key: str, value: Any) -> None:
        self.samples.setdefault(key, []).append(value)

    def arm_state(self, side: str) -> dict[str, np.ndarray | float]:
        robot = self.task.robot
        entity = getattr(robot, f"{side}_entity")
        active_joints = list(getattr(robot, f"{side}_active_joints"))
        arm_joints = list(getattr(robot, f"{side}_arm_joints"))
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        qvel = np.asarray(entity.get_qvel(), dtype=np.float64)
        arm_indices = [active_joints.index(joint) for joint in arm_joints]
        arm_qpos = qpos[arm_indices].astype(np.float32)
        arm_qvel = qvel[arm_indices].astype(np.float32)
        commanded_qpos = np.asarray([scalar(joint.get_drive_target()) for joint in arm_joints], dtype=np.float32)

        gripper = getattr(robot, f"{side}_gripper")[0][0]
        gripper_index = active_joints.index(gripper)
        scale_min, scale_max = getattr(robot, f"{side}_gripper_scale")
        scale_range = float(scale_max - scale_min)
        actual_normalized = np.clip((qpos[gripper_index] - scale_min) / scale_range, 0.0, 1.0)
        command_normalized = np.clip((scalar(gripper.get_drive_target()) - scale_min) / scale_range, 0.0, 1.0)
        gripper_velocity = qvel[gripper_index] / scale_range * self.gripper_max_width_m

        return {
            "qpos": arm_qpos,
            "qvel": arm_qvel,
            "commanded_qpos": commanded_qpos,
            "gripper_width_m": float(actual_normalized * self.gripper_max_width_m),
            "commanded_gripper_width_m": float(command_normalized * self.gripper_max_width_m),
            "gripper_velocity_m_s": float(gripper_velocity),
            "ee_pose": np.asarray(self.task.get_arm_pose(side), dtype=np.float32),
        }

    def object_state(self, actor: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        component = dynamic_component(actor)
        linear_velocity = np.zeros(3, dtype=np.float32)
        angular_velocity = np.zeros(3, dtype=np.float32)
        if component is not None:
            linear_velocity = np.asarray(component.get_linear_velocity(), dtype=np.float32)
            angular_velocity = np.asarray(component.get_angular_velocity(), dtype=np.float32)
        return pose_vector(actor.get_pose()), linear_velocity, angular_velocity

    def capture(self, sim_step: int) -> None:
        if self.max_step is not None and sim_step > self.max_step:
            return
        if sim_step <= self.last_step:
            return

        self.task._update_render()
        self.task.cameras.update_picture()
        rgb = self.task.cameras.get_rgb()
        camera_config = self.task.cameras.get_config()

        for label in self.labels:
            source_name = CAMERA_SOURCE_NAMES[label]
            frame = rgb[source_name]["rgb"]
            writer = self.writers.get(label)
            if writer is None:
                writer = StreamingVideoWriter(
                    self.episode_dir / f"{label}.mp4",
                    self.sample_hz,
                    frame,
                    self.crf,
                )
                self.writers[label] = writer
            writer.write(frame)

            config = camera_config[source_name]
            for matrix_name in ("intrinsic_cv", "extrinsic_cv", "cam2world_gl"):
                self.append(f"cameras/{label}/{matrix_name}", np.asarray(config[matrix_name], dtype=np.float32))

        left = self.arm_state("left")
        right = self.arm_state("right")
        for side, state in (("left", left), ("right", right)):
            for name, value in state.items():
                self.append(f"robot/{side}/{name}", value)

        state_vector = np.concatenate(
            [
                np.asarray(right["qpos"]),
                np.asarray(left["qpos"]),
                [right["gripper_width_m"], left["gripper_width_m"]],
            ]
        ).astype(np.float32)
        action_vector = np.concatenate(
            [
                np.asarray(right["commanded_qpos"]),
                np.asarray(left["commanded_qpos"]),
                [right["commanded_gripper_width_m"], left["commanded_gripper_width_m"]],
            ]
        ).astype(np.float32)
        self.append("observation/state", state_vector)
        self.append("action/commanded", action_vector)

        for name, actor in self.object_actors.items():
            pose, linear_velocity, angular_velocity = self.object_state(actor)
            self.append(f"objects/{name}/pose", pose)
            self.append(f"objects/{name}/linear_velocity", linear_velocity)
            self.append(f"objects/{name}/angular_velocity", angular_velocity)

        self.append("timestamp", len(self.samples.get("timestamp", [])) / self.sample_hz)
        self.append("sim_step", sim_step)
        self.append("frame_index", len(self.samples["timestamp"]) - 1)
        self.append("success", bool(self.task.check_success()))
        self.last_step = sim_step

    def write_hdf5(self, final_success: bool) -> None:
        output = self.episode_dir / "data.hdf5"
        temporary = self.episode_dir / "data.hdf5.tmp"
        with h5py.File(temporary, "w") as handle:
            handle.attrs["format_version"] = "robotwin_multiview_v1"
            handle.attrs["task"] = self.task_name
            handle.attrs["split"] = self.record["split"]
            handle.attrs["dataset_index"] = self.record["dataset_index"]
            handle.attrs["source_episode"] = self.record["source_episode"]
            handle.attrs["instruction"] = self.record["instruction"]
            handle.attrs["physics_hz"] = PHYSICS_HZ
            handle.attrs["sample_hz"] = self.sample_hz
            handle.attrs["gripper_unit"] = "meter"
            handle.attrs["gripper_max_width_m"] = self.gripper_max_width_m
            handle.attrs["final_success"] = bool(final_success)
            handle.attrs["camera_transforms_json"] = json.dumps(self.camera_shifts, sort_keys=True)
            if all(isinstance(value, (int, float)) for value in self.camera_shifts.values()):
                handle.attrs["camera_pitch_degrees_json"] = json.dumps(
                    self.camera_shifts, sort_keys=True
                )
            handle.attrs["state_order"] = "right_arm_7,left_arm_7,right_gripper_width_m,left_gripper_width_m"
            handle.attrs["action_order"] = handle.attrs["state_order"]
            handle.attrs["num_frames"] = len(self.samples.get("timestamp", []))

            for key, values in sorted(self.samples.items()):
                array = np.asarray(values)
                group_path, dataset_name = key.rsplit("/", 1) if "/" in key else ("", key)
                group = handle.require_group(group_path) if group_path else handle
                compression = "gzip" if array.ndim > 1 else None
                group.create_dataset(dataset_name, data=array, compression=compression)

            for label in self.labels:
                handle[f"cameras/{label}"].attrs["video"] = f"{label}.mp4"
                if label in self.camera_shifts:
                    transform = self.camera_shifts[label]
                    if isinstance(transform, dict):
                        for key, value in transform.items():
                            handle[f"cameras/{label}"].attrs[key] = value
                    else:
                        handle[f"cameras/{label}"].attrs["pitch_degrees"] = transform
        temporary.replace(output)

    def finalize(self, final_success: bool) -> None:
        if not self.samples.get("timestamp"):
            raise RuntimeError("No frames were captured")
        for writer in self.writers.values():
            writer.close()
        frame_counts = {writer.frames for writer in self.writers.values()}
        if frame_counts != {len(self.samples["timestamp"])}:
            raise RuntimeError(f"Video/state frame count mismatch: {frame_counts}")
        self.write_hdf5(final_success)

    def abort(self) -> None:
        for writer in self.writers.values():
            writer.abort()


def ffprobe_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def validate_episode(episode_dir: Path, expected_views: list[str], require_success: bool = True) -> dict[str, Any]:
    hdf5_path = episode_dir / "data.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    with h5py.File(hdf5_path, "r") as handle:
        frame_count = int(handle.attrs["num_frames"])
        timestamps = handle["timestamp"][:]
        if len(timestamps) != frame_count or frame_count < 2:
            raise ValueError(f"Invalid frame count in {hdf5_path}")
        expected_delta = 1.0 / float(handle.attrs["sample_hz"])
        if not np.allclose(np.diff(timestamps), expected_delta, atol=1e-8):
            raise ValueError(f"Non-uniform timestamps in {hdf5_path}")
        if require_success and not bool(handle.attrs["final_success"]):
            raise ValueError(f"Replay did not finish successfully: {hdf5_path}")
        required = [
            "observation/state",
            "action/commanded",
            "robot/left/qpos",
            "robot/right/qpos",
        ]
        for key in required:
            if key not in handle or handle[key].shape[0] != frame_count:
                raise ValueError(f"Missing or misaligned dataset {key} in {hdf5_path}")

    video_counts = {}
    for label in expected_views:
        video_path = episode_dir / f"{label}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        video_counts[label] = ffprobe_frame_count(video_path)
        if video_counts[label] != frame_count:
            raise ValueError(f"{video_path} has {video_counts[label]} frames, expected {frame_count}")
    return {"frames": frame_count, "video_frames": video_counts}


def episode_path(output_root: Path, record: dict[str, Any]) -> Path:
    return output_root / record["split"] / f"episode_{record['dataset_index']:03d}"


def replay_episode(
    base_args: dict[str, Any],
    source_root: Path,
    output_root: Path,
    record: dict[str, Any],
    sample_hz: int,
    max_duration: float,
    gripper_max_width_m: float,
    crf: int,
    overwrite: bool,
    camera_shifts: dict[str, float],
) -> dict[str, Any]:
    destination = episode_path(output_root, record)
    success_marker = destination / "_SUCCESS"
    if success_marker.exists() and not overwrite:
        result = validate_episode(destination, record["views"])
        print(f"[skip] {destination.relative_to(output_root)} ({result['frames']} frames)")
        return result
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    args = copy.deepcopy(base_args)
    args["save_path"] = str(source_root)
    camera_labels = configure_cameras(args, record["split"], camera_shifts)
    args["now_ep_num"] = record["source_episode"]
    args["seed"] = int((source_root / "seed.txt").read_text(encoding="utf-8").split()[record["source_episode"]])

    module = importlib.import_module(f"envs.{TASK_NAME}")
    task = getattr(module, TASK_NAME)()
    recorder = None
    try:
        print(
            f"[replay] index={record['dataset_index']:03d} source={record['source_episode']:02d} "
            f"split={record['split']} views={','.join(record['views'])}"
        )
        task.setup_demo(**args)
        trajectory = task.load_tran_data(record["source_episode"])
        args["left_joint_path"] = trajectory["left_joint_path"]
        args["right_joint_path"] = trajectory["right_joint_path"]
        task.set_path_lst(args)

        recorder = EpisodeRecorder(
            task=task,
            episode_dir=destination,
            record=record,
            camera_labels=camera_labels,
            sample_hz=sample_hz,
            max_duration=max_duration,
            gripper_max_width_m=gripper_max_width_m,
            crf=crf,
            camera_shifts=camera_shifts,
            task_name=TASK_NAME,
            object_actors=task_object_actors(TASK_NAME, task),
        )
        task.set_scene_step_callback(recorder.on_step, reset_counter=True)
        recorder.capture(0)
        task.play_once()
        final_success = bool(task.plan_success and task.check_success())
        recorder.finalize(final_success)

        source_instruction = load_json(
            source_root / "instructions" / f"episode{record['source_episode']}.json"
        )
        write_json(
            destination / "instruction.json",
            {
                "language_instruction": record["instruction"],
                "source_instruction": source_instruction,
            },
        )
        digest = hashlib.sha256((destination / "data.hdf5").read_bytes()).hexdigest()
        write_json(
            destination / "episode.json",
            {
                **record,
                "source_seed": args["seed"],
                "sample_hz": sample_hz,
                "physics_hz": PHYSICS_HZ,
                "max_duration_seconds": max_duration,
                "final_success": final_success,
                "data_hdf5_sha256": digest,
            },
        )
        result = validate_episode(destination, record["views"])
        success_marker.write_text("ok\n", encoding="ascii")
        print(f"[done] {destination.relative_to(output_root)} ({result['frames']} frames)")
        return result
    except Exception:
        if recorder is not None:
            recorder.abort()
        raise
    finally:
        if hasattr(task, "scene_step_callback"):
            task.set_scene_step_callback(None)
        if hasattr(task, "scene"):
            task.close_env(clear_cache=True)


def validate_dataset(output_root: Path, manifest: dict[str, Any], selected_ids: set[int] | None = None) -> dict[str, Any]:
    checked = []
    for record in manifest["episodes"]:
        if selected_ids is not None and record["source_episode"] not in selected_ids:
            continue
        destination = episode_path(output_root, record)
        if not (destination / "_SUCCESS").exists():
            raise FileNotFoundError(f"Incomplete episode: {destination}")
        result = validate_episode(destination, record["views"])
        checked.append({"dataset_index": record["dataset_index"], **result})
    return {
        "checked_episodes": len(checked),
        "total_frames": sum(item["frames"] for item in checked),
        "episodes": checked,
    }


def parse_episode_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    result = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def completed_source_episodes(dataset_root: Path) -> set[int]:
    completed = set()
    for marker in dataset_root.glob("*/episode_*/_SUCCESS"):
        metadata_path = marker.parent / "episode.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Completed episode has no metadata: {metadata_path}")
        completed.add(int(load_json(metadata_path)["source_episode"]))
    return completed


def subset_manifest(
    manifest: dict[str, Any],
    selected_ids: set[int],
    excluded_root: Path,
) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    result["collection_role"] = "continuation"
    result["excluded_completed_root"] = str(excluded_root.resolve())
    result["excluded_completed_source_episodes"] = sorted(
        {record["source_episode"] for record in manifest["episodes"]} - selected_ids
    )
    result["episodes"] = [
        record for record in manifest["episodes"] if record["source_episode"] in selected_ids
    ]
    result["split_counts"] = {
        split: sum(record["split"] == split for record in result["episodes"])
        for split in ("train", "val", "test")
    }
    result["instruction_counts"] = {
        instruction: sum(record["instruction"] == instruction for record in result["episodes"])
        for instruction in sorted({record["instruction"] for record in result["episodes"]})
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episodes", help="Comma-separated source episode IDs; default processes all selected 50")
    parser.add_argument(
        "--exclude-completed-root",
        type=Path,
        help="Create a continuation manifest excluding _SUCCESS episodes in another dataset root",
    )
    parser.add_argument("--train-pitch-degrees", type=float, default=DEFAULT_CAMERA_SHIFTS["c1"])
    parser.add_argument("--test-pitch-degrees", type=float, default=DEFAULT_CAMERA_SHIFTS["c3"])
    parser.add_argument("--sample-hz", type=int, default=DEFAULT_SAMPLE_HZ)
    parser.add_argument("--max-duration", type=float, default=20.0, help="Seconds; <=0 records the complete replay")
    parser.add_argument("--gripper-max-width", type=float, default=DEFAULT_GRIPPER_MAX_WIDTH_M)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    source_root = cli.source_root.resolve()
    output_root = cli.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"

    camera_shifts = {
        "c0": 0.0,
        "c1": float(cli.train_pitch_degrees),
        "c2": -float(cli.train_pitch_degrees),
        "c3": float(cli.test_pitch_degrees),
        "c4": -float(cli.test_pitch_degrees),
    }
    expected_manifest = build_manifest(source_root, camera_shifts)
    if cli.exclude_completed_root is not None:
        excluded_root = cli.exclude_completed_root.resolve()
        completed_ids = completed_source_episodes(excluded_root)
        selected_from_full = {
            record["source_episode"] for record in expected_manifest["episodes"]
        } - completed_ids
        expected_manifest = subset_manifest(expected_manifest, selected_from_full, excluded_root)
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest != expected_manifest:
            raise ValueError(
                f"Existing manifest differs from the current source selection: {manifest_path}. "
                "Use a new output directory or remove the unstarted output explicitly."
            )
    else:
        manifest = expected_manifest
        write_json(manifest_path, manifest)

    if cli.sample_hz != int(manifest["sample_hz"]):
        raise ValueError(
            f"This dataset manifest requires {manifest['sample_hz']} Hz, got --sample-hz {cli.sample_hz}"
        )

    selected_ids = parse_episode_ids(cli.episodes)
    available_selected = {record["source_episode"] for record in manifest["episodes"]}
    if selected_ids is not None:
        unknown = selected_ids - available_selected
        if unknown:
            raise ValueError(f"Requested source episodes are not in the selected 50: {sorted(unknown)}")

    print(f"Manifest: {manifest_path}")
    print(f"Splits: {manifest['split_counts']}; reserve: {manifest['reserve_source_episodes']}")
    if cli.prepare_only:
        return
    if cli.validate_only:
        result = validate_dataset(output_root, manifest, selected_ids)
        print(json.dumps(result, indent=2))
        return

    base_args = load_task_args(TASK_CONFIG)
    for record in manifest["episodes"]:
        if selected_ids is not None and record["source_episode"] not in selected_ids:
            continue
        replay_episode(
            base_args=base_args,
            source_root=source_root,
            output_root=output_root,
            record=record,
            sample_hz=cli.sample_hz,
            max_duration=cli.max_duration,
            gripper_max_width_m=cli.gripper_max_width,
            crf=cli.crf,
            overwrite=cli.overwrite,
            camera_shifts=camera_shifts,
        )

    result = validate_dataset(output_root, manifest, selected_ids)
    write_json(output_root / "validation.json", result)
    print(f"Validated {result['checked_episodes']} episodes and {result['total_frames']} synchronized frames")


if __name__ == "__main__":
    main()
