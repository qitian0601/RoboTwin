#!/usr/bin/env python3

"""Replay LeRobot-v3 episodes with six synchronized third-person camera shifts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PYARROW_SITE = Path(os.environ.get("ROBOTWIN_PYARROW_SITE", ""))
if PYARROW_SITE.is_dir():
    sys.path.append(str(PYARROW_SITE))
import pyarrow.parquet as pq  # noqa: E402

from script.replay_multiview_dataset import (  # noqa: E402
    CAMERA_SOURCE_NAMES,
    DEFAULT_GRIPPER_MAX_WIDTH_M,
    EpisodeRecorder,
    StreamingVideoWriter,
    load_json,
    load_task_args,
    parse_episode_ids,
    rotate_about_axis,
    task_object_actors,
    validate_dataset,
    validate_episode,
    write_json,
)


TASK_NAME = "place_two_cubes_box"
TASK_CONFIG = "demo_nero_two_cubes"
DEFAULT_TEMPLATE = ROOT / "data/place_two_cubes_box_lerobot_v3"
DEFAULT_SOURCE = ROOT / "data/place_two_cubes_box/demo_nero_two_cubes"
DEFAULT_OUTPUT = ROOT / "data/place_two_cubes_box_adapter_multiview_50"
VIEWS = ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]
ALL_VIEWS = VIEWS + ["left_wrist", "right_wrist"]
CAMERA_TRANSFORMS: dict[str, dict[str, float | str]] = {
    "c0": {"kind": "canonical", "value": 0.0},
    "c1": {"kind": "yaw", "degrees": -10.0},
    "c2": {"kind": "yaw", "degrees": 10.0},
    "c3": {"kind": "pitch", "degrees": -7.0},
    "c4": {"kind": "pitch", "degrees": 7.0},
    "c5": {"kind": "translation", "meters": 0.1},
    "c6": {"kind": "translation", "meters": -0.1},
}


def load_template_records(
    template_root: Path, episode_count: int
) -> list[dict[str, Any]]:
    episode_files = sorted((template_root / "meta/episodes").glob("*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"No episode metadata under {template_root}")
    rows = []
    for path in episode_files:
        table = pq.read_table(path, columns=["episode_index", "length", "tasks"])
        rows.extend(table.to_pylist())
    rows.sort(key=lambda row: int(row["episode_index"]))
    selected = rows[:episode_count]
    episode_ids = [int(row["episode_index"]) for row in selected]
    if episode_ids != list(range(episode_count)):
        raise ValueError(
            f"Expected template episodes 0..{episode_count - 1}, got {episode_ids}"
        )
    return selected


def load_template_episode(
    template_root: Path, episode_id: int
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[int, list[float], list[float]]] = []
    for path in sorted((template_root / "data").glob("*/*.parquet")):
        table = pq.read_table(
            path,
            columns=["episode_index", "frame_index", "observation.state", "action"],
        )
        episode_indices = table["episode_index"].to_numpy()
        frame_indices = table["frame_index"].to_numpy()
        states = table["observation.state"]
        actions = table["action"]
        for row_index in np.flatnonzero(episode_indices == episode_id):
            rows.append(
                (
                    int(frame_indices[row_index]),
                    states[row_index].as_py(),
                    actions[row_index].as_py(),
                )
            )
    rows.sort(key=lambda row: row[0])
    if [row[0] for row in rows] != list(range(len(rows))):
        raise ValueError(f"Template episode {episode_id} has non-contiguous frames")
    return (
        np.asarray([row[1] for row in rows], dtype=np.float32),
        np.asarray([row[2] for row in rows], dtype=np.float32),
    )


def build_manifest(
    template_root: Path,
    source_root: Path,
    sample_hz: int,
    task_name: str,
    task_config: str,
    episode_count: int,
    train_count: int,
) -> dict[str, Any]:
    if episode_count <= 0:
        raise ValueError("episode_count must be positive")
    if not 0 <= train_count <= episode_count:
        raise ValueError("train_count must be between 0 and episode_count")
    records = []
    for row in load_template_records(template_root, episode_count):
        episode_id = int(row["episode_index"])
        tasks = row["tasks"]
        if len(tasks) != 1:
            raise ValueError(f"Episode {episode_id} must have one task, got {tasks}")
        instruction = str(tasks[0]).strip()
        source_instruction_path = source_root / "instructions" / f"episode{episode_id}.json"
        if source_instruction_path.exists():
            source_instruction = load_json(source_instruction_path)
            available_instructions = source_instruction.get("seen", []) + source_instruction.get("unseen", [])
            normalized_instructions = [str(value).strip() for value in available_instructions]
            generic_fallback = task_name.replace("_", " ")
            if instruction == generic_fallback and normalized_instructions:
                instruction = normalized_instructions[0]
            elif instruction not in set(normalized_instructions):
                raise ValueError(
                    f"Template/source instruction mismatch for episode {episode_id}: {instruction!r}"
                )
        records.append(
            {
                "dataset_index": episode_id,
                "source_episode": episode_id,
                "template_episode": episode_id,
                "template_num_frames": int(row["length"]),
                "split": "train" if episode_id < train_count else "val",
                "instruction": instruction,
                "instruction_id": 0,
                "views": ALL_VIEWS,
                "pair_quality": 1.0,
            }
        )
    instruction_values = sorted({record["instruction"] for record in records})
    for record in records:
        record["instruction_id"] = instruction_values.index(record["instruction"])
    return {
        "format_version": "robotwin_adapter_multiview_v1",
        "task": task_name,
        "task_config": task_config,
        "template_root": str(template_root.resolve()),
        "source_root": str(source_root.resolve()),
        "source_selection": (
            f"first {episode_count} LeRobot-v3 episodes "
            f"(episode_index 0..{episode_count - 1})"
        ),
        "physics_hz": 250,
        "sample_hz": sample_hz,
        "camera_transforms": CAMERA_TRANSFORMS,
        "split_counts": {
            "train": train_count,
            "val": episode_count - train_count,
        },
        "episodes": records,
    }


def configure_cameras(args: dict[str, Any]) -> list[str]:
    camera = args["camera"]
    original = copy.deepcopy(camera["static_camera_list"][0])
    canonical_position = np.asarray(original["position"], dtype=np.float64)
    canonical_forward = np.asarray(original["forward"], dtype=np.float64)
    canonical_forward /= np.linalg.norm(canonical_forward)
    canonical_left = np.asarray(original["left"], dtype=np.float64)
    canonical_left /= np.linalg.norm(canonical_left)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    static_cameras = []
    for label in VIEWS:
        transform = CAMERA_TRANSFORMS[label]
        position = canonical_position.copy()
        forward = canonical_forward.copy()
        left = canonical_left.copy()
        kind = transform["kind"]
        if kind == "yaw":
            degrees = float(transform["degrees"])
            forward = rotate_about_axis(forward, world_up, degrees)
            left = rotate_about_axis(left, world_up, degrees)
        elif kind == "pitch":
            forward = rotate_about_axis(forward, canonical_left, float(transform["degrees"]))
        elif kind == "translation":
            position += canonical_forward * float(transform["meters"])
        elif kind != "canonical":
            raise ValueError(f"Unsupported camera transform {kind!r}")

        info = copy.deepcopy(original)
        info.update(
            {
                "name": CAMERA_SOURCE_NAMES[label],
                "position": position.tolist(),
                "forward": forward.tolist(),
                "left": left.tolist(),
            }
        )
        if label != "c0":
            info["preserve_task_rng"] = True
        static_cameras.append(info)
    camera["static_camera_list"] = static_cameras

    base_distortion = copy.deepcopy(camera.get("rgb_distortion", {}).get("head_camera", {}))
    if base_distortion:
        camera["rgb_distortion"] = {
            CAMERA_SOURCE_NAMES[label]: copy.deepcopy(base_distortion) for label in VIEWS
        }
    return VIEWS


def align_state_sequences(template: np.ndarray, replay: np.ndarray) -> np.ndarray:
    """Map every template frame to a monotonic replay frame using DTW."""
    scale = np.maximum(np.std(template, axis=0), 0.02)
    pair_cost = np.mean(((template[:, None, :] - replay[None, :, :]) / scale) ** 2, axis=-1)
    rows, columns = pair_cost.shape
    accumulated = np.full((rows, columns), np.inf, dtype=np.float64)
    predecessor = np.zeros((rows, columns), dtype=np.uint8)
    accumulated[0, 0] = pair_cost[0, 0]
    for row in range(rows):
        for column in range(columns):
            if row == 0 and column == 0:
                continue
            candidates = (
                accumulated[row - 1, column - 1] if row > 0 and column > 0 else np.inf,
                accumulated[row - 1, column] if row > 0 else np.inf,
                accumulated[row, column - 1] if column > 0 else np.inf,
            )
            direction = int(np.argmin(candidates))
            accumulated[row, column] = pair_cost[row, column] + candidates[direction]
            predecessor[row, column] = direction

    path = []
    row, column = rows - 1, columns - 1
    while True:
        path.append((row, column))
        if row == 0 and column == 0:
            break
        direction = predecessor[row, column]
        if direction == 0:
            row -= 1
            column -= 1
        elif direction == 1:
            row -= 1
        else:
            column -= 1
    path.reverse()

    mapping = np.empty(rows, dtype=np.int64)
    for template_index in range(rows):
        candidates = [replay_index for row_index, replay_index in path if row_index == template_index]
        mapping[template_index] = min(
            candidates,
            key=lambda replay_index: pair_cost[template_index, replay_index],
        )
    if np.any(np.diff(mapping) < 0):
        raise RuntimeError("DTW produced a non-monotonic frame mapping")
    return mapping


def rewrite_video_with_mapping(path: Path, mapping: np.ndarray, fps: int, crf: int) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open replay video {path}")
    aligned_path = path.with_name(f"{path.stem}.aligned.mp4")
    writer = None
    mapping_position = 0
    frame_index = 0
    try:
        while mapping_position < len(mapping):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(
                    f"Video {path} ended before replay frame {int(mapping[mapping_position])}"
                )
            while mapping_position < len(mapping) and mapping[mapping_position] == frame_index:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if writer is None:
                    writer = StreamingVideoWriter(aligned_path, fps, rgb, crf)
                writer.write(rgb)
                mapping_position += 1
            frame_index += 1
        if writer is None:
            raise RuntimeError(f"No aligned frames selected from {path}")
        writer.close()
        aligned_path.replace(path)
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    finally:
        capture.release()


def finalize_aligned_episode(
    recorder: EpisodeRecorder,
    template_states: np.ndarray,
    template_actions: np.ndarray,
    final_success: bool,
) -> tuple[np.ndarray, float]:
    for writer in recorder.writers.values():
        writer.close()
    replay_states = np.asarray(recorder.samples["observation/state"], dtype=np.float32)
    mapping = align_state_sequences(template_states, replay_states)
    aligned_states = replay_states[mapping]
    state_max_error = float(np.max(np.abs(aligned_states - template_states)))

    for label in recorder.labels:
        rewrite_video_with_mapping(
            recorder.episode_dir / f"{label}.mp4",
            mapping,
            recorder.sample_hz,
            recorder.crf,
        )
    for key, values in list(recorder.samples.items()):
        recorder.samples[key] = list(np.asarray(values)[mapping])
    recorder.samples["observation/state"] = list(template_states)
    recorder.samples["action/commanded"] = list(template_actions)
    recorder.samples["timestamp"] = list(
        np.arange(len(template_states), dtype=np.float64) / recorder.sample_hz
    )
    recorder.samples["frame_index"] = list(np.arange(len(template_states), dtype=np.int64))
    recorder.write_hdf5(final_success)
    return mapping, state_max_error


def episode_path(output_root: Path, record: dict[str, Any]) -> Path:
    return output_root / record["split"] / f"episode_{record['dataset_index']:03d}"


def replay_episode(
    base_args: dict[str, Any],
    template_root: Path,
    source_root: Path,
    output_root: Path,
    record: dict[str, Any],
    sample_hz: int,
    max_duration: float,
    gripper_max_width: float,
    crf: int,
    overwrite: bool,
    task_name: str,
) -> None:
    destination = episode_path(output_root, record)
    marker = destination / "_SUCCESS"
    if marker.exists() and not overwrite:
        result = validate_episode(destination, record["views"])
        if result["frames"] != record["template_num_frames"]:
            raise ValueError(f"Completed episode length mismatch: {destination}")
        print(f"[skip] {destination.relative_to(output_root)} ({result['frames']} frames)")
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    args = copy.deepcopy(base_args)
    args["save_path"] = str(source_root)
    camera_labels = configure_cameras(args)
    episode_id = record["source_episode"]
    args["now_ep_num"] = episode_id
    args["seed"] = int((source_root / "seed.txt").read_text().split()[episode_id])
    task = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
    recorder = None
    try:
        print(
            f"[replay] episode={episode_id:03d} split={record['split']} "
            f"views={','.join(record['views'])}"
        )
        task.setup_demo(**args)
        template_states, template_actions = load_template_episode(template_root, episode_id)
        if len(template_states) != record["template_num_frames"]:
            raise ValueError(
                f"Template episode {episode_id} metadata says {record['template_num_frames']} frames, "
                f"data contains {len(template_states)}"
            )
        recorder = EpisodeRecorder(
            task=task,
            episode_dir=destination,
            record=record,
            camera_labels=camera_labels,
            sample_hz=sample_hz,
            max_duration=max_duration,
            gripper_max_width_m=gripper_max_width,
            crf=crf,
            camera_shifts=CAMERA_TRANSFORMS,
            task_name=task_name,
            object_actors=task_object_actors(task_name, task),
        )
        task.set_scene_step_callback(recorder.on_step, reset_counter=True)
        recorder.capture(0)
        trajectory = task.load_tran_data(episode_id)
        args["left_joint_path"] = trajectory["left_joint_path"]
        args["right_joint_path"] = trajectory["right_joint_path"]
        task.set_path_lst(args)
        task.play_once()
        final_success = bool(task.plan_success and task.check_success())
        mapping, state_max_error = finalize_aligned_episode(
            recorder,
            template_states,
            template_actions,
            final_success,
        )
        print(
            f"[alignment] episode={episode_id:03d} replay_frames={len(mapping) and int(mapping[-1]) + 1} "
            f"output_frames={len(mapping)} state_max_error={state_max_error:.6f}"
        )

        source_instruction_path = source_root / "instructions" / f"episode{episode_id}.json"
        source_instruction = (
            load_json(source_instruction_path)
            if source_instruction_path.exists()
            else {"seen": [record["instruction"]], "unseen": []}
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
                "physics_hz": 250,
                "max_duration_seconds": max_duration,
                "final_success": final_success,
                "state_replay_max_error": state_max_error,
                "replay_frame_indices": mapping.tolist(),
                "camera_transforms": CAMERA_TRANSFORMS,
                "data_hdf5_sha256": digest,
            },
        )
        result = validate_episode(destination, record["views"])
        if result["frames"] != record["template_num_frames"]:
            raise ValueError(
                f"Episode {episode_id} produced {result['frames']} frames, "
                f"template has {record['template_num_frames']}"
            )
        marker.write_text("ok\n", encoding="ascii")
        print(f"[done] {destination.relative_to(output_root)} ({result['frames']} frames)")
    except Exception:
        if recorder is not None:
            recorder.abort()
        raise
    finally:
        if hasattr(task, "scene_step_callback"):
            task.set_scene_step_callback(None)
        if hasattr(task, "scene"):
            task.close_env(clear_cache=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-root", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task-name", default=TASK_NAME)
    parser.add_argument("--task-config", default=TASK_CONFIG)
    parser.add_argument("--episode-count", type=int, default=50)
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--episodes", help="Comma-separated episode IDs")
    parser.add_argument("--sample-hz", type=int, default=30)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--gripper-max-width", type=float, default=DEFAULT_GRIPPER_MAX_WIDTH_M)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--success-hold-s", type=float)
    parser.add_argument("--success-max-linear-velocity", type=float)
    parser.add_argument("--success-max-angular-velocity", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    template_root = cli.template_root.expanduser().resolve()
    source_root = cli.source_root.expanduser().resolve()
    output_root = cli.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    expected_manifest = build_manifest(
        template_root,
        source_root,
        cli.sample_hz,
        cli.task_name,
        cli.task_config,
        cli.episode_count,
        cli.train_count,
    )
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest != expected_manifest:
            raise ValueError(
                f"Existing manifest differs: {manifest_path}. Use a new output directory."
            )
    else:
        manifest = expected_manifest
        write_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")
    print(f"Splits: {manifest['split_counts']}; views: {','.join(ALL_VIEWS)}")
    if cli.prepare_only:
        return

    selected_ids = parse_episode_ids(cli.episodes)
    if selected_ids is not None:
        unknown = selected_ids - set(range(cli.episode_count))
        if unknown:
            raise ValueError(
                f"Episodes must be in 0..{cli.episode_count - 1}, got {sorted(unknown)}"
            )
    if cli.validate_only:
        result = validate_dataset(output_root, manifest, selected_ids)
        print(json.dumps(result, indent=2))
        return

    base_args = load_task_args(cli.task_config, cli.task_name)
    if cli.success_hold_s is not None:
        base_args["success_hold_s"] = cli.success_hold_s
    if cli.success_max_linear_velocity is not None:
        base_args["success_max_linear_velocity_m_s"] = cli.success_max_linear_velocity
    if cli.success_max_angular_velocity is not None:
        base_args["success_max_angular_velocity_rad_s"] = cli.success_max_angular_velocity
    for record in manifest["episodes"]:
        if selected_ids is not None and record["source_episode"] not in selected_ids:
            continue
        replay_episode(
            base_args,
            template_root,
            source_root,
            output_root,
            record,
            cli.sample_hz,
            cli.max_duration,
            cli.gripper_max_width,
            cli.crf,
            cli.overwrite,
            cli.task_name,
        )
    result = validate_dataset(output_root, manifest, selected_ids)
    if selected_ids is None:
        write_json(output_root / "validation.json", result)
    print(f"Validated {result['checked_episodes']} episodes and {result['total_frames']} frames")


if __name__ == "__main__":
    main()
