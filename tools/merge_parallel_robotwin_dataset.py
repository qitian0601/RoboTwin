#!/usr/bin/env python3
"""Merge isolated RoboTwin collection workers without duplicating large files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py


BOTTLE_SEEN = [
    "Use both arms simultaneously to pick up the two bottles and move them to the raised target positions.",
    "Grasp both bottles at the same time, one with each arm, and carry them to the raised targets.",
    "Pick up both bottles simultaneously with the two grippers and move them to their target positions.",
    "With one arm per bottle, lift both bottles together and carry them to the raised targets.",
]
BOTTLE_UNSEEN = [
    "Lift the two bottles together using both arms and move them to the elevated goal positions.",
    "Simultaneously grasp one bottle in each gripper and carry both to the raised goals.",
]
HAMMER_SEEN = [
    "Use the {arm} arm to pick up the hammer and strike the block.",
    "Grasp the hammer with the {arm} arm, lift it, and hit the block.",
    "With the {arm} arm, pick up the hammer and bring its head down onto the block.",
    "Take the hammer in the {arm} gripper and strike the block.",
]
HAMMER_UNSEEN = [
    "Use the {arm} manipulator to grasp the hammer and hit the block.",
    "Lift the hammer with the {arm} arm and make contact with the block.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("pick_dual_bottles", "beat_block_hammer"), required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, action="append", required=True)
    parser.add_argument("--expected-total", type=int, default=120)
    return parser.parse_args()


def episode_id(path: Path) -> int:
    return int(path.stem.replace("episode", ""))


def hardlink_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def instruction(task: str, idx: int, scene_entry: dict) -> dict[str, list[str]]:
    if task == "pick_dual_bottles":
        return {
            "seen": [BOTTLE_SEEN[idx % len(BOTTLE_SEEN)]],
            "unseen": [BOTTLE_UNSEEN[idx % len(BOTTLE_UNSEEN)]],
        }
    arm = str(scene_entry.get("info", {}).get("{a}", "")).lower()
    if arm not in {"left", "right"}:
        raise ValueError(f"Hammer episode {idx} has invalid arm label: {arm!r}")
    return {
        "seen": [HAMMER_SEEN[idx % len(HAMMER_SEEN)].format(arm=arm)],
        "unseen": [HAMMER_UNSEEN[idx % len(HAMMER_UNSEEN)].format(arm=arm)],
    }


def validate_hdf5(path: Path) -> int:
    required = (
        "joint_action/vector",
        "observation/head_camera/rgb",
        "observation/left_camera/rgb",
        "observation/right_camera/rgb",
    )
    with h5py.File(path, "r") as handle:
        lengths = [len(handle[key]) for key in required]
    if len(set(lengths)) != 1 or lengths[0] < 50:
        raise ValueError(f"Invalid frame lengths in {path}: {lengths}")
    return lengths[0]


def main() -> None:
    args = parse_args()
    output = args.output_root.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ("data", "video", "_traj_data", "instructions"):
        (output / subdir).mkdir(exist_ok=True)

    merged_scene: dict[str, dict] = {}
    merged_seeds: list[int] = []
    mappings: list[dict] = []
    frame_counts: list[int] = []
    link_modes: dict[str, int] = {}
    global_idx = 0

    for worker_arg in args.worker_root:
        worker = worker_arg.expanduser().resolve()
        data_files = sorted((worker / "data").glob("episode*.hdf5"), key=episode_id)
        if not data_files:
            raise FileNotFoundError(f"No HDF5 episodes in worker root: {worker}")
        local_ids = [episode_id(path) for path in data_files]
        if local_ids != list(range(len(local_ids))):
            raise ValueError(f"Worker episodes are not contiguous from zero: {worker}: {local_ids}")

        seeds = [int(value) for value in (worker / "seed.txt").read_text().split()]
        scene = json.loads((worker / "scene_info.json").read_text())
        if len(seeds) < len(data_files):
            raise ValueError(f"Worker has fewer seeds than episodes: {worker}")

        for local_idx, data_file in enumerate(data_files):
            scene_key = f"episode_{local_idx}"
            if scene_key not in scene:
                raise KeyError(f"Missing {scene_key} in {worker / 'scene_info.json'}")
            source_files = {
                "data": (
                    data_file,
                    output / "data" / f"episode{global_idx}.hdf5",
                ),
                "_traj_data": (
                    worker / "_traj_data" / f"episode{local_idx}.pkl",
                    output / "_traj_data" / f"episode{global_idx}.pkl",
                ),
                "video": (
                    worker / "video" / f"episode{local_idx}.mp4",
                    output / "video" / f"episode{global_idx}.mp4",
                ),
                "video_left": (
                    worker / "video" / f"episode{local_idx}_left_camera.mp4",
                    output / "video" / f"episode{global_idx}_left_camera.mp4",
                ),
                "video_right": (
                    worker / "video" / f"episode{local_idx}_right_camera.mp4",
                    output / "video" / f"episode{global_idx}_right_camera.mp4",
                ),
            }
            for kind, (source, destination) in source_files.items():
                if not source.is_file():
                    raise FileNotFoundError(source)
                mode = hardlink_or_copy(source, destination)
                link_modes[mode] = link_modes.get(mode, 0) + 1

            frame_counts.append(validate_hdf5(output / "data" / f"episode{global_idx}.hdf5"))
            merged_scene[f"episode_{global_idx}"] = scene[scene_key]
            merged_seeds.append(seeds[local_idx])
            payload = instruction(args.task, global_idx, scene[scene_key])
            (output / "instructions" / f"episode{global_idx}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            mappings.append(
                {
                    "episode": global_idx,
                    "worker_root": str(worker),
                    "worker_episode": local_idx,
                    "seed": seeds[local_idx],
                    "frames": frame_counts[-1],
                }
            )
            global_idx += 1

    if global_idx != args.expected_total:
        raise ValueError(f"Expected {args.expected_total} episodes, merged {global_idx}")

    (output / "seed.txt").write_text(" ".join(map(str, merged_seeds)) + " ", encoding="utf-8")
    (output / "scene_info.json").write_text(
        json.dumps(merged_scene, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    arm_counts: dict[str, int] = {}
    if args.task == "beat_block_hammer":
        for entry in merged_scene.values():
            arm = str(entry.get("info", {}).get("{a}", "")).lower()
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
        expected_per_arm = args.expected_total // 2
        if arm_counts != {"left": expected_per_arm, "right": expected_per_arm}:
            raise ValueError(f"Hammer arm balance is incorrect: {arm_counts}")

    report = {
        "task": args.task,
        "task_config": args.task_config,
        "episodes": global_idx,
        "total_frames": sum(frame_counts),
        "mean_frames": sum(frame_counts) / len(frame_counts),
        "min_frames": min(frame_counts),
        "max_frames": max(frame_counts),
        "arm_counts": arm_counts,
        "link_modes": link_modes,
        "mappings": mappings,
    }
    (output / "generation_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "mappings"}, indent=2))


if __name__ == "__main__":
    main()
