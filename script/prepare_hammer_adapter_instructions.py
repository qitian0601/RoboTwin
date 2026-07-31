#!/usr/bin/env python3

"""Write arm-consistent instructions for balanced hammer Adapter episodes."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


PROMPTS = {
    "left": "Use the left arm to pick up the hammer and strike the block.",
    "right": "Grasp the hammer with the right arm, lift it, and hit the block.",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, ensure_ascii=False, indent=2)
        destination.write("\n")


def episode_arm(source_root: Path, episode_id: int) -> str:
    trajectory_path = source_root / "_traj_data" / f"episode{episode_id}.pkl"
    with trajectory_path.open("rb") as trajectory_file:
        trajectory = pickle.load(trajectory_file)
    left_active = bool(trajectory.get("left_joint_path"))
    right_active = bool(trajectory.get("right_joint_path"))
    if left_active == right_active:
        raise ValueError(
            f"Episode {episode_id} must have exactly one active arm: {trajectory_path}"
        )
    return "left" if left_active else "right"


def update_multiview(
    multiview_root: Path,
    cache_root: Path | None,
    episode_prompts: dict[int, str],
) -> None:
    manifest_path = multiview_root / "manifest.json"
    manifest = load_json(manifest_path)
    instruction_values = sorted(set(episode_prompts.values()))
    for record in manifest["episodes"]:
        episode_id = int(record["dataset_index"])
        prompt = episode_prompts[episode_id]
        record["instruction"] = prompt
        record["instruction_id"] = instruction_values.index(prompt)

        episode_dir = (
            multiview_root
            / record["split"]
            / f"episode_{episode_id:03d}"
        )
        episode_metadata = load_json(episode_dir / "episode.json")
        episode_metadata["instruction"] = prompt
        episode_metadata["instruction_id"] = instruction_values.index(prompt)
        write_json(episode_dir / "episode.json", episode_metadata)
        write_json(
            episode_dir / "instruction.json",
            {
                "language_instruction": prompt,
                "source_instruction": {"seen": [prompt], "unseen": []},
            },
        )
    write_json(manifest_path, manifest)

    if cache_root is None:
        return
    for split in ("train", "val"):
        index_path = cache_root / f"{split}_index.json"
        if not index_path.exists():
            continue
        index = load_json(index_path)
        for episode in index["episodes"]:
            episode_id = int(Path(episode["episode_dir"]).name.removeprefix("episode_"))
            episode["instruction"] = episode_prompts[episode_id]
        write_json(index_path, index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--episode-count", required=True, type=int)
    parser.add_argument("--multiview-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    episode_prompts = {}
    arm_counts = {"left": 0, "right": 0}
    for episode_id in range(args.episode_count):
        arm = episode_arm(source_root, episode_id)
        arm_counts[arm] += 1
        prompt = PROMPTS[arm]
        episode_prompts[episode_id] = prompt
        write_json(
            source_root / "instructions" / f"episode{episode_id}.json",
            {"seen": [prompt], "unseen": []},
        )

    if args.multiview_root is not None:
        update_multiview(
            args.multiview_root.expanduser().resolve(),
            args.cache_root.expanduser().resolve() if args.cache_root else None,
            episode_prompts,
        )
    print(
        f"Prepared {args.episode_count} hammer instructions: "
        f"left={arm_counts['left']}, right={arm_counts['right']}"
    )


if __name__ == "__main__":
    main()
