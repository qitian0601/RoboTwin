#!/usr/bin/env python3

"""Merge Adapter cache indexes while balancing their effective frame counts."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must have the form TASK=INDEX.json")
    task, path = value.split("=", 1)
    if not task.strip() or not path.strip():
        raise argparse.ArgumentTypeError("source must have the form TASK=INDEX.json")
    return task.strip(), Path(path).expanduser().resolve()


def select_repeats(episodes: list[dict], extra_frames: int) -> list[dict]:
    """Select a deterministic prefix whose frames best match the requested deficit."""
    if extra_frames <= 0 or not episodes:
        return []
    selected: list[dict] = []
    accumulated = 0
    cursor = 0
    while accumulated < extra_frames:
        episode = episodes[cursor % len(episodes)]
        frames = int(episode["num_frames"])
        previous_error = abs(extra_frames - accumulated)
        next_error = abs(extra_frames - (accumulated + frames))
        if selected and next_error > previous_error:
            break
        selected.append(copy.deepcopy(episode))
        accumulated += frames
        cursor += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=parse_source)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--balance-frames",
        action="store_true",
        help="Repeat complete episodes until each task is near the largest task's frame count.",
    )
    args = parser.parse_args()

    if len(args.source) < 2:
        raise ValueError("At least two --source entries are required")
    task_names = [task for task, _ in args.source]
    if len(task_names) != len(set(task_names)):
        raise ValueError(f"Task names must be unique: {task_names}")

    loaded: list[tuple[str, Path, dict]] = []
    for task, path in args.source:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as source_file:
            index = json.load(source_file)
        episodes = index.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"No episodes in {path}")
        loaded.append((task, path, index))

    target_frames = max(
        sum(int(episode["num_frames"]) for episode in index["episodes"])
        for _, _, index in loaded
    )
    merged_episodes: list[dict] = []
    summary: list[dict] = []
    for task, path, index in loaded:
        originals = [copy.deepcopy(episode) for episode in index["episodes"]]
        original_frames = sum(int(episode["num_frames"]) for episode in originals)
        repeats = select_repeats(originals, target_frames - original_frames) if args.balance_frames else []
        for repeat_index, episode in enumerate(repeats):
            episode["balance_repeat"] = repeat_index
        effective = originals + repeats
        for episode in effective:
            episode["adapter_task"] = task
        merged_episodes.extend(effective)
        summary.append(
            {
                "task": task,
                "source_index": str(path),
                "original_episodes": len(originals),
                "repeated_episodes": len(repeats),
                "original_frames": original_frames,
                "effective_frames": sum(int(episode["num_frames"]) for episode in effective),
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "pi05_balanced_multitask_adapter_index_v1",
        "balance_frames": bool(args.balance_frames),
        "target_frames": target_frames,
        "tasks": summary,
        "episodes": merged_episodes,
    }
    with output.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)
        output_file.write("\n")
    print(json.dumps({"output": str(output), "tasks": summary}, indent=2))


if __name__ == "__main__":
    main()
