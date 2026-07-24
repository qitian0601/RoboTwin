#!/usr/bin/env python3

"""Export small HDF5 arrays needed by PI0.5 Adapter training to NumPy caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", action="append", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episodes = []
    for dataset_root in args.dataset_root:
        split_root = dataset_root.expanduser().resolve() / args.split
        for episode_dir in sorted(split_root.glob("episode_*")):
            if not (episode_dir / "_SUCCESS").exists():
                continue
            with (episode_dir / "episode.json").open(encoding="utf-8") as episode_file:
                metadata = json.load(episode_file)
            with h5py.File(episode_dir / "data.hdf5", "r") as source:
                state = source["observation/state"][:].astype(np.float32)
                action = source["action/commanded"][:].astype(np.float32)
                timestamp = source["timestamp"][:].astype(np.float64)

            relative_dir = Path(dataset_root.name) / args.split
            cache_dir = args.output / relative_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"{episode_dir.name}.npz"
            np.savez_compressed(cache_path, state=state, action=action, timestamp=timestamp)
            episodes.append(
                {
                    "cache": str(cache_path.resolve()),
                    "episode_dir": str(episode_dir.resolve()),
                    "instruction": metadata["instruction"],
                    "pair_quality": float(metadata.get("pair_quality", 1.0)),
                    "num_frames": int(state.shape[0]),
                    "sample_hz": float(metadata.get("sample_hz", 10.0)),
                    "views": metadata["views"],
                }
            )

    index = {
        "format": "pi05_adapter_cache_v1",
        "split": args.split,
        "episodes": episodes,
    }
    with (args.output / f"{args.split}_index.json").open("w", encoding="utf-8") as index_file:
        json.dump(index, index_file, indent=2)
        index_file.write("\n")
    print(f"Exported {len(episodes)} successful episodes to {args.output}")


if __name__ == "__main__":
    main()
