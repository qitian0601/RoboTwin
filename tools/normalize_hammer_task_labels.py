#!/usr/bin/env python3
"""Create a two-label Hammer dataset without modifying the source dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CANONICAL_TASKS = {
    0: "Use the left arm to pick up the hammer and strike the block.",
    1: "Take the hammer in the right gripper and strike the block.",
}
OLD_TO_CANONICAL = {0: 0, 1: 1, 2: 0, 3: 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as temporary_file:
        temporary = Path(temporary_file.name)
        json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary, path)


def atomic_write_parquet(path: Path, table: pa.Table) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as f:
        temporary = Path(f.name)
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    return table.set_column(table.schema.get_field_index(name), name, values)


def mapped_indices(values: pa.ChunkedArray) -> np.ndarray:
    source = np.asarray(values.to_pylist(), dtype=np.int64)
    unexpected = sorted(set(source) - set(OLD_TO_CANONICAL))
    if unexpected:
        raise ValueError(f"Unexpected Hammer task indices: {unexpected}")
    return np.asarray([OLD_TO_CANONICAL[int(value)] for value in source], dtype=np.int64)


def normalize_data(path: Path) -> np.ndarray:
    table = pq.read_table(path)
    mapped = mapped_indices(table["task_index"])
    table = replace_column(table, "task_index", pa.array(mapped, type=pa.int64()))
    atomic_write_parquet(path, table)
    return mapped


def normalize_tasks(path: Path) -> None:
    source = pq.read_table(path)
    table = pa.Table.from_arrays(
        [
            pa.array([0, 1], type=pa.int64()),
            pa.array([CANONICAL_TASKS[0], CANONICAL_TASKS[1]], type=pa.string()),
        ],
        schema=source.schema,
    )
    atomic_write_parquet(path, table)


def normalize_episode_stats(table: pa.Table, mapped: np.ndarray) -> pa.Table:
    task_column = table["tasks"]
    canonical_tasks = pa.array(
        [[CANONICAL_TASKS[int(index)]] for index in mapped], type=task_column.type
    )
    table = replace_column(table, "tasks", canonical_tasks)
    for name in table.column_names:
        if not name.startswith("stats/task_index/") or name.endswith("/count"):
            continue
        field = table.schema.field(name)
        suffix = name.rsplit("/", 1)[1]
        if suffix in {"min", "max"}:
            values = [[int(index)] for index in mapped]
        elif suffix == "std":
            values = [[0.0] for _ in mapped]
        else:
            values = [[float(index)] for index in mapped]
        table = replace_column(table, name, pa.array(values, type=field.type))
    return table


def normalize_episodes(path: Path) -> None:
    table = pq.read_table(path)
    task_lists = table["tasks"].to_pylist()
    source_tasks = [tasks[0] if len(tasks) == 1 else None for tasks in task_lists]
    task_lookup = {value: key for key, value in CANONICAL_TASKS.items()}
    old_tasks = pq.read_table(path.parent.parent.parent / "tasks.parquet")
    old_labels = old_tasks["__index_level_0__"].to_pylist()
    old_index_by_label = {label: index for index, label in enumerate(old_labels)}
    old_indices = []
    for task in source_tasks:
        if task not in old_index_by_label:
            raise ValueError(f"Unexpected episode task label: {task!r}")
        old_indices.append(old_index_by_label[task])
    mapped = np.asarray([OLD_TO_CANONICAL[index] for index in old_indices], dtype=np.int64)
    if any(task not in task_lookup for task in CANONICAL_TASKS.values()):
        raise AssertionError("Canonical task lookup is incomplete")
    atomic_write_parquet(path, normalize_episode_stats(table, mapped))


def normalize_stats(path: Path, mapped: np.ndarray) -> None:
    with path.open(encoding="utf-8") as stats_file:
        stats = json.load(stats_file)
    values = mapped.astype(np.float64)
    stats["task_index"] = {
        "min": [int(values.min())],
        "max": [int(values.max())],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
        **{f"q{percent:02d}": [float(np.quantile(values, percent / 100))] for percent in (1, 10, 50, 90, 99)},
    }
    atomic_write_json(path, stats)


def validate(output_dir: Path) -> None:
    task_table = pq.read_table(output_dir / "meta" / "tasks.parquet")
    if task_table["task_index"].to_pylist() != [0, 1]:
        raise ValueError("Normalized tasks.parquet must contain task indices 0 and 1")
    if task_table["__index_level_0__"].to_pylist() != list(CANONICAL_TASKS.values()):
        raise ValueError("Normalized tasks.parquet has unexpected labels")
    data = pq.read_table(output_dir / "data" / "chunk-000" / "file-000.parquet", columns=["task_index"])
    if set(data["task_index"].to_pylist()) != {0, 1}:
        raise ValueError("Normalized frame task indices must be exactly {0, 1}")
    episodes = pq.read_table(output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet", columns=["tasks"])
    if {row[0] for row in episodes["tasks"].to_pylist()} != set(CANONICAL_TASKS.values()):
        raise ValueError("Normalized episode labels are inconsistent")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not (input_dir / "meta" / "tasks.parquet").is_file():
        raise FileNotFoundError(f"Not a LeRobot v3 dataset: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    shutil.copytree(input_dir, output_dir, copy_function=os.link)
    frame_task_indices = normalize_data(output_dir / "data" / "chunk-000" / "file-000.parquet")
    normalize_episodes(output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    normalize_tasks(output_dir / "meta" / "tasks.parquet")
    normalize_stats(output_dir / "meta" / "stats.json", frame_task_indices)
    info_path = output_dir / "meta" / "info.json"
    with info_path.open(encoding="utf-8") as info_file:
        info = json.load(info_file)
    info["total_tasks"] = len(CANONICAL_TASKS)
    atomic_write_json(info_path, info)
    atomic_write_json(
        output_dir / "task_normalization.json",
        {
            "source": str(input_dir),
            "canonical_tasks": CANONICAL_TASKS,
            "old_to_canonical": OLD_TO_CANONICAL,
            "video_files_hardlinked": True,
        },
    )
    validate(output_dir)
    print(f"Normalized Hammer labels: {output_dir}")


if __name__ == "__main__":
    main()
