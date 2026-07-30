from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = [
    "Use the left arm to pick up the hammer and strike the block.",
    "Take the hammer in the right gripper and strike the block.",
]
OLD_LABELS = [
    CANONICAL[0],
    "Grasp the hammer with the right arm, lift it, and hit the block.",
    "With the left arm, pick up the hammer and bring its head down onto the block.",
    CANONICAL[1],
]


def _write_source(root: Path) -> None:
    data_dir = root / "data/chunk-000"
    episode_dir = root / "meta/episodes/chunk-000"
    data_dir.mkdir(parents=True)
    episode_dir.mkdir(parents=True)
    pq.write_table(
        pa.table({"task_index": pa.array([0, 1, 2, 3], type=pa.int64())}),
        data_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 1, 2, 3], type=pa.int64()),
                "tasks": pa.array([[label] for label in OLD_LABELS], type=pa.list_(pa.string())),
                "stats/task_index/min": pa.array([[index] for index in range(4)], type=pa.list_(pa.int64())),
                "stats/task_index/max": pa.array([[index] for index in range(4)], type=pa.list_(pa.int64())),
                "stats/task_index/mean": pa.array([[float(index)] for index in range(4)], type=pa.list_(pa.float64())),
                "stats/task_index/std": pa.array([[0.0]] * 4, type=pa.list_(pa.float64())),
                "stats/task_index/count": pa.array([[1]] * 4, type=pa.list_(pa.int64())),
            }
        ),
        episode_dir / "file-000.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array(range(4), type=pa.int64()),
                "__index_level_0__": pa.array(OLD_LABELS, type=pa.string()),
            }
        ),
        root / "meta/tasks.parquet",
    )
    (root / "meta/stats.json").write_text(
        json.dumps({"task_index": {"count": [4]}}), encoding="utf-8"
    )
    (root / "meta/info.json").write_text(
        json.dumps({"total_tasks": 4}), encoding="utf-8"
    )


def test_normalizer_creates_two_label_copy_without_modifying_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "normalized"
    _write_source(source)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/normalize_hammer_task_labels.py"),
            "--input-dir",
            str(source),
            "--output-dir",
            str(output),
        ],
        check=True,
    )

    source_indices = pq.read_table(source / "data/chunk-000/file-000.parquet")[
        "task_index"
    ].to_pylist()
    output_indices = pq.read_table(output / "data/chunk-000/file-000.parquet")[
        "task_index"
    ].to_pylist()
    output_tasks = pq.read_table(output / "meta/tasks.parquet")
    output_episodes = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet")

    assert source_indices == [0, 1, 2, 3]
    assert output_indices == [0, 1, 0, 1]
    assert output_tasks["task_index"].to_pylist() == [0, 1]
    assert output_tasks["__index_level_0__"].to_pylist() == CANONICAL
    assert output_episodes["tasks"].to_pylist() == [[CANONICAL[index]] for index in output_indices]
    assert json.loads((output / "meta/info.json").read_text())["total_tasks"] == 2
