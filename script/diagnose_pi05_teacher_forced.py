#!/usr/bin/env python3

"""Compare PI0.5 action chunks with expert chunks on exact dataset frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = Path(
    os.environ.get("ROBOTWIN_PI05_SERVER_SRC", ROOT / "third_party/lerobot_nero_runtime/src")
)
sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402


CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
ACTION = "action"
STATE = "observation.state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--episode-index", required=True, type=int)
    parser.add_argument("--frames", default="160,180,200,210,220,225")
    parser.add_argument("--inference-seeds", default="123,124,125")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def decode_frame(capture: cv2.VideoCapture, global_index: int) -> torch.Tensor:
    capture.set(cv2.CAP_PROP_POS_FRAMES, global_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not decode global video frame {global_index}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-8:
        return None
    return float(np.dot(a, b) / denominator)


def summarize_chunk(
    predicted: np.ndarray,
    expert: np.ndarray,
    state: np.ndarray,
) -> dict[str, float | None | list[float]]:
    predicted_right = predicted[:, :7]
    expert_right = expert[:, :7]
    state_right = state[:7]
    target = expert_right[-1]
    initial_target_error = float(np.linalg.norm(state_right - target))
    endpoint_target_error = float(np.linalg.norm(predicted_right[-1] - target))
    progress = None
    # A progress ratio is numerically meaningless once the expert chunk is
    # already static at contact (for example, sub-millimetric joint noise).
    if initial_target_error > 1e-2:
        progress = float(1.0 - endpoint_target_error / initial_target_error)

    predicted_steps = np.diff(np.vstack((state_right, predicted_right)), axis=0)
    expert_steps = np.diff(np.vstack((state_right, expert_right)), axis=0)
    return {
        "right_rmse_50": float(np.sqrt(np.mean((predicted_right - expert_right) ** 2))),
        "right_rmse_first10": float(np.sqrt(np.mean((predicted_right[:10] - expert_right[:10]) ** 2))),
        "all_arm_rmse_50": float(np.sqrt(np.mean((predicted[:, :14] - expert[:, :14]) ** 2))),
        "right_endpoint_target_error": endpoint_target_error,
        "right_terminal_progress": progress,
        "right_first10_direction_cosine": cosine(
            (predicted_right[9] - state_right), (expert_right[9] - state_right)
        ),
        "right_full_step_velocity_cosine": cosine(predicted_steps.ravel(), expert_steps.ravel()),
        "predicted_right_max_step_norm": float(np.linalg.norm(predicted_steps, axis=1).max()),
        "expert_right_max_step_norm": float(np.linalg.norm(expert_steps, axis=1).max()),
        "predicted_right_motion_at_10_20_30_40_50": [
            float(np.linalg.norm(predicted_right[index] - state_right))
            for index in (9, 19, 29, 39, 49)
        ],
        "expert_right_motion_at_10_20_30_40_50": [
            float(np.linalg.norm(expert_right[index] - state_right))
            for index in (9, 19, 29, 39, 49)
        ],
    }


def postprocess_chunk(postprocessor, chunk: torch.Tensor) -> np.ndarray:
    actions = [postprocessor(chunk[:, index, :]) for index in range(chunk.shape[1])]
    return torch.stack(actions, dim=1).squeeze(0).detach().float().cpu().numpy()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = parse_ints(args.frames)
    seeds = parse_ints(args.inference_seeds)
    if not frames or not seeds:
        raise ValueError("--frames and --inference-seeds must not be empty")

    episode_table = pq.read_table(dataset_root / "meta/episodes/chunk-000/file-000.parquet")
    episode_rows = episode_table.filter(pc.equal(episode_table["episode_index"], args.episode_index))
    if len(episode_rows) != 1:
        raise ValueError(f"Expected one episode {args.episode_index}, found {len(episode_rows)}")
    episode = episode_rows.to_pylist()[0]
    episode_length = int(episode["length"])
    invalid_frames = [frame for frame in frames if not 0 <= frame < episode_length]
    if invalid_frames:
        raise ValueError(f"Frames outside episode length {episode_length}: {invalid_frames}")

    data_table = pq.read_table(
        dataset_root / "data/chunk-000/file-000.parquet",
        columns=[STATE, ACTION, "episode_index", "frame_index"],
    )
    episode_data = data_table.filter(pc.equal(data_table["episode_index"], args.episode_index))
    states = np.asarray(episode_data[STATE].to_pylist(), dtype=np.float32)
    actions = np.asarray(episode_data[ACTION].to_pylist(), dtype=np.float32)
    if len(states) != episode_length:
        raise RuntimeError(f"Episode metadata says {episode_length} frames, data has {len(states)}")

    captures = {}
    for camera_key in CAMERA_KEYS:
        path = dataset_root / f"videos/{camera_key}/chunk-000/file-000.mp4"
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {path}")
        captures[camera_key] = capture

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = args.device
    policy = PI05Policy.from_pretrained(checkpoint, config=config, strict=True)
    policy.to(args.device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    records = []
    predictions: dict[tuple[int, int], np.ndarray] = {}
    snapshots = []
    with torch.inference_mode():
        for frame in frames:
            global_index = int(episode["dataset_from_index"]) + frame
            images = {
                camera_key: decode_frame(capture, global_index)
                for camera_key, capture in captures.items()
            }
            snapshots.append(images[CAMERA_KEYS[0]].permute(1, 2, 0).numpy())
            sample = {
                STATE: torch.from_numpy(states[frame].copy()),
                **images,
                "task": episode["tasks"][0],
            }
            end = min(frame + 50, episode_length)
            expert = actions[frame:end]
            if len(expert) < 50:
                expert = np.concatenate(
                    (expert, np.repeat(expert[-1][None, :], 50 - len(expert), axis=0)), axis=0
                )

            for seed in seeds:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                processed = preprocessor(sample)
                chunk = policy.predict_action_chunk(processed)
                predicted = postprocess_chunk(postprocessor, chunk)
                predictions[(frame, seed)] = predicted
                records.append(
                    {
                        "episode_index": args.episode_index,
                        "frame_index": frame,
                        "inference_seed": seed,
                        **summarize_chunk(predicted, expert, states[frame]),
                    }
                )
                print(
                    f"frame={frame} seed={seed} "
                    f"rmse={records[-1]['right_rmse_50']:.4f} "
                    f"progress={records[-1]['right_terminal_progress']}"
                )

    for capture in captures.values():
        capture.release()

    frame_summaries = []
    for frame in frames:
        rows = [record for record in records if record["frame_index"] == frame]
        progress_values = [
            row["right_terminal_progress"]
            for row in rows
            if row["right_terminal_progress"] is not None
        ]
        frame_summaries.append(
            {
                "frame_index": frame,
                "right_rmse_50_mean": float(np.mean([row["right_rmse_50"] for row in rows])),
                "right_rmse_first10_mean": float(
                    np.mean([row["right_rmse_first10"] for row in rows])
                ),
                "right_terminal_progress_mean": (
                    float(np.mean(progress_values)) if progress_values else None
                ),
                "right_first10_direction_cosine_mean": float(
                    np.mean(
                        [
                            row["right_first10_direction_cosine"]
                            for row in rows
                            if row["right_first10_direction_cosine"] is not None
                        ]
                    )
                ),
            }
        )

    report = {
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "episode_index": args.episode_index,
        "episode_length": episode_length,
        "task": episode["tasks"][0],
        "frames": frames,
        "inference_seeds": seeds,
        "frame_summaries": frame_summaries,
        "records": records,
    }
    (output_dir / "teacher_forced_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(len(frames), 2, figsize=(12, 3.2 * len(frames)), squeeze=False)
    horizon = np.arange(1, 51)
    for row_index, frame in enumerate(frames):
        end = min(frame + 50, episode_length)
        expert = actions[frame:end, :7]
        if len(expert) < 50:
            expert = np.concatenate(
                (expert, np.repeat(expert[-1][None, :], 50 - len(expert), axis=0)), axis=0
            )
        state = states[frame, :7]
        expert_motion = np.linalg.norm(expert - state, axis=1)
        axes[row_index, 0].plot(horizon, expert_motion, color="black", linewidth=2, label="expert")
        for seed in seeds:
            predicted = predictions[(frame, seed)][:, :7]
            axes[row_index, 0].plot(
                horizon, np.linalg.norm(predicted - state, axis=1), label=f"pred {seed}", alpha=0.8
            )
            axes[row_index, 1].plot(
                horizon,
                np.linalg.norm(predicted - expert, axis=1),
                label=f"pred {seed}",
                alpha=0.8,
            )
        axes[row_index, 0].set_title(f"Frame {frame}: right-arm motion from observation")
        axes[row_index, 1].set_title(f"Frame {frame}: right-arm error to expert")
        for axis in axes[row_index]:
            axis.set_xlabel("action horizon")
            axis.set_ylabel("joint L2 (rad)")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "teacher_forced_curves.png", dpi=160)
    plt.close(figure)

    snapshot_columns = min(3, len(frames))
    snapshot_rows = (len(frames) + snapshot_columns - 1) // snapshot_columns
    snapshot_figure, snapshot_axes = plt.subplots(
        snapshot_rows, snapshot_columns, figsize=(5 * snapshot_columns, 3.25 * snapshot_rows), squeeze=False
    )
    for axis, frame, snapshot in zip(snapshot_axes.flat, frames, snapshots):
        axis.imshow(snapshot)
        axis.set_title(f"Expert episode {args.episode_index}, frame {frame}")
        axis.axis("off")
    for axis in list(snapshot_axes.flat)[len(frames) :]:
        axis.axis("off")
    snapshot_figure.tight_layout()
    snapshot_figure.savefig(output_dir / "teacher_forced_frames.jpg", dpi=150)
    plt.close(snapshot_figure)


if __name__ == "__main__":
    main()
