#!/usr/bin/env python3

"""Train a third-person ViewFeatureAdapter while freezing the complete PI0.5 backbone."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION


CAMERA_KEYS = {
    "front": "observation.images.front",
    "left_wrist": "observation.images.left_wrist",
    "right_wrist": "observation.images.right_wrist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--cache-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shifted-views", default="c1,c2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--flow-loss-weight", type=float, default=0.05)
    parser.add_argument("--velocity-loss-weight", type=float, default=1.0)
    parser.add_argument("--global-feature-loss-weight", type=float, default=0.05)
    parser.add_argument("--canonical-identity-loss-weight", type=float, default=0.2)
    parser.add_argument("--canonical-velocity-loss-weight", type=float, default=1.0)
    parser.add_argument("--residual-loss-weight", type=float, default=0.01)
    return parser.parse_args()


class PairedMultiviewDataset(Dataset):
    def __init__(self, index_path: Path, shifted_views: list[str], frame_stride: int, seed: int):
        with index_path.open(encoding="utf-8") as index_file:
            self.episodes = json.load(index_file)["episodes"]
        self.shifted_views = shifted_views
        self.samples = [
            (episode_index, frame_index)
            for episode_index, episode in enumerate(self.episodes)
            for frame_index in range(0, episode["num_frames"], frame_stride)
        ]
        self.rng = random.Random(seed)
        self.array_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.video_cache: OrderedDict[str, cv2.VideoCapture] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    def _arrays(self, path: str) -> dict[str, np.ndarray]:
        if path not in self.array_cache:
            archive = np.load(path)
            self.array_cache[path] = {name: archive[name] for name in archive.files}
            if len(self.array_cache) > 8:
                self.array_cache.popitem(last=False)
        return self.array_cache[path]

    def _frame(self, path: Path, frame_index: int) -> torch.Tensor:
        key = str(path)
        capture = self.video_cache.get(key)
        if capture is None:
            capture = cv2.VideoCapture(key)
            if not capture.isOpened():
                raise RuntimeError(f"Could not open video {path}")
            self.video_cache[key] = capture
            if len(self.video_cache) > 16:
                _, old_capture = self.video_cache.popitem(last=False)
                old_capture.release()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {frame_index} from {path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255.0)

    @staticmethod
    def _action_chunk(arrays: dict[str, np.ndarray], frame_index: int) -> torch.Tensor:
        timestamp = arrays["timestamp"]
        target_time = timestamp[frame_index] + np.arange(50, dtype=np.float64) / 30.0
        action = np.stack(
            [np.interp(target_time, timestamp, arrays["action"][:, dim]) for dim in range(16)], axis=-1
        ).astype(np.float32)
        # This checkpoint was trained with arm joints relative to the observation
        # that produced the whole chunk; both gripper widths remain absolute.
        action[:, :14] -= arrays["state"][frame_index, :14]
        return torch.from_numpy(action)

    def __getitem__(self, index: int) -> dict:
        episode_index, frame_index = self.samples[index]
        episode = self.episodes[episode_index]
        arrays = self._arrays(episode["cache"])
        episode_dir = Path(episode["episode_dir"])
        shifted_view = self.rng.choice(self.shifted_views)
        common = {
            "observation.state": torch.from_numpy(arrays["state"][frame_index].copy()),
            CAMERA_KEYS["left_wrist"]: self._frame(episode_dir / "left_wrist.mp4", frame_index),
            CAMERA_KEYS["right_wrist"]: self._frame(episode_dir / "right_wrist.mp4", frame_index),
            ACTION: self._action_chunk(arrays, frame_index),
            "task": episode["instruction"],
        }
        canonical = dict(common)
        canonical[CAMERA_KEYS["front"]] = self._frame(episode_dir / "c0.mp4", frame_index)
        shifted = dict(common)
        shifted[CAMERA_KEYS["front"]] = self._frame(
            episode_dir / f"{shifted_view}.mp4", frame_index
        )
        return {
            "canonical": canonical,
            "shifted": shifted,
            "pair_quality": float(episode["pair_quality"]),
        }


def stack_processed(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    common_keys = set.intersection(*(set(sample) for sample in samples))
    return {
        key: torch.cat([sample[key] for sample in samples], dim=0)
        for key in common_keys
        if isinstance(samples[0][key], torch.Tensor)
    }


def preprocess_sample(preprocessor, sample: dict) -> dict[str, torch.Tensor]:
    processed = preprocessor(sample)
    if processed[ACTION].ndim == 2:
        processed[ACTION] = processed[ACTION].unsqueeze(0)
    return processed


def make_deployment_checkpoint(base: Path, adapter: Path, output: Path, policy_config) -> Path:
    deployment = output / "deployment_checkpoint"
    deployment.mkdir(parents=True, exist_ok=True)
    for source in base.iterdir():
        if source.name == "config.json":
            continue
        destination = deployment / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())
    with (base / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config.update(
        {
            "use_feature_adapter": True,
            "feature_adapter_camera_keys": [CAMERA_KEYS["front"]],
            "feature_adapter_bottleneck_dim": 256,
            "feature_adapter_num_heads": 8,
            "feature_adapter_checkpoint": "feature_adapter",
            "feature_adapter_flow_loss_weight": policy_config.feature_adapter_flow_loss_weight,
            "feature_adapter_velocity_loss_weight": policy_config.feature_adapter_velocity_loss_weight,
            "feature_adapter_global_feature_loss_weight": policy_config.feature_adapter_global_feature_loss_weight,
            "feature_adapter_canonical_identity_loss_weight": (
                policy_config.feature_adapter_canonical_identity_loss_weight
            ),
            "feature_adapter_canonical_velocity_loss_weight": (
                policy_config.feature_adapter_canonical_velocity_loss_weight
            ),
            "feature_adapter_residual_loss_weight": policy_config.feature_adapter_residual_loss_weight,
        }
    )
    with (deployment / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    destination_adapter = deployment / "feature_adapter"
    destination_adapter.mkdir(exist_ok=True)
    for source in adapter.iterdir():
        shutil.copy2(source, destination_adapter / source.name)
    return deployment


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    config = PreTrainedConfig.from_pretrained(base_checkpoint)
    config.device = args.device
    config.use_feature_adapter = True
    config.feature_adapter_camera_keys = [CAMERA_KEYS["front"]]
    config.feature_adapter_checkpoint = None
    loss_weights = {
        "feature_adapter_flow_loss_weight": args.flow_loss_weight,
        "feature_adapter_velocity_loss_weight": args.velocity_loss_weight,
        "feature_adapter_global_feature_loss_weight": args.global_feature_loss_weight,
        "feature_adapter_canonical_identity_loss_weight": args.canonical_identity_loss_weight,
        "feature_adapter_canonical_velocity_loss_weight": args.canonical_velocity_loss_weight,
        "feature_adapter_residual_loss_weight": args.residual_loss_weight,
    }
    if any(weight < 0 for weight in loss_weights.values()):
        raise ValueError(f"Adapter loss weights must be non-negative: {loss_weights}")
    for name, weight in loss_weights.items():
        setattr(config, name, weight)
    policy = PI05Policy.from_pretrained(base_checkpoint, config=config, strict=True)
    if args.adapter_checkpoint:
        policy.load_feature_adapter(args.adapter_checkpoint)
    trainable_count = policy.configure_feature_adapter_training()
    total_count = sum(parameter.numel() for parameter in policy.parameters())
    print(f"Trainable Adapter parameters: {trainable_count:,} / {total_count:,}")
    print(f"Adapter loss weights: {json.dumps(loss_weights, sort_keys=True)}")
    with (args.output / "training_config.json").open("w", encoding="utf-8") as config_file:
        json.dump(
            {
                "base_checkpoint": str(base_checkpoint),
                "cache_index": str(args.cache_index.expanduser().resolve()),
                "shifted_views": args.shifted_views,
                "seed": args.seed,
                "loss_weights": loss_weights,
            },
            config_file,
            indent=2,
        )
        config_file.write("\n")

    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(base_checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    dataset = PairedMultiviewDataset(
        args.cache_index,
        [view.strip() for view in args.shifted_views.split(",") if view.strip()],
        args.frame_stride,
        args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda items: items,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(
        policy.feature_adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    log_path = args.output / "train_metrics.jsonl"

    step = 0
    while step < args.max_steps:
        for raw_samples in loader:
            canonical = stack_processed(
                [preprocess_sample(preprocessor, sample["canonical"]) for sample in raw_samples]
            )
            shifted = stack_processed(
                [preprocess_sample(preprocessor, sample["shifted"]) for sample in raw_samples]
            )
            pair_quality = torch.tensor(
                [sample["pair_quality"] for sample in raw_samples],
                dtype=torch.float32,
                device=args.device,
            )
            noise = policy.model.sample_noise(
                policy.prepare_action(canonical).shape, torch.device(args.device)
            )
            time = policy.model.sample_time(len(raw_samples), torch.device(args.device))
            loss, metrics = policy.forward_feature_adapter(
                canonical, shifted, pair_quality=pair_quality, noise=noise, time=time
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = policy.feature_adapter_gradient_norm()
            torch.nn.utils.clip_grad_norm_(policy.feature_adapter.parameters(), args.grad_clip_norm)
            optimizer.step()
            step += 1

            record = {"step": step, "adapter_gradient_norm": float(gradient_norm.detach().cpu())}
            record.update({name: float(value.detach().cpu()) for name, value in metrics.items()})
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(record) + "\n")
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(record))
            if step % args.save_every == 0 or step == args.max_steps:
                checkpoint = policy.save_feature_adapter(args.output / f"step_{step:06d}")
                make_deployment_checkpoint(base_checkpoint, checkpoint, args.output, config)
            if step >= args.max_steps:
                break

    final_checkpoint = policy.save_feature_adapter(args.output / "feature_adapter")
    deployment = make_deployment_checkpoint(base_checkpoint, final_checkpoint, args.output, config)
    print(f"Adapter checkpoint: {final_checkpoint}")
    print(f"Inference checkpoint: {deployment}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
