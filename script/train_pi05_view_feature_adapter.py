#!/usr/bin/env python3

"""Train a third-person ViewFeatureAdapter while freezing the complete PI0.5 backbone."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import OrderedDict
from datetime import datetime
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the newest complete step_*/train_state.pt inside --output.",
    )
    parser.add_argument(
        "--adapter-variant",
        choices=(
            "legacy",
            "pure_teacher_gated_residual",
            "multi_scale_dynamic",
            "image_routed_moe",
        ),
        default="legacy",
    )
    parser.add_argument("--adapter-num-experts", type=int, default=4)
    parser.add_argument(
        "--pose-enabled",
        action="store_true",
        help="Enable the optional 9D pose conditioner. Leave off for image-only training/inference.",
    )
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


def newest_resume_checkpoint(output: Path) -> tuple[int, Path, dict] | None:
    candidates: list[tuple[int, Path]] = []
    for path in output.glob("step_*"):
        if not path.is_dir() or not (path / "train_state.pt").is_file():
            continue
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    for step, checkpoint in sorted(candidates, reverse=True):
        required_adapter_files = (
            checkpoint / "adapter_model.safetensors",
            checkpoint / "adapter_config.json",
        )
        if not all(path.is_file() for path in required_adapter_files):
            print(f"Ignoring incomplete Adapter checkpoint: {checkpoint}")
            continue
        try:
            state = torch.load(
                checkpoint / "train_state.pt", map_location="cpu", weights_only=False
            )
        except Exception as error:
            print(
                f"Ignoring unreadable Adapter resume state {checkpoint}: "
                f"{type(error).__name__}: {error}"
            )
            continue
        if int(state.get("step", -1)) != step:
            print(f"Ignoring step-mismatched Adapter checkpoint: {checkpoint}")
            continue
        return step, checkpoint, state
    return None


def training_signature(args: argparse.Namespace, base_checkpoint: Path) -> dict:
    return {
        "base_checkpoint": str(base_checkpoint),
        "cache_index": str(args.cache_index.expanduser().resolve()),
        "shifted_views": args.shifted_views,
        "seed": args.seed,
        "adapter_variant": args.adapter_variant,
        "adapter_num_experts": args.adapter_num_experts,
        "pose_enabled": args.pose_enabled,
        "pose_dim": 9,
        "frame_stride": args.frame_stride,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss_weights": {
            "feature_adapter_flow_loss_weight": args.flow_loss_weight,
            "feature_adapter_velocity_loss_weight": args.velocity_loss_weight,
            "feature_adapter_global_feature_loss_weight": args.global_feature_loss_weight,
            "feature_adapter_canonical_identity_loss_weight": args.canonical_identity_loss_weight,
            "feature_adapter_canonical_velocity_loss_weight": args.canonical_velocity_loss_weight,
            "feature_adapter_residual_loss_weight": args.residual_loss_weight,
        },
    }


def save_training_checkpoint(
    policy: PI05Policy,
    optimizer: torch.optim.Optimizer,
    dataset: PairedMultiviewDataset,
    loader_generator: torch.Generator,
    output: Path,
    step: int,
    signature: dict,
) -> Path:
    checkpoint = output / f"step_{step:06d}"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {checkpoint}")
    staging = output / f".step_{step:06d}.{os.getpid()}.tmp"
    if staging.exists():
        raise FileExistsError(f"Refusing to reuse checkpoint staging directory: {staging}")
    policy.save_feature_adapter(staging)
    state = {
        "format": "pi05_feature_adapter_train_state_v2",
        "step": step,
        "signature": signature,
        "optimizer": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "dataset_rng_state": dataset.rng.getstate(),
        "loader_generator_state": loader_generator.get_state(),
    }
    state_staging = staging / ".train_state.pt.tmp"
    torch.save(state, state_staging)
    os.replace(state_staging, staging / "train_state.pt")
    staging.rename(checkpoint)
    return checkpoint


def make_deployment_checkpoint(base: Path, adapter: Path, output: Path, policy_config) -> Path:
    deployment = output / "deployment_checkpoint"
    if deployment.exists() or deployment.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing deployment: {deployment}")
    staging = output / f".deployment_checkpoint.{os.getpid()}.tmp"
    if staging.exists():
        raise FileExistsError(f"Refusing to reuse deployment staging directory: {staging}")
    staging.mkdir(parents=True)
    for source in base.iterdir():
        if source.name == "config.json":
            continue
        destination = staging / source.name
        destination.symlink_to(source.resolve())
    with (base / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config.update(
        {
            "use_feature_adapter": True,
            "feature_adapter_camera_keys": [CAMERA_KEYS["front"]],
            "feature_adapter_variant": policy_config.feature_adapter_variant,
            "feature_adapter_bottleneck_dim": policy_config.feature_adapter_bottleneck_dim,
            "feature_adapter_num_heads": policy_config.feature_adapter_num_heads,
            "feature_adapter_num_blocks": policy_config.feature_adapter_num_blocks,
            "feature_adapter_ffn_expansion": policy_config.feature_adapter_ffn_expansion,
            "feature_adapter_num_experts": policy_config.feature_adapter_num_experts,
            "feature_adapter_pose_enabled": policy_config.feature_adapter_pose_enabled,
            "feature_adapter_pose_dim": policy_config.feature_adapter_pose_dim,
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
    with (staging / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")
    destination_adapter = staging / "feature_adapter"
    destination_adapter.mkdir()
    for source in adapter.iterdir():
        if source.name in {"adapter_model.safetensors", "adapter_config.json"}:
            shutil.copy2(source, destination_adapter / source.name)
    staging.rename(deployment)
    return deployment


def main() -> None:
    args = parse_args()
    if args.resume and args.adapter_checkpoint is not None:
        raise ValueError("--resume and --adapter-checkpoint are mutually exclusive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base_checkpoint = args.base_checkpoint.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    signature = training_signature(args, base_checkpoint)
    resume_info = None
    if args.resume:
        if not args.output.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {args.output}")
        resume_info = newest_resume_checkpoint(args.output)
        if resume_info is None:
            raise FileNotFoundError(f"No complete step_*/train_state.pt found in {args.output}")
        resume_step, resume_checkpoint, resume_state = resume_info
        if resume_state.get("signature") != signature:
            raise RuntimeError(
                "Resume configuration differs from the saved training signature:\n"
                f"saved={json.dumps(resume_state.get('signature'), sort_keys=True)}\n"
                f"current={json.dumps(signature, sort_keys=True)}"
            )
        args.adapter_checkpoint = resume_checkpoint
    else:
        if args.output.exists():
            raise FileExistsError(f"Refusing to reuse or overwrite output directory: {args.output}")
        args.output.mkdir(parents=True)

    config = PreTrainedConfig.from_pretrained(base_checkpoint)
    config.device = args.device
    config.use_feature_adapter = True
    config.feature_adapter_camera_keys = [CAMERA_KEYS["front"]]
    config.feature_adapter_variant = args.adapter_variant
    config.feature_adapter_num_experts = args.adapter_num_experts
    config.feature_adapter_pose_enabled = args.pose_enabled
    config.feature_adapter_pose_dim = 9
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
    training_config_path = args.output / "training_config.json"
    if not args.resume:
        training_record = dict(signature)
        training_record.update(
            {
                "max_steps": args.max_steps,
                "save_every": args.save_every,
                "log_every": args.log_every,
                "grad_clip_norm": args.grad_clip_norm,
            }
        )
        with training_config_path.open("x", encoding="utf-8") as config_file:
            json.dump(training_record, config_file, indent=2)
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
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda items: items,
        generator=loader_generator,
    )
    optimizer = torch.optim.AdamW(
        policy.feature_adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    step = 0
    if resume_info is not None:
        step, resume_checkpoint, resume_state = resume_info
        optimizer.load_state_dict(resume_state["optimizer"])
        random.setstate(resume_state["python_random_state"])
        np.random.set_state(resume_state["numpy_random_state"])
        torch.set_rng_state(resume_state["torch_rng_state"])
        cuda_state = resume_state.get("cuda_rng_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)
        dataset.rng.setstate(resume_state["dataset_rng_state"])
        loader_generator.set_state(resume_state["loader_generator_state"])
        if step > args.max_steps:
            raise ValueError(
                f"Resume checkpoint step {step} exceeds requested max steps {args.max_steps}"
            )
        print(f"Resumed Adapter training from step {step}: {resume_checkpoint}")
        segment = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = args.output / f"train_metrics_resume_from_{step:06d}_{segment}.jsonl"
        with (args.output / "resume_events.jsonl").open("a", encoding="utf-8") as event_file:
            event_file.write(
                json.dumps(
                    {
                        "resumed_at": datetime.now().astimezone().isoformat(),
                        "checkpoint": str(resume_checkpoint),
                        "step": step,
                        "metrics_segment": str(log_path),
                    }
                )
                + "\n"
            )
    else:
        log_path = args.output / "train_metrics.jsonl"

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
                save_training_checkpoint(
                    policy,
                    optimizer,
                    dataset,
                    loader_generator,
                    args.output,
                    step,
                    signature,
                )
            if step >= args.max_steps:
                break

    final_checkpoint = args.output / f"step_{step:06d}"
    if not (final_checkpoint / "adapter_model.safetensors").is_file():
        raise RuntimeError(f"Final Adapter checkpoint is incomplete: {final_checkpoint}")
    deployment = make_deployment_checkpoint(base_checkpoint, final_checkpoint, args.output, config)
    final_link = args.output / "feature_adapter"
    if final_link.exists() or final_link.is_symlink():
        raise FileExistsError(f"Refusing to overwrite final Adapter link: {final_link}")
    final_link.symlink_to(final_checkpoint.name, target_is_directory=True)
    print(f"Adapter checkpoint: {final_checkpoint}")
    print(f"Inference checkpoint: {deployment}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
