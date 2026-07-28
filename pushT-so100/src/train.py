from pathlib import Path
import argparse
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import math

_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_REPO_ROOT / ".cache" / "huggingface" / "datasets"))

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACTPolicy on LeRobot dataset")
    parser.add_argument("--data-path", type=str, default="data/nero-dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/nero_act")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=8000)
    parser.add_argument("--warmup-steps", type=int, default=800)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--dim-model", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--vision-backbone", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image-keys", nargs="*",
        default=["observation.images.cam_top", "observation.images.cam_side"])
    parser.add_argument("--mask-keys", nargs="*", default=[])
    return parser.parse_args()


def build_lr_lambda(warmup_steps, training_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


def main():
    args = parse_args()

    output_directory = Path(args.output_dir)
    data_dir = Path(args.data_path)
    checkpoints_dir = output_directory / f"checkpoints_{time.strftime('%Y-%m-%d_%H:%M')}"
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(output_directory / f"runs_{time.strftime('%Y-%m-%d_%H:%M')}"))
    device = torch.device(args.device)

    dataset_metadata = LeRobotDatasetMetadata(data_dir.absolute())
    features = dataset_to_policy_features(dataset_metadata.features)

    for key in args.image_keys:
        if key in features:
            features[key].shape = (3, 224, 224)

    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features  = {k: ft for k, ft in features.items()
                       if k not in output_features and k not in args.mask_keys}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        dim_model=args.dim_model,
        latent_dim=args.latent_dim,
        vision_backbone=args.vision_backbone,
        use_vae=True,
    )

    # 自动处理 delta_timestamps + action_is_pad
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    dataset = LeRobotDataset(
        repo_id=data_dir.name,
        root=str(data_dir),
        delta_timestamps=delta_timestamps,
    )

    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        cfg, dataset_stats=dataset_metadata.stats
    )

    backbone_params = list(policy.model.backbone.parameters())
    other_params = [p for p in policy.parameters()
                    if not any(p is bp for bp in backbone_params)]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": other_params,    "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, build_lr_lambda(args.warmup_steps, args.training_steps)
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    step = 0
    done = False
    print(f"Training started. Saving to {output_directory}", flush=True)

    while not done:
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            batch = preprocessor(batch)
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("Loss/train", loss.item(), step)
                writer.add_scalar("LR/train", current_lr, step)
                l_kld = loss_dict.get("kld_loss", 0) if isinstance(loss_dict, dict) else 0
                print(f"step: {step}  loss: {loss.item():.4f}  "
                      f"l1: {loss_dict.get('l1_loss', loss.item()):.4f}  "
                      f"kld: {l_kld:.4f}  lr: {current_lr:.2e}", flush=True)

            if step > 0 and step % args.save_freq == 0:
                step_ckpt_dir = checkpoints_dir / f"step_{step}"
                step_ckpt_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(step_ckpt_dir)
                preprocessor.save_pretrained(step_ckpt_dir)
                postprocessor.save_pretrained(step_ckpt_dir)
                print(f"Checkpoint saved at step {step}", flush=True)

            step += 1
            if step >= args.training_steps:
                done = True
                break

    final_dir = checkpoints_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    preprocessor.save_pretrained(final_dir)
    postprocessor.save_pretrained(final_dir)
    writer.close()
    print(f"Training finished. Final model saved to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
