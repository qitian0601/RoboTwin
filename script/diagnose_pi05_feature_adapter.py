#!/usr/bin/env python3

"""Measure held-out paired-view PI0.5 feature alignment without control rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

from train_pi05_view_feature_adapter import PairedMultiviewDataset, preprocess_sample


FRONT_KEY = "observation.images.front"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--views", default="c1,c2,c3,c4,c5,c6")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--flow-timestep", type=float, default=0.5)
    parser.add_argument(
        "--action-chunk-steps",
        type=int,
        default=10,
        help="Fixed-noise flow integration steps for the full action-chunk diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def scalar_metrics(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def global_cosine(tokens_a: torch.Tensor, tokens_b: torch.Tensor) -> float:
    return float(
        F.cosine_similarity(tokens_a.mean(dim=1), tokens_b.mean(dim=1), dim=-1)
        .mean()
        .detach()
        .cpu()
    )


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a, b).detach().cpu())


def rms(tokens: torch.Tensor) -> float:
    return float(tokens.float().square().mean().sqrt().detach().cpu())


def select_indices(length: int, max_samples: int) -> np.ndarray:
    if max_samples <= 0 or max_samples >= length:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, max_samples, dtype=np.int64))


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    views = [view.strip() for view in args.views.split(",") if view.strip()]
    if not views:
        raise ValueError("--views must contain at least one shifted view")

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = args.device
    if not config.use_feature_adapter:
        raise ValueError("--checkpoint must be an Adapter deployment checkpoint")
    policy = PI05Policy.from_pretrained(checkpoint, config=config, strict=True)
    policy.eval()
    if policy.feature_adapter is None:
        raise RuntimeError("Adapter checkpoint did not create a Feature Adapter")

    preprocessor, _ = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    datasets = {
        view: PairedMultiviewDataset(args.cache_index, [view], args.frame_stride, args.seed)
        for view in views
    }
    reference = datasets[views[0]]
    if any(len(dataset) != len(reference) for dataset in datasets.values()):
        raise RuntimeError("All view datasets must contain the same paired sample indices")
    indices = select_indices(len(reference), args.max_samples)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    per_view: dict[str, dict[str, list[float]]] = {
        view: {
            "feature_cosine_raw": [],
            "feature_cosine_adapted": [],
            "velocity_mse_raw": [],
            "velocity_mse_adapted": [],
            "flow_mse_raw": [],
            "flow_mse_adapted": [],
            "action_chunk_mse_raw": [],
            "action_chunk_mse_adapted": [],
            "delta_z_rms": [],
        }
        for view in views
    }
    c0_metrics = {
        "feature_cosine": [],
        "velocity_mse": [],
        "action_chunk_mse": [],
        "delta_z_rms": [],
    }
    records = []

    with torch.inference_mode():
        for position, sample_index in enumerate(indices, start=1):
            canonical_raw = reference[int(sample_index)]["canonical"]
            canonical = preprocess_sample(preprocessor, canonical_raw)
            canonical_images, canonical_masks, image_keys = policy._preprocess_images(canonical)
            tokens = canonical["observation.language.tokens"]
            masks = canonical["observation.language.attention_mask"]
            actions = policy.prepare_action(canonical)
            noise = torch.randn(tuple(actions.shape), generator=rng, dtype=torch.float32).to(actions.device)
            time = torch.full(
                (actions.shape[0],), args.flow_timestep, dtype=torch.float32, device=actions.device
            )
            teacher_velocity, flow_target, teacher_features = policy.model.predict_velocity(
                canonical_images,
                canonical_masks,
                tokens,
                masks,
                actions,
                noise,
                time,
                image_keys=image_keys,
                apply_feature_adapter=False,
                return_image_features=True,
            )
            teacher_action_chunk = policy.model.sample_actions(
                canonical_images,
                canonical_masks,
                tokens,
                masks,
                noise=noise,
                num_steps=args.action_chunk_steps,
                image_keys=image_keys,
                apply_feature_adapter=False,
            )
            c0_velocity, _, c0_features = policy.model.predict_velocity(
                canonical_images,
                canonical_masks,
                tokens,
                masks,
                actions,
                noise,
                time,
                image_keys=image_keys,
                apply_feature_adapter=True,
                return_image_features=True,
            )
            c0_action_chunk = policy.model.sample_actions(
                canonical_images,
                canonical_masks,
                tokens,
                masks,
                noise=noise,
                num_steps=args.action_chunk_steps,
                image_keys=image_keys,
                apply_feature_adapter=True,
            )
            teacher_tokens = teacher_features[FRONT_KEY]["raw"]
            c0_aligned = c0_features[FRONT_KEY]["aligned"]
            c0_delta = c0_features[FRONT_KEY]["delta"]
            c0_record = {
                "feature_cosine": global_cosine(c0_aligned, teacher_tokens),
                "velocity_mse": mse(c0_velocity, teacher_velocity),
                "action_chunk_mse": mse(c0_action_chunk, teacher_action_chunk),
                "delta_z_rms": rms(c0_delta),
            }
            for name, value in c0_record.items():
                c0_metrics[name].append(value)

            episode_index, frame_index = reference.samples[int(sample_index)]
            for view, dataset in datasets.items():
                shifted_raw = dataset[int(sample_index)]["shifted"]
                shifted = preprocess_sample(preprocessor, shifted_raw)
                shifted_images, shifted_masks, shifted_keys = policy._preprocess_images(shifted)
                raw_velocity, _, raw_features = policy.model.predict_velocity(
                    shifted_images,
                    shifted_masks,
                    tokens,
                    masks,
                    actions,
                    noise,
                    time,
                    image_keys=shifted_keys,
                    apply_feature_adapter=False,
                    return_image_features=True,
                )
                raw_action_chunk = policy.model.sample_actions(
                    shifted_images,
                    shifted_masks,
                    tokens,
                    masks,
                    noise=noise,
                    num_steps=args.action_chunk_steps,
                    image_keys=shifted_keys,
                    apply_feature_adapter=False,
                )
                adapted_velocity, adapted_flow_target, adapted_features = policy.model.predict_velocity(
                    shifted_images,
                    shifted_masks,
                    tokens,
                    masks,
                    actions,
                    noise,
                    time,
                    image_keys=shifted_keys,
                    apply_feature_adapter=True,
                    return_image_features=True,
                )
                adapted_action_chunk = policy.model.sample_actions(
                    shifted_images,
                    shifted_masks,
                    tokens,
                    masks,
                    noise=noise,
                    num_steps=args.action_chunk_steps,
                    image_keys=shifted_keys,
                    apply_feature_adapter=True,
                )
                if not torch.equal(flow_target, adapted_flow_target):
                    raise RuntimeError("Canonical and shifted flow targets differ")
                raw_tokens = raw_features[FRONT_KEY]["raw"]
                adapted_tokens = adapted_features[FRONT_KEY]["aligned"]
                delta_tokens = adapted_features[FRONT_KEY]["delta"]
                record = {
                    "feature_cosine_raw": global_cosine(raw_tokens, teacher_tokens),
                    "feature_cosine_adapted": global_cosine(adapted_tokens, teacher_tokens),
                    "velocity_mse_raw": mse(raw_velocity, teacher_velocity),
                    "velocity_mse_adapted": mse(adapted_velocity, teacher_velocity),
                    "flow_mse_raw": mse(raw_velocity, flow_target),
                    "flow_mse_adapted": mse(adapted_velocity, flow_target),
                    "action_chunk_mse_raw": mse(raw_action_chunk, teacher_action_chunk),
                    "action_chunk_mse_adapted": mse(adapted_action_chunk, teacher_action_chunk),
                    "delta_z_rms": rms(delta_tokens),
                }
                for name, value in record.items():
                    per_view[view][name].append(value)
                records.append(
                    {
                        "sample_index": int(sample_index),
                        "episode_index": int(episode_index),
                        "frame_index": int(frame_index),
                        "view": view,
                        **c0_record,
                        **record,
                    }
                )
            print(f"[{position}/{len(indices)}] episode={episode_index} frame={frame_index}")

    summary = {
        "checkpoint": str(checkpoint),
        "cache_index": str(args.cache_index.resolve()),
        "views": views,
        "frame_stride": args.frame_stride,
        "samples_per_view": len(indices),
        "flow_timestep": args.flow_timestep,
        "action_chunk_steps": args.action_chunk_steps,
        "c0_identity": {name: scalar_metrics(values) for name, values in c0_metrics.items()},
        "per_view": {},
    }
    for view, metrics in per_view.items():
        view_summary = {name: scalar_metrics(values) for name, values in metrics.items()}
        view_summary["feature_cosine_improvement"] = scalar_metrics(
            [adapted - raw for raw, adapted in zip(metrics["feature_cosine_raw"], metrics["feature_cosine_adapted"])]
        )
        view_summary["velocity_mse_improvement"] = scalar_metrics(
            [raw - adapted for raw, adapted in zip(metrics["velocity_mse_raw"], metrics["velocity_mse_adapted"])]
        )
        view_summary["flow_mse_improvement"] = scalar_metrics(
            [raw - adapted for raw, adapted in zip(metrics["flow_mse_raw"], metrics["flow_mse_adapted"])]
        )
        view_summary["action_chunk_mse_improvement"] = scalar_metrics(
            [
                raw - adapted
                for raw, adapted in zip(
                    metrics["action_chunk_mse_raw"], metrics["action_chunk_mse_adapted"]
                )
            ]
        )
        summary["per_view"][view] = view_summary

    with (args.output / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    with (args.output / "per_sample.jsonl").open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
