#!/usr/bin/env python3
"""Create an isolated PI0.5 deployment with a scaled Adapter residual."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


OUTPUT_KEYS = ("up_projection.weight", "up_projection.bias")


def scaled_state_dict(source: Path, scale: float) -> dict[str, torch.Tensor]:
    if not 0.0 <= scale <= 1.0:
        raise ValueError("--scale must be in [0, 1]")
    state = load_file(source)
    missing = [key for key in OUTPUT_KEYS if key not in state]
    if missing:
        raise KeyError(f"Adapter checkpoint is missing output projection keys: {missing}")
    return {
        key: value.mul(scale) if key in OUTPUT_KEYS else value
        for key, value in state.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-deployment", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_deployment.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_adapter = source / "feature_adapter" / "adapter_model.safetensors"
    if not source_adapter.is_file():
        raise FileNotFoundError(source_adapter)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output}")

    staging = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        for item in source.iterdir():
            destination = staging / item.name
            if item.name == "feature_adapter":
                destination.mkdir()
                for adapter_item in item.iterdir():
                    if adapter_item.name != "adapter_model.safetensors":
                        shutil.copy2(adapter_item, destination / adapter_item.name)
            elif item.is_symlink():
                destination.symlink_to(item.resolve())
            elif item.is_dir():
                shutil.copytree(item, destination, symlinks=True)
            else:
                shutil.copy2(item, destination)

        scaled = scaled_state_dict(source_adapter, args.scale)
        save_file(
            scaled,
            staging / "feature_adapter" / "adapter_model.safetensors",
        )
        provenance = {
            "format": "pi05_feature_adapter_residual_scale_v1",
            "source_deployment": str(source),
            "source_adapter": str(source_adapter),
            "residual_scale": args.scale,
            "scaled_parameters": list(OUTPUT_KEYS),
            "equivalence": "z_aligned = z + scale * delta_z",
        }
        (staging / "adapter_residual_scale.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(output)


if __name__ == "__main__":
    main()
