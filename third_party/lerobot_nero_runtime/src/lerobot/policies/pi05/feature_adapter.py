#!/usr/bin/env python

"""View-conditioned feature adapter for frozen PI0.5 image tokens."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors.torch import load_file, save_file
from torch import Tensor, nn


@dataclass(frozen=True)
class ViewFeatureAdapterSpec:
    token_dim: int
    bottleneck_dim: int
    num_heads: int = 8
    num_blocks: int = 2
    ffn_expansion: int = 4


class ResidualSpatialBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_expansion: int) -> None:
        super().__init__()
        self.conv_norm = nn.LayerNorm(dim)
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.attn_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_expansion),
            nn.GELU(),
            nn.Linear(dim * ffn_expansion, dim),
        )

    @staticmethod
    def _grid_size(num_tokens: int) -> tuple[int, int] | None:
        side = math.isqrt(num_tokens)
        return (side, side) if side * side == num_tokens else None

    def forward(self, tokens: Tensor) -> Tensor:
        grid = self._grid_size(tokens.shape[1])
        if grid is not None:
            height, width = grid
            conv_input = self.conv_norm(tokens).transpose(1, 2).reshape(
                tokens.shape[0], tokens.shape[2], height, width
            )
            conv_output = self.depthwise_conv(conv_input).flatten(2).transpose(1, 2)
            tokens = tokens + conv_output

        attn_input = self.attn_norm(tokens)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_output
        return tokens + self.ffn(self.ffn_norm(tokens))


class ViewFeatureAdapter(nn.Module):
    """Align shifted-view image tokens while preserving an identity initialization."""

    def __init__(
        self,
        token_dim: int,
        bottleneck_dim: int | None = None,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_expansion: int = 4,
    ) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or min(256, token_dim // 4)
        if bottleneck_dim % num_heads != 0:
            raise ValueError(
                f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.spec = ViewFeatureAdapterSpec(
            token_dim=token_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_expansion=ffn_expansion,
        )
        self.pre_norm = nn.LayerNorm(token_dim)
        self.down_projection = nn.Linear(token_dim, bottleneck_dim)
        self.film = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
        )
        self.spatial_blocks = nn.ModuleList(
            ResidualSpatialBlock(bottleneck_dim, num_heads, ffn_expansion)
            for _ in range(num_blocks)
        )
        self.up_projection = nn.Linear(bottleneck_dim, token_dim)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward_with_delta(self, image_tokens: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = image_tokens.dtype
        parameter_dtype = self.down_projection.weight.dtype
        normalized = self.pre_norm(image_tokens.to(dtype=parameter_dtype))
        hidden = self.down_projection(normalized)
        context = hidden.mean(dim=1)
        gamma, beta = self.film(context).chunk(2, dim=-1)
        hidden = hidden * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        for block in self.spatial_blocks:
            hidden = block(hidden)
        delta = self.up_projection(hidden).to(dtype=input_dtype)
        return image_tokens + delta, delta

    def forward(self, image_tokens: Tensor) -> Tensor:
        return self.forward_with_delta(image_tokens)[0]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def save_checkpoint(self, output: str | Path) -> Path:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        tensors = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(tensors, output / "adapter_model.safetensors", metadata={"format": "pi05_view_feature_adapter"})
        with (output / "adapter_config.json").open("w", encoding="utf-8") as config_file:
            json.dump(asdict(self.spec), config_file, indent=2)
            config_file.write("\n")
        return output

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        model_path = checkpoint / "adapter_model.safetensors" if checkpoint.is_dir() else checkpoint
        self.load_state_dict(load_file(model_path), strict=True)


def weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    per_sample = values.flatten(1).mean(dim=1)
    if weights is None:
        return per_sample.mean()
    weights = weights.to(device=values.device, dtype=values.dtype).flatten()
    return (per_sample * weights).sum() / weights.sum().clamp_min(torch.finfo(values.dtype).eps)


def global_feature_cosine_loss(student_tokens: Tensor, teacher_tokens: Tensor, weights: Tensor | None) -> Tensor:
    cosine = F.cosine_similarity(student_tokens.mean(dim=1), teacher_tokens.mean(dim=1), dim=-1)
    losses = 1.0 - cosine
    if weights is None:
        return losses.mean()
    weights = weights.to(device=losses.device, dtype=losses.dtype).flatten()
    return (losses * weights).sum() / weights.sum().clamp_min(torch.finfo(losses.dtype).eps)
