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


class _CheckpointableAdapter(nn.Module):
    """Common checkpoint and diagnostic helpers for all isolated adapters."""

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def save_checkpoint(self, output: str | Path) -> Path:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        tensors = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(tensors, output / "adapter_model.safetensors", metadata={"format": "pi05_view_feature_adapter"})
        spec = getattr(self, "spec", None)
        if spec is None:
            raise RuntimeError("Adapter does not expose a serializable spec")
        with (output / "adapter_config.json").open("w", encoding="utf-8") as config_file:
            json.dump(asdict(spec), config_file, indent=2)
            config_file.write("\n")
        return output

    def load_checkpoint(self, checkpoint: str | Path) -> None:
        checkpoint = Path(checkpoint)
        model_path = checkpoint / "adapter_model.safetensors" if checkpoint.is_dir() else checkpoint
        self.load_state_dict(load_file(model_path), strict=True)


class ViewFeatureAdapter(_CheckpointableAdapter):
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

    def forward_with_delta(
        self,
        image_tokens: Tensor,
        pose_condition: Tensor | None = None,
        pose_valid_mask: Tensor | None = None,
        pose_confidence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        # Legacy Adapter intentionally ignores optional pose fields.  This
        # keeps old checkpoints and image-only inference numerically stable.
        del pose_condition, pose_valid_mask, pose_confidence
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

    # save_checkpoint/load_checkpoint/trainable_parameter_count are inherited
    # from _CheckpointableAdapter.  Keeping this class unchanged preserves the
    # existing teacher-first Adapter checkpoint format.


@dataclass(frozen=True)
class VariantFeatureAdapterSpec:
    token_dim: int
    bottleneck_dim: int
    num_heads: int = 8
    num_blocks: int = 2
    ffn_expansion: int = 4
    variant: str = "pure_teacher_gated_residual"
    pose_enabled: bool = False
    pose_dim: int = 9
    num_experts: int = 4


class OptionalPoseConditioner(nn.Module):
    """Optional zero-effect pose input for future image+pose adapters.

    ``pose_condition`` is translation (3) plus continuous-6D rotation (6).
    The valid mask and confidence gate the condition.  The projection is zero
    initialized, so even a valid pose is numerically ignored until explicitly
    trained and enabled.  Missing/invalid pose therefore exactly falls back to
    the image-only path.
    """

    def __init__(self, hidden_dim: int, pose_dim: int = 9, enabled: bool = False) -> None:
        super().__init__()
        self.pose_dim = pose_dim
        self.enabled = enabled
        self.projection = nn.Linear(pose_dim + 1, hidden_dim)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    @staticmethod
    def _batch_vector(value: Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
        if value is None:
            return None
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 1:
            value = value[:, None]
        value = value.reshape(value.shape[0], -1)
        if value.shape[0] != batch_size:
            raise ValueError(f"Pose input batch {value.shape[0]} does not match token batch {batch_size}")
        return value

    def forward(
        self,
        context: Tensor,
        pose_condition: Tensor | None = None,
        pose_valid_mask: Tensor | None = None,
        pose_confidence: Tensor | None = None,
    ) -> Tensor:
        if not self.enabled or pose_condition is None:
            return torch.zeros_like(context)
        pose = self._batch_vector(pose_condition, context.shape[0], context.device, context.dtype)
        assert pose is not None
        if pose.shape[1] != self.pose_dim:
            raise ValueError(
                f"pose_condition must contain {self.pose_dim} values (translation+6D rotation), got {pose.shape[1]}"
            )
        valid = self._batch_vector(pose_valid_mask, context.shape[0], context.device, context.dtype)
        confidence = self._batch_vector(pose_confidence, context.shape[0], context.device, context.dtype)
        if valid is None:
            valid = torch.ones((context.shape[0], 1), device=context.device, dtype=context.dtype)
        if confidence is None:
            confidence = torch.ones((context.shape[0], 1), device=context.device, dtype=context.dtype)
        valid = valid[:, :1].clamp(0, 1)
        confidence = confidence[:, :1].clamp(0, 1)
        pose_input = torch.cat([pose, confidence], dim=1)
        return self.projection(pose_input) * (valid * confidence)


def _prepare_adapter_hidden(
    image_tokens: Tensor,
    pre_norm: nn.LayerNorm,
    down_projection: nn.Linear,
) -> tuple[Tensor, torch.dtype]:
    input_dtype = image_tokens.dtype
    parameter_dtype = down_projection.weight.dtype
    normalized = pre_norm(image_tokens.to(dtype=parameter_dtype))
    return down_projection(normalized), input_dtype


class PureTeacherGatedResidualAdapter(_CheckpointableAdapter):
    """Conservative image-only residual adapter gated by global image context."""

    def __init__(
        self,
        token_dim: int,
        bottleneck_dim: int | None = None,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_expansion: int = 4,
        pose_enabled: bool = False,
        pose_dim: int = 9,
    ) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or min(256, token_dim // 4)
        if bottleneck_dim % num_heads != 0:
            raise ValueError(f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})")
        self.spec = VariantFeatureAdapterSpec(
            token_dim=token_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_expansion=ffn_expansion,
            variant="pure_teacher_gated_residual",
            pose_enabled=pose_enabled,
            pose_dim=pose_dim,
        )
        self.pre_norm = nn.LayerNorm(token_dim)
        self.down_projection = nn.Linear(token_dim, bottleneck_dim)
        self.pose_conditioner = OptionalPoseConditioner(bottleneck_dim, pose_dim, pose_enabled)
        self.gate = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.blocks = nn.ModuleList(
            ResidualSpatialBlock(bottleneck_dim, num_heads, ffn_expansion) for _ in range(num_blocks)
        )
        self.up_projection = nn.Linear(bottleneck_dim, token_dim)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward_with_delta(
        self,
        image_tokens: Tensor,
        pose_condition: Tensor | None = None,
        pose_valid_mask: Tensor | None = None,
        pose_confidence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden, input_dtype = _prepare_adapter_hidden(image_tokens, self.pre_norm, self.down_projection)
        context = hidden.mean(dim=1)
        context = context + self.pose_conditioner(context, pose_condition, pose_valid_mask, pose_confidence)
        gate = torch.tanh(self.gate(context))
        hidden = hidden * (1.0 + gate[:, None, :])
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.up_projection(hidden).to(dtype=input_dtype)
        return image_tokens + delta, delta

    def forward(self, image_tokens: Tensor, **pose_kwargs: Tensor | None) -> Tensor:
        return self.forward_with_delta(image_tokens, **pose_kwargs)[0]


class MultiScaleDynamicAdapter(_CheckpointableAdapter):
    """Dynamic mixture of 3x3/5x5 spatial filters with token fallback."""

    def __init__(
        self,
        token_dim: int,
        bottleneck_dim: int | None = None,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_expansion: int = 4,
        pose_enabled: bool = False,
        pose_dim: int = 9,
    ) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or min(256, token_dim // 4)
        if bottleneck_dim % num_heads != 0:
            raise ValueError(f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})")
        self.spec = VariantFeatureAdapterSpec(
            token_dim=token_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_expansion=ffn_expansion,
            variant="multi_scale_dynamic",
            pose_enabled=pose_enabled,
            pose_dim=pose_dim,
        )
        self.pre_norm = nn.LayerNorm(token_dim)
        self.down_projection = nn.Linear(token_dim, bottleneck_dim)
        self.pose_conditioner = OptionalPoseConditioner(bottleneck_dim, pose_dim, pose_enabled)
        self.scale_router = nn.Linear(bottleneck_dim, 2)
        nn.init.zeros_(self.scale_router.weight)
        nn.init.zeros_(self.scale_router.bias)
        self.conv3 = nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1, groups=bottleneck_dim)
        self.conv5 = nn.Conv2d(bottleneck_dim, bottleneck_dim, 5, padding=2, groups=bottleneck_dim)
        self.blocks = nn.ModuleList(
            ResidualSpatialBlock(bottleneck_dim, num_heads, ffn_expansion) for _ in range(num_blocks)
        )
        self.up_projection = nn.Linear(bottleneck_dim, token_dim)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    @staticmethod
    def _grid_size(num_tokens: int) -> tuple[int, int] | None:
        side = math.isqrt(num_tokens)
        return (side, side) if side * side == num_tokens else None

    def forward_with_delta(
        self,
        image_tokens: Tensor,
        pose_condition: Tensor | None = None,
        pose_valid_mask: Tensor | None = None,
        pose_confidence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden, input_dtype = _prepare_adapter_hidden(image_tokens, self.pre_norm, self.down_projection)
        context = hidden.mean(dim=1)
        context = context + self.pose_conditioner(context, pose_condition, pose_valid_mask, pose_confidence)
        weights = torch.softmax(self.scale_router(context), dim=-1)
        grid = self._grid_size(hidden.shape[1])
        if grid is not None:
            height, width = grid
            conv_input = hidden.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], height, width)
            conv3 = self.conv3(conv_input).flatten(2).transpose(1, 2)
            conv5 = self.conv5(conv_input).flatten(2).transpose(1, 2)
            hidden = hidden + weights[:, 0, None, None] * conv3 + weights[:, 1, None, None] * conv5
        # ResidualSpatialBlock skips its convolution automatically for a
        # non-square token sequence, retaining attention+FFN in that case.
        for block in self.blocks:
            hidden = block(hidden)
        delta = self.up_projection(hidden).to(dtype=input_dtype)
        return image_tokens + delta, delta

    def forward(self, image_tokens: Tensor, **pose_kwargs: Tensor | None) -> Tensor:
        return self.forward_with_delta(image_tokens, **pose_kwargs)[0]


class ImageRoutedMoEAdapter(_CheckpointableAdapter):
    """Image-global router over several residual token experts."""

    def __init__(
        self,
        token_dim: int,
        bottleneck_dim: int | None = None,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_expansion: int = 4,
        num_experts: int = 4,
        pose_enabled: bool = False,
        pose_dim: int = 9,
    ) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim or min(256, token_dim // 4)
        if bottleneck_dim % num_heads != 0:
            raise ValueError(f"bottleneck_dim ({bottleneck_dim}) must be divisible by num_heads ({num_heads})")
        if num_experts < 2:
            raise ValueError("Image-routed MoE requires at least two experts")
        self.spec = VariantFeatureAdapterSpec(
            token_dim=token_dim,
            bottleneck_dim=bottleneck_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            ffn_expansion=ffn_expansion,
            variant="image_routed_moe",
            pose_enabled=pose_enabled,
            pose_dim=pose_dim,
            num_experts=num_experts,
        )
        self.pre_norm = nn.LayerNorm(token_dim)
        self.down_projection = nn.Linear(token_dim, bottleneck_dim)
        self.pose_conditioner = OptionalPoseConditioner(bottleneck_dim, pose_dim, pose_enabled)
        self.router = nn.Linear(bottleneck_dim, num_experts)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.blocks = nn.ModuleList(
            ResidualSpatialBlock(bottleneck_dim, num_heads, ffn_expansion) for _ in range(num_blocks)
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(bottleneck_dim),
                nn.Linear(bottleneck_dim, bottleneck_dim * ffn_expansion),
                nn.GELU(),
                nn.Linear(bottleneck_dim * ffn_expansion, bottleneck_dim),
            )
            for _ in range(num_experts)
        )
        self.up_projection = nn.Linear(bottleneck_dim, token_dim)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward_with_delta(
        self,
        image_tokens: Tensor,
        pose_condition: Tensor | None = None,
        pose_valid_mask: Tensor | None = None,
        pose_confidence: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        hidden, input_dtype = _prepare_adapter_hidden(image_tokens, self.pre_norm, self.down_projection)
        for block in self.blocks:
            hidden = block(hidden)
        context = hidden.mean(dim=1)
        context = context + self.pose_conditioner(context, pose_condition, pose_valid_mask, pose_confidence)
        route = torch.softmax(self.router(context), dim=-1)
        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=1)
        hidden = hidden + (route[:, :, None, None] * expert_outputs).sum(dim=1)
        delta = self.up_projection(hidden).to(dtype=input_dtype)
        return image_tokens + delta, delta

    def forward(self, image_tokens: Tensor, **pose_kwargs: Tensor | None) -> Tensor:
        return self.forward_with_delta(image_tokens, **pose_kwargs)[0]


FEATURE_ADAPTER_VARIANTS = {
    "legacy": ViewFeatureAdapter,
    "view_feature_adapter": ViewFeatureAdapter,
    "pure_teacher_gated_residual": PureTeacherGatedResidualAdapter,
    "multi_scale_dynamic": MultiScaleDynamicAdapter,
    "image_routed_moe": ImageRoutedMoEAdapter,
}


def build_feature_adapter(
    variant: str,
    token_dim: int,
    bottleneck_dim: int | None = None,
    num_heads: int = 8,
    num_blocks: int = 2,
    ffn_expansion: int = 4,
    num_experts: int = 4,
    pose_enabled: bool = False,
    pose_dim: int = 9,
) -> _CheckpointableAdapter:
    try:
        adapter_class = FEATURE_ADAPTER_VARIANTS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown feature_adapter_variant={variant!r}; choose one of {sorted(FEATURE_ADAPTER_VARIANTS)}"
        ) from exc
    kwargs = {
        "token_dim": token_dim,
        "bottleneck_dim": bottleneck_dim,
        "num_heads": num_heads,
        "num_blocks": num_blocks,
        "ffn_expansion": ffn_expansion,
    }
    if adapter_class is not ViewFeatureAdapter:
        kwargs.update(
            {
                "pose_enabled": pose_enabled,
                "pose_dim": pose_dim,
            }
        )
    if adapter_class is ImageRoutedMoEAdapter:
        kwargs["num_experts"] = num_experts
    return adapter_class(**kwargs)


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
