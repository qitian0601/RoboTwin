from __future__ import annotations

import json

import pytest
import torch

from lerobot.policies.pi05.feature_adapter import (
    FEATURE_ADAPTER_VARIANTS,
    MultiScaleDynamicAdapter,
    build_feature_adapter,
)


CANDIDATE_VARIANTS = (
    "pure_teacher_gated_residual",
    "multi_scale_dynamic",
    "image_routed_moe",
)


@pytest.mark.parametrize("variant", ("legacy", *CANDIDATE_VARIANTS))
@pytest.mark.parametrize("num_tokens", (16, 15))
def test_all_adapter_variants_are_exact_identity_at_initialization(
    variant: str, num_tokens: int
) -> None:
    adapter = build_feature_adapter(
        variant,
        token_dim=64,
        bottleneck_dim=16,
        num_heads=8,
        num_blocks=2,
        num_experts=3,
    )
    tokens = torch.randn(2, num_tokens, 64)

    output, delta = adapter.forward_with_delta(tokens)

    torch.testing.assert_close(output, tokens, atol=0.0, rtol=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("variant", CANDIDATE_VARIANTS)
def test_candidate_adapter_has_nonzero_gradient_while_backbone_stays_frozen(variant: str) -> None:
    adapter = build_feature_adapter(
        variant,
        token_dim=64,
        bottleneck_dim=16,
        num_heads=8,
        num_experts=3,
    )
    backbone = torch.nn.Linear(64, 12).requires_grad_(False)
    tokens = torch.randn(2, 16, 64)

    backbone(adapter(tokens)).square().mean().backward()

    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in adapter.parameters()
    )


@pytest.mark.parametrize("variant", CANDIDATE_VARIANTS)
def test_missing_invalid_or_zero_confidence_pose_is_exact_image_only_fallback(variant: str) -> None:
    torch.manual_seed(4)
    adapter = build_feature_adapter(
        variant,
        token_dim=64,
        bottleneck_dim=16,
        num_heads=8,
        num_experts=3,
        pose_enabled=True,
        pose_dim=9,
    )
    with torch.no_grad():
        adapter.up_projection.weight.normal_(std=0.02)
        adapter.up_projection.bias.normal_(std=0.02)
        adapter.pose_conditioner.projection.weight.normal_(std=0.05)
        adapter.pose_conditioner.projection.bias.normal_(std=0.05)
    tokens = torch.randn(2, 16, 64)
    pose = torch.randn(2, 9)
    image_only = adapter(tokens)

    invalid = adapter(
        tokens,
        pose_condition=pose,
        pose_valid_mask=torch.zeros(2),
        pose_confidence=torch.ones(2),
    )
    zero_confidence = adapter(
        tokens,
        pose_condition=pose,
        pose_valid_mask=torch.ones(2),
        pose_confidence=torch.zeros(2),
    )

    torch.testing.assert_close(invalid, image_only, atol=0.0, rtol=0.0)
    torch.testing.assert_close(zero_confidence, image_only, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("variant", CANDIDATE_VARIANTS)
def test_adapter_checkpoint_round_trip_preserves_variant_and_output(tmp_path, variant: str) -> None:
    torch.manual_seed(9)
    adapter = build_feature_adapter(
        variant,
        token_dim=64,
        bottleneck_dim=16,
        num_heads=8,
        num_experts=3,
    )
    with torch.no_grad():
        adapter.up_projection.weight.normal_(std=0.01)
        adapter.up_projection.bias.normal_(std=0.01)
    tokens = torch.randn(2, 16, 64)
    expected = adapter(tokens)
    checkpoint = tmp_path / variant

    adapter.save_checkpoint(checkpoint)
    restored = build_feature_adapter(
        variant,
        token_dim=64,
        bottleneck_dim=16,
        num_heads=8,
        num_experts=3,
    )
    restored.load_checkpoint(checkpoint)

    spec = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    assert spec["variant"] == variant
    torch.testing.assert_close(restored(tokens), expected, atol=0.0, rtol=0.0)


def test_multiscale_non_grid_tokens_skip_all_convolutions() -> None:
    adapter = MultiScaleDynamicAdapter(token_dim=64, bottleneck_dim=16, num_heads=8)
    calls: list[str] = []
    hooks = [
        adapter.conv3.register_forward_hook(lambda *_: calls.append("conv3")),
        adapter.conv5.register_forward_hook(lambda *_: calls.append("conv5")),
    ]
    hooks.extend(
        block.depthwise_conv.register_forward_hook(lambda *_: calls.append("block_conv"))
        for block in adapter.blocks
    )
    try:
        adapter(torch.randn(2, 15, 64))
    finally:
        for hook in hooks:
            hook.remove()
    assert calls == []


def test_declared_variants_include_stable_legacy_and_three_candidates() -> None:
    assert {"legacy", *CANDIDATE_VARIANTS}.issubset(FEATURE_ADAPTER_VARIANTS)
