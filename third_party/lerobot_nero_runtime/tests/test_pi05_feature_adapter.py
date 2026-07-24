from __future__ import annotations

import pytest
import torch
from torch import nn

from lerobot.policies.pi05.feature_adapter import ViewFeatureAdapter


def make_adapter() -> ViewFeatureAdapter:
    return ViewFeatureAdapter(token_dim=64, bottleneck_dim=16, num_heads=8)


@pytest.mark.parametrize("num_tokens", [16, 15])
def test_adapter_identity_initialization(num_tokens: int) -> None:
    adapter = make_adapter()
    tokens = torch.randn(2, num_tokens, 64)

    output, delta = adapter.forward_with_delta(tokens)

    torch.testing.assert_close(output, tokens, atol=0.0, rtol=0.0)
    torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0.0, rtol=0.0)


def test_only_adapter_receives_gradients() -> None:
    backbone = nn.Linear(64, 32)
    backbone.requires_grad_(False)
    adapter = make_adapter()
    tokens = torch.randn(2, 16, 64)

    loss = backbone(adapter(tokens)).square().mean()
    loss.backward()

    assert all(parameter.grad is None for parameter in backbone.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in adapter.parameters()
    )


def test_teacher_student_loss_is_reproducible() -> None:
    torch.manual_seed(7)
    adapter = make_adapter()
    velocity_head = nn.Linear(64, 12).requires_grad_(False)
    canonical = torch.randn(2, 16, 64)
    shifted = torch.randn(2, 16, 64)
    flow_target = torch.randn(2, 16, 12)

    def paired_loss() -> torch.Tensor:
        with torch.no_grad():
            teacher_velocity = velocity_head(canonical)
        student_velocity = velocity_head(adapter(shifted))
        return (student_velocity - flow_target).square().mean() + 0.5 * (
            student_velocity - teacher_velocity
        ).square().mean()

    first = paired_loss()
    second = paired_loss()
    torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


def test_adapter_checkpoint_round_trip(tmp_path) -> None:
    adapter = make_adapter()
    with torch.no_grad():
        adapter.up_projection.weight.normal_(std=0.01)
        adapter.up_projection.bias.normal_(std=0.01)
    tokens = torch.randn(2, 16, 64)
    expected = adapter(tokens)

    adapter.save_checkpoint(tmp_path)
    restored = make_adapter()
    restored.load_checkpoint(tmp_path)

    torch.testing.assert_close(restored(tokens), expected, atol=0.0, rtol=0.0)


def test_small_batch_overfit_reduces_flow_and_velocity_losses() -> None:
    torch.manual_seed(11)
    adapter = make_adapter()
    teacher_head = nn.Linear(64, 12).requires_grad_(False)
    canonical = torch.randn(2, 16, 64)
    shifted = canonical + 0.25 * torch.randn_like(canonical)
    with torch.no_grad():
        teacher_velocity = teacher_head(canonical)
        flow_target = teacher_velocity + 0.05 * torch.randn_like(teacher_velocity)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=3e-3, weight_decay=0.0)

    def losses() -> tuple[torch.Tensor, torch.Tensor]:
        student_velocity = teacher_head(adapter(shifted))
        return (
            (student_velocity - flow_target).square().mean(),
            (student_velocity - teacher_velocity).square().mean(),
        )

    initial_flow, initial_velocity = (value.detach() for value in losses())
    for _ in range(40):
        flow_loss, velocity_loss = losses()
        loss = flow_loss + 0.5 * velocity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    final_flow, final_velocity = (value.detach() for value in losses())

    assert final_flow < initial_flow * 0.5
    assert final_velocity < initial_velocity * 0.5


def test_zero_initialized_adapter_preserves_velocity_prediction() -> None:
    torch.manual_seed(17)
    adapter = make_adapter()
    frozen_velocity_model = nn.Sequential(
        nn.Linear(64, 64),
        nn.GELU(),
        nn.Linear(64, 12),
    ).requires_grad_(False)
    tokens = torch.randn(2, 16, 64)

    baseline = frozen_velocity_model(tokens)
    adapted = frozen_velocity_model(adapter(tokens))

    torch.testing.assert_close(adapted, baseline, atol=0.0, rtol=0.0)
