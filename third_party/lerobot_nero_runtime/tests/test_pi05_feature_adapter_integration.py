from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


CHECKPOINT = os.environ.get("PI05_ADAPTER_TEST_CHECKPOINT")


@pytest.mark.skipif(
    not CHECKPOINT or not torch.cuda.is_available(),
    reason="Set PI05_ADAPTER_TEST_CHECKPOINT and provide CUDA for the full PI0.5 test",
)
def test_full_pi05_adapter_identity_freezing_and_reproducibility() -> None:
    checkpoint = Path(CHECKPOINT)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = "cuda"
    config.use_feature_adapter = True
    config.feature_adapter_checkpoint = None
    policy = PI05Policy.from_pretrained(checkpoint, config=config, strict=True)
    policy.configure_feature_adapter_training()

    batch_size = 1
    camera_shape = (batch_size, 3, *config.image_resolution)
    canonical = {
        "observation.images.front": torch.rand(camera_shape, device="cuda"),
        "observation.images.left_wrist": torch.rand(camera_shape, device="cuda"),
        "observation.images.right_wrist": torch.rand(camera_shape, device="cuda"),
        "observation.language.tokens": torch.zeros(
            batch_size, config.tokenizer_max_length, dtype=torch.long, device="cuda"
        ),
        "observation.language.attention_mask": torch.ones(
            batch_size, config.tokenizer_max_length, dtype=torch.bool, device="cuda"
        ),
        "action": torch.randn(batch_size, config.chunk_size, 16, device="cuda"),
    }
    shifted = dict(canonical)
    shifted["observation.images.front"] = torch.rand(camera_shape, device="cuda")
    actions = policy.prepare_action(canonical)
    noise = policy.model.sample_noise(actions.shape, actions.device)
    time = policy.model.sample_time(batch_size, actions.device)
    images, image_masks, image_keys = policy._preprocess_images(canonical)

    with torch.no_grad():
        baseline_velocity, _ = policy.model.predict_velocity(
            images,
            image_masks,
            canonical["observation.language.tokens"],
            canonical["observation.language.attention_mask"],
            actions,
            noise,
            time,
            image_keys=image_keys,
            apply_feature_adapter=False,
        )
        adapted_velocity, _ = policy.model.predict_velocity(
            images,
            image_masks,
            canonical["observation.language.tokens"],
            canonical["observation.language.attention_mask"],
            actions,
            noise,
            time,
            image_keys=image_keys,
            apply_feature_adapter=True,
        )
        first_loss, _ = policy.forward_feature_adapter(
            canonical, shifted, noise=noise, time=time
        )
        second_loss, _ = policy.forward_feature_adapter(
            canonical, shifted, noise=noise, time=time
        )

    torch.testing.assert_close(adapted_velocity, baseline_velocity, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first_loss, second_loss, atol=0.0, rtol=0.0)

    loss, _ = policy.forward_feature_adapter(canonical, shifted, noise=noise, time=time)
    loss.backward()
    assert all(
        parameter.grad is None
        for name, parameter in policy.named_parameters()
        if not name.startswith("model.feature_adapter.")
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.feature_adapter.parameters()
    )
