from pathlib import Path

import torch
from safetensors.torch import save_file

from script.make_scaled_feature_adapter_checkpoint import OUTPUT_KEYS, scaled_state_dict


def test_scaled_state_dict_changes_only_output_projection(tmp_path: Path) -> None:
    source = tmp_path / "adapter.safetensors"
    original = {
        "down_projection.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "up_projection.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "up_projection.bias": torch.arange(4, dtype=torch.float32),
    }
    save_file(original, source)

    scaled = scaled_state_dict(source, 0.25)

    assert set(OUTPUT_KEYS) == {"up_projection.weight", "up_projection.bias"}
    torch.testing.assert_close(
        scaled["down_projection.weight"], original["down_projection.weight"]
    )
    for key in OUTPUT_KEYS:
        torch.testing.assert_close(scaled[key], original[key] * 0.25)


def test_unit_scale_is_exactly_the_original_checkpoint(tmp_path: Path) -> None:
    source = tmp_path / "adapter.safetensors"
    original = {
        "up_projection.weight": torch.randn(4, 2),
        "up_projection.bias": torch.randn(4),
    }
    save_file(original, source)

    scaled = scaled_state_dict(source, 1.0)

    for key, value in original.items():
        torch.testing.assert_close(scaled[key], value, atol=0.0, rtol=0.0)
