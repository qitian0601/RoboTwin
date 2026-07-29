"""Tests for the multi-task evaluation contract."""

from script.evaluate_pi05_three_tasks_camera_views import TASK_SPECS


def test_multi_task_prompts_match_training_manifest():
    prompts = {spec.name: spec.instruction for spec in TASK_SPECS}
    assert prompts == {
        "place_two_cubes_box": (
            "Put the yellow cube into the black box first, then put the green cube into the black box."
        ),
        "pick_dual_bottles": "Pick up both bottles, one with each arm.",
        "beat_block_hammer": "Pick up the hammer with the right arm and strike the block.",
    }
