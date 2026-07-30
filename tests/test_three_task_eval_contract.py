"""Tests for the multi-task evaluation contract."""

from script.hammer_task_prompts import hammer_instruction_for_block_x
from script.evaluate_pi05_three_tasks_camera_views import TASK_SPECS


def test_multi_task_prompts_match_training_manifest():
    prompts = {spec.name: spec.instruction for spec in TASK_SPECS}
    assert prompts == {
        "place_two_cubes_box": (
            "Put the yellow cube into the black box first, then put the green cube into the black box."
        ),
        "pick_dual_bottles": "Pick up both bottles, one with each arm.",
        "beat_block_hammer": "Take the hammer in the right gripper and strike the block.",
    }

    hammer = next(spec for spec in TASK_SPECS if spec.name == "beat_block_hammer")
    assert hammer.hammer_arm_aware_instruction is True


def test_hammer_prompt_matches_the_task_selected_arm():
    assert hammer_instruction_for_block_x(-0.2, 0.0) == (
        "Use the left arm to pick up the hammer and strike the block."
    )
    assert hammer_instruction_for_block_x(0.2, 0.0) == (
        "Take the hammer in the right gripper and strike the block."
    )
