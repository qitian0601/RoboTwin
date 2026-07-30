"""Canonical language conditions for the two Hammer task variants."""

HAMMER_ARM_INSTRUCTIONS = {
    "left": "Use the left arm to pick up the hammer and strike the block.",
    "right": "Take the hammer in the right gripper and strike the block.",
}


def hammer_instruction_for_block_x(block_x: float, table_x: float) -> str:
    """Choose the prompt for the arm selected by the Hammer task geometry."""
    arm = "left" if block_x < table_x else "right"
    return HAMMER_ARM_INSTRUCTIONS[arm]
