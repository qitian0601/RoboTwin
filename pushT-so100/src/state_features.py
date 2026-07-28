JOINT_STATE_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

EE_STATE_NAMES = [
    "tcp_x",
    "tcp_y",
    "tcp_z",
    "tcp_qw",
    "tcp_qx",
    "tcp_qy",
    "tcp_qz",
    "gripper",
]

OBS_STATE_NAMES = JOINT_STATE_NAMES + EE_STATE_NAMES
OBS_STATE_DIM = len(OBS_STATE_NAMES)
