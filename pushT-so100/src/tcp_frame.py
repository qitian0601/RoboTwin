import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_xyzw(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def quat_xyzw_to_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def normalize_quat_wxyz(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm > 1e-8:
        quat = quat / norm
    return quat


def pose_to_matrix(pos, quat_wxyz):
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_quat(quat_wxyz_to_xyzw(normalize_quat_wxyz(quat_wxyz))).as_matrix()
    mat[:3, 3] = np.asarray(pos, dtype=np.float64)
    return mat


def matrix_to_pose(mat):
    pos = np.asarray(mat[:3, 3], dtype=np.float64)
    quat = quat_xyzw_to_wxyz(R.from_matrix(mat[:3, :3]).as_quat())
    return pos, normalize_quat_wxyz(quat)


def get_body_pose(data, body_id):
    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def get_site_pose(data, site_id):
    pos = data.site_xpos[site_id].copy()
    xmat = data.site_xmat[site_id].reshape(3, 3).copy()
    quat = quat_xyzw_to_wxyz(R.from_matrix(xmat).as_quat())
    return pos, quat


def compute_mocap_to_tcp_transform(model, data, mocap_id, tcp_site_id):
    """Return constant transform T_mocap_tcp for the current equality setup."""
    mujoco.mj_forward(model, data)
    mocap_world = pose_to_matrix(data.mocap_pos[mocap_id], data.mocap_quat[mocap_id])
    tcp_pos, tcp_quat = get_site_pose(data, tcp_site_id)
    tcp_world = pose_to_matrix(tcp_pos, tcp_quat)
    return np.linalg.inv(mocap_world) @ tcp_world


def mocap_pose_to_tcp_pose(mocap_pos, mocap_quat, mocap_to_tcp):
    tcp_world = pose_to_matrix(mocap_pos, mocap_quat) @ mocap_to_tcp
    return matrix_to_pose(tcp_world)


def tcp_pose_to_mocap_pose(tcp_pos, tcp_quat, mocap_to_tcp):
    mocap_world = pose_to_matrix(tcp_pos, tcp_quat) @ np.linalg.inv(mocap_to_tcp)
    return matrix_to_pose(mocap_world)


def pack_pose(pos, quat_wxyz, gripper):
    return np.concatenate(
        [
            np.asarray(pos, dtype=np.float64),
            normalize_quat_wxyz(quat_wxyz),
            np.array([float(gripper)], dtype=np.float64),
        ]
    ).astype(np.float32)

