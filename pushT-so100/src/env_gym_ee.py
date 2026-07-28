import os
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import cv2
from scipy.spatial.transform import Rotation as R
from helper import *
from state_features import JOINT_STATE_NAMES, OBS_STATE_DIM
from tcp_frame import (
    compute_mocap_to_tcp_transform,
    get_site_pose,
    mocap_pose_to_tcp_pose,
    pack_pose,
    tcp_pose_to_mocap_pose,
)

GRIPPER_OPEN_VAL  = 0.045
GRIPPER_CLOSE_VAL = 0.002

class PushT(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, xml_path, max_steps=300, render_mode=None):
        super(PushT, self).__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.pos_random_range = 0.03

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)

        # 8D action: [x, y, z, qw, qx, qy, qz, gripper(0=open,1=closed)]
        self.action_space = spaces.Box(
            low =np.array([-1,-1, 0,-1,-1,-1,-1, 0], dtype=np.float32),
            high=np.array([ 1, 1, 2, 1, 1, 1, 1, 1], dtype=np.float32),
        )

        self.observation_space = spaces.Dict({
            "cam_top":  spaces.Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
            "cam_side": spaces.Box(low=0, high=255, shape=(224,224,3), dtype=np.uint8),
            "observation.state": spaces.Box(low=-10.0, high=10.0, shape=(OBS_STATE_DIM,), dtype=np.float32),
        })

        self.act_names  = JOINT_STATE_NAMES
        self.act_ids    = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n+"_act") for n in self.act_names]
        self.joint_ids  = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    n)        for n in self.act_names]
        self.mocap_id   = self.model.body("target_mocap").mocapid[0]
        self.gripper_tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
        self._mocap_to_tcp = compute_mocap_to_tcp_transform(
            self.model, self.data, self.mocap_id, self.gripper_tcp_id
        )

        block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
        self.block_body_id  = block_body_id
        self.block_qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[block_body_id]]

        gj1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    "gripper_joint1")
        gj2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,    "gripper_joint2")
        ga1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_joint1_act")
        ga2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_joint2_act")
        self._gj_ids = (gj1, gj2, ga1, ga2)

        self.current_step = 0
        self.key_id = 0
        self._obs = None
        self._last_gripper = 0.0

    def _apply_action_8d(self, action):
        """action: TCP target [x,y,z, qw,qx,qy,qz, gripper(0=open,1=closed)]"""
        x, y, z, qw, qx, qy, qz, gripper = action

        mocap_pos, mocap_quat = tcp_pose_to_mocap_pose(
            [x, y, z], [qw, qx, qy, qz], self._mocap_to_tcp
        )
        self.data.mocap_pos[self.mocap_id] = mocap_pos
        self.data.mocap_quat[self.mocap_id] = mocap_quat

        # 夹爪: 0=open, 1=closed (连续值用阈值判断)
        self._last_gripper = float(gripper)
        gj1, gj2, ga1, ga2 = self._gj_ids
        is_closed = gripper > 0.5
        val1 =  GRIPPER_OPEN_VAL  if not is_closed else  GRIPPER_CLOSE_VAL
        val2 = -GRIPPER_OPEN_VAL  if not is_closed else -GRIPPER_CLOSE_VAL
        # if gj1 >= 0: self.data.qpos[self.model.jnt_qposadr[gj1]] = val1
        # if gj2 >= 0: self.data.qpos[self.model.jnt_qposadr[gj2]] = val2
        if ga1 >= 0: self.data.ctrl[ga1] = val1
        if ga2 >= 0: self.data.ctrl[ga2] = val2

    def get_observation(self):
        self.renderer.update_scene(self.data, camera="top_view")
        img_top = self.renderer.render().copy()
        self.renderer.update_scene(self.data, camera="side_view")
        img_side = self.renderer.render().copy()
        joint_qpos = np.array(
            [self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids],
            dtype=np.float32
        )
        tcp_pos, tcp_quat = get_site_pose(self.data, self.gripper_tcp_id)
        ee_state = pack_pose(tcp_pos, tcp_quat, self._read_gripper_state())
        obs_state = np.concatenate([joint_qpos, ee_state]).astype(np.float32)
        self._obs = {"cam_top": img_top, "cam_side": img_side, "observation.state": obs_state}
        return self._obs

    def _read_gripper_state(self):
        return self._last_gripper

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self._last_gripper = 0.0
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        mujoco.mj_forward(self.model, self.data)

        tcp_pos, tcp_quat = get_site_pose(self.data, self.gripper_tcp_id)
        mocap_pos, mocap_quat = tcp_pose_to_mocap_pose(tcp_pos, tcp_quat, self._mocap_to_tcp)
        self.data.mocap_pos[self.mocap_id] = mocap_pos
        self.data.mocap_quat[self.mocap_id] = mocap_quat

        # 随机放置圆柱体（桌面中心 y=0.4，高度 0.403）
        self.data.qpos[self.block_qpos_adr:self.block_qpos_adr+3] = [
            np.random.uniform(-self.pos_random_range, self.pos_random_range),
            0.4 + np.random.uniform(-self.pos_random_range, self.pos_random_range),
            0.403
        ]
        rad = np.random.uniform(-3.14, 3.14)
        self.data.qpos[self.block_qpos_adr+3:self.block_qpos_adr+7] = [
            np.cos(rad/2), 0.0, 0.0, np.sin(rad/2)
        ]
        mujoco.mj_forward(self.model, self.data)
        return self.get_observation(), {}

    def step(self, action):
        """action: 8D TCP target [x,y,z,qw,qx,qy,qz,gripper]"""
        # Clamp TCP target to safe workspace. This is intentionally wider than the
        # old mocap workspace because TCP and mocap frames have a fixed offset.
        x = np.clip(float(action[0]), -0.30, 0.30)
        y = np.clip(float(action[1]), -0.05, 0.65)
        z = np.clip(float(action[2]),  0.35, 0.90)
        q = np.array(action[3:7], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm > 1e-8:
            q = q / norm
        grip = np.clip(float(action[7]), 0.0, 1.0)
        self._last_gripper = grip

        mocap_target_pos, mocap_target_quat = tcp_pose_to_mocap_pose(
            [x, y, z], q, self._mocap_to_tcp
        )

        # Get current mocap position for smoothing
        cur_pos = self.data.mocap_pos[self.mocap_id].copy()
        target_pos = mocap_target_pos
        cur_quat = self.data.mocap_quat[self.mocap_id].copy()
        q = mocap_target_quat

        control_dt = 1.0 / 10.0
        sim_steps = int(control_dt / self.model.opt.timestep)

        for step_i in range(sim_steps):
            # Smoothly interpolate mocap towards target
            alpha = (step_i + 1) / sim_steps
            interp_pos = cur_pos + (target_pos - cur_pos) * alpha
            self.data.mocap_pos[self.mocap_id] = interp_pos
            # SLERP for quaternion
            dot = np.dot(cur_quat, q)
            # Ensure shortest path
            if dot < 0:
                q = -q
                dot = -dot
            dot = np.clip(dot, -1.0, 1.0)
            if dot > 0.9999:
                interp_quat = cur_quat + (q - cur_quat) * alpha
            else:
                theta_0 = np.arccos(dot)
                theta = theta_0 * alpha
                s0 = np.sin(theta_0 - theta) / np.sin(theta_0)
                s1 = np.sin(theta) / np.sin(theta_0)
                interp_quat = s0 * cur_quat + s1 * q
            self.data.mocap_quat[self.mocap_id] = interp_quat

            # Set gripper
            gj1, gj2, ga1, ga2 = self._gj_ids
            is_closed = grip > 0.5
            val1 =  GRIPPER_OPEN_VAL  if not is_closed else  GRIPPER_CLOSE_VAL
            val2 = -GRIPPER_OPEN_VAL  if not is_closed else -GRIPPER_CLOSE_VAL
            # val1 = GRIPPER_OPEN_VAL - (GRIPPER_OPEN_VAL - GRIPPER_CLOSE_VAL) * grip
            # val2 = -val1
            # if gj1 >= 0: self.data.qpos[self.model.jnt_qposadr[gj1]] = val1
            # if gj2 >= 0: self.data.qpos[self.model.jnt_qposadr[gj2]] = val2
            if ga1 >= 0: self.data.ctrl[ga1] = val1
            if ga2 >= 0: self.data.ctrl[ga2] = val2
            
            # Set position control targets
            for aid, jid in zip(self.act_ids, self.joint_ids):
                self.data.ctrl[aid] = self.data.qpos[self.model.jnt_qposadr[jid]]

            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_step(self.model, self.data)

        obs = self.get_observation()

        # 抓取任务奖励
        ee_pos    = self.data.site_xpos[self.gripper_tcp_id]
        block_pos = self.data.xpos[self.block_body_id]
        dist      = float(np.linalg.norm(ee_pos - block_pos))

        # 基础奖励：EE 靠近方块
        reward = -dist
        # 夹爪关闭且靠近：额外奖励
        is_closed = action[7] > 0.5
        if dist < 0.05 and is_closed:
            reward += 5.0
        # 成功：方块被抬起（z > 0.45）
        block_lifted = block_pos[2] > 0.45
        terminated = bool(block_lifted)
        if terminated:
            reward += 50.0

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        info = {"dist_to_block": dist, "block_lifted": block_lifted}
        return obs, reward, terminated, truncated, info

    def render(self):
        img1 = cv2.resize(self._obs["cam_top"],  (448, 448))
        img2 = cv2.resize(self._obs["cam_side"], (448, 448))
        return np.concatenate([img1, img2], axis=1)

    def close(self):
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        return super().close()
