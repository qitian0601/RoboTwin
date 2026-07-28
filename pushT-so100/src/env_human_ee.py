import os
import argparse
from pathlib import Path

# Offscreen rendering configuration
os.environ["MUJOCO_GL"] = "egl"
_REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(_REPO_ROOT / ".cache" / "huggingface" / "datasets"))

import time
import threading
import logging
import mujoco
import mujoco.viewer
import numpy as np
import pygame
import cv2
from scipy.spatial.transform import Rotation as R
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from helper import *
from state_features import JOINT_STATE_NAMES, OBS_STATE_NAMES
from tcp_frame import (
    compute_mocap_to_tcp_transform,
    get_site_pose,
    mocap_pose_to_tcp_pose,
    pack_pose,
)

# --- Argument parsing ---
parser = argparse.ArgumentParser(description="MuJoCo SO100 data collection script")

parser.add_argument(
    "--repo_id",
    type=str,
    default="./data/NewData3.9-ee-2d-pos",
    help="Path to the LeRobot dataset repository"
)
parser.add_argument(
    "--fps",
    type=int,
    default=10,
    help="Recording frame rate"
)
parser.add_argument(
    "--move_speed",
    type=float,
    default=0.05,
    help="Mocap translation speed"
)
parser.add_argument(
    "--rot_speed",
    type=float,
    default=0.3,
    help="Mocap rotation speed"
)

args = parser.parse_args()

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Import LeRobot related libraries
# --- Global configuration ---
MOVE_SPEED = args.move_speed
ROT_SPEED = args.rot_speed
DEADZONE = 0.1
FPS = args.fps
VIDEO_STEP = 1.0 / FPS
OBS_IMAGES_SHAPE = (224, 224, 3)

# Environment configuration
pos_random_range = 0.05

# LeRobot dataset path and parameters
REPO_ID =  Path(args.repo_id).absolute()
ROBOT_TYPE = "nero_arm"
TOLERANCE = 0.0001  # Tolerance for action deduplication

# Recording state
is_recording = False
record_buffer = []  # Temporary buffer for the current episode
save_thread = None  # Used to track the background saving thread

# Gripper state (open=True, closed=False)
gripper_open = True
GRIPPER_OPEN_VAL  = 0.045   # gripper_joint1 open qpos (range 0~0.05), 最大开合
GRIPPER_CLOSE_VAL = 0.002   # gripper_joint1 closed qpos

# mocap 位置限幅，防止超出工作空间导致物理崩溃归位
MOCAP_POS_MIN = np.array([-0.8, -0.8, 0.1])
MOCAP_POS_MAX = np.array([ 0.8,  0.8, 1.4])

# Button debounce
buttonCooldown = 0.0
COOLDOWN_SEC = 0.3

# --- Load MuJoCo model ---
XML_PATH = "./chernyadev mujoco_menagerie add-so-arm100 trs_so_arm100/nero_human_env.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Get joint and mocap IDs
# Nero 臂 7 个关节（不含夹爪，夹爪单独控制）
act_names = JOINT_STATE_NAMES
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_act") for n in act_names]
mocap_name = "target_mocap"
mocap_id = model.body(mocap_name).mocapid[0]
tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
mocap_to_tcp = compute_mocap_to_tcp_transform(model, data, mocap_id, tcp_site_id)

# Address of target object block
block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_block")
block_qpos_adr = model.jnt_qposadr[model.body_jntadr[block_body_id]]
is_success = False

dataset = None


# --- Initialize hardware input ---
pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
if joystick:
    joystick.init()
    logger.info(f"Joystick: {joystick.get_name()}")
else:
    logger.warning("No joystick detected - keyboard-only mode (no control input)")

# --- Initialize LeRobot dataset ---
def init_lerobot_dataset(repo_path):
    features = {
        "observation.images.cam_top": {
            "dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]
        },
        "observation.images.cam_side": {
            "dtype": "video", "shape": OBS_IMAGES_SHAPE, "names": ["channels", "height", "width"]
        },
        "observation.state": {
            "dtype": "float32", "shape": (len(OBS_STATE_NAMES),), "names": OBS_STATE_NAMES
        },
        "action": {
            "dtype": "float32", "shape": (8,),
            "names": ["tcp_x", "tcp_y", "tcp_z",
                      "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz",
                      "gripper"]
        }
    }

    if os.path.exists(repo_path):
        logger.info(f"Existing dataset detected, loading: {repo_path}")
        return LeRobotDataset(
            repo_id=Path(repo_path).name,
            root=Path(repo_path)
        )
    else:
        logger.info(f"Dataset not found, creating new repository: {repo_path}")
        return LeRobotDataset.create(
            repo_id=repo_path,
            fps=FPS,
            robot_type=ROBOT_TYPE,
            features=features
        )

def get_mocap_4d_pose(mocap_pos, mocap_quat, gripper_open: bool):
    """返回 8 维 action: [x, y, z, qw, qx, qy, qz, gripper(0=open,1=closed)]"""
    x, y, z = mocap_pos
    qw, qx, qy, qz = mocap_quat  # MuJoCo 格式: (w, x, y, z)
    gripper_val = 0.0 if gripper_open else 1.0
    return np.array([x, y, z, qw, qx, qy, qz, gripper_val], dtype=np.float32)


def get_tcp_target_pose_from_mocap(gripper_open: bool):
    tcp_pos, tcp_quat = mocap_pose_to_tcp_pose(
        data.mocap_pos[mocap_id].copy(),
        data.mocap_quat[mocap_id].copy(),
        mocap_to_tcp,
    )
    gripper_val = 0.0 if gripper_open else 1.0
    return pack_pose(tcp_pos, tcp_quat, gripper_val)


def get_current_tcp_state(gripper_open: bool):
    tcp_pos, tcp_quat = get_site_pose(data, tcp_site_id)
    gripper_val = 0.0 if gripper_open else 1.0
    return pack_pose(tcp_pos, tcp_quat, gripper_val)

def async_save_to_lerobot(frames_list):
    """Process data and write it to disk in a separate thread."""
    global dataset
    if len(frames_list) < 2:
        return
        
    num_frames = len(frames_list)
    logger.info(f"[Background thread] Start processing new episode ({num_frames} frames)")
    added_count = 0
    last_action = None
    
    for i in range(num_frames):
        curr_frame = frames_list[i]
        target_action = curr_frame["action"]

        # --- Action deduplication ---
        if last_action is not None and np_allabs(target_action - last_action):
            continue
        frame_data = {
            "observation.images.cam_top": curr_frame["cam_top"],
            "observation.images.cam_side": curr_frame["cam_side"],
            "observation.state": curr_frame["state"].astype(np.float32),
            "action": target_action,
            "task": "grasp the red cylinder on the table"
        }
        dataset.add_frame(frame_data)
        added_count += 1
        last_action = target_action
    
    if added_count > 0:
        dataset.save_episode()
        logger.info(f"[Background thread] Save completed, valid frames: {added_count}, total episodes: {dataset.num_episodes}")
    else:
        logger.warning("[Background thread] Skipped: no valid action changes.")

# Utility function: check whether all absolute values are below tolerance
def np_allabs(x):
    return np.all(np.abs(x) < TOLERANCE)

# --- Joystick and recording control ---
def record_toggle():
    global is_recording, record_buffer, save_thread
    if is_recording:
        is_recording = False
        logger.info("Recording stopped.")
        if len(record_buffer) > 0:
            # Start a new thread for saving to avoid blocking the main loop
            save_thread = threading.Thread(
                target=async_save_to_lerobot,
                args=(list(record_buffer),),
                daemon=True
            )
            save_thread.start()
        record_buffer = []
    else:
        logger.info("Recording started (Recording ON)")
        record_buffer = []
        is_recording = True

def reset_env():
    global is_success, is_recording, record_buffer, pos_random_range
    is_success = False
    is_recording = False
    record_buffer.clear()
    key_id = 0

    # Reset model state to home keyframe
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Randomize block position on the table (±5cm)
    data.qpos[block_qpos_adr:block_qpos_adr+3] = [
        np.random.uniform(-pos_random_range, pos_random_range),          # x
        0.4 + np.random.uniform(-pos_random_range, pos_random_range),    # y (桌面中心)
        0.403                                                             # z (圆柱体中心在桌面上方)
    ]
    # 随机旋转
    rad = np.random.uniform(-3.14, 3.14)
    data.qpos[block_qpos_adr+3:block_qpos_adr+7] = [np.cos(rad/2), 0.0, 0.0, np.sin(rad/2)]
    mujoco.mj_forward(model, data)

def apply_rotation(mocap_id, axis: str, angle: float):
    """绕 TCP 当前世界坐标为旋转中心旋转 mocap，保持 TCP 世界位置不变，只改变朝向。

    直接从仿真状态读取 TCP site 的当前世界坐标，无需硬编码任何偏移。
    """
    if abs(angle) < 1e-8:
        return

    # 旋转前：读取 TCP 当前世界坐标
    tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")
    tcp_world_before = data.site_xpos[tcp_site_id].copy()

    # 叠加旋转（绕世界系轴）
    q = data.mocap_quat[mocap_id]                      # MuJoCo: (w, x, y, z)
    r_curr = R.from_quat([q[1], q[2], q[3], q[0]])    # scipy: (x, y, z, w)
    r_delta = R.from_euler(axis, angle)
    r_new = r_delta * r_curr
    new_q = r_new.as_quat()                            # scipy: (x, y, z, w)
    data.mocap_quat[mocap_id] = [new_q[3], new_q[0], new_q[1], new_q[2]]

    # 更新物理状态，让 site_xpos 反映新朝向
    mujoco.mj_forward(model, data)

    # 旋转后：TCP 的新世界坐标
    tcp_world_after = data.site_xpos[tcp_site_id].copy()

    # 补偿 mocap_pos，使 TCP 保持在旋转前的世界位置
    data.mocap_pos[mocap_id] += tcp_world_before - tcp_world_after

    # 再次 forward 使补偿生效
    mujoco.mj_forward(model, data)


def set_gripper(open_gripper: bool):
    """直接设置夹爪关节 qpos 和 ctrl。"""
    val1 =  GRIPPER_OPEN_VAL  if open_gripper else  GRIPPER_CLOSE_VAL
    val2 = -GRIPPER_OPEN_VAL  if open_gripper else -GRIPPER_CLOSE_VAL
    # gripper_joint1 (range 0~0.05), gripper_joint2 (range -0.05~0)
    gj1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1")
    gj2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint2")
    ga1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_joint1_act")
    ga2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_joint2_act")
    if gj1_id >= 0:
        data.qpos[model.jnt_qposadr[gj1_id]] = val1
    if gj2_id >= 0:
        data.qpos[model.jnt_qposadr[gj2_id]] = val2
    if ga1_id >= 0:
        data.ctrl[ga1_id] = val1
    if ga2_id >= 0:
        data.ctrl[ga2_id] = val2
    state_str = "open" if open_gripper else "closed"
    logger.info(f"Gripper -> {state_str} (joint1={val1:.4f}, joint2={val2:.4f})")


def joystick_control():
    global buttonCooldown, gripper_open
    if not joystick:
        return False
    pygame.event.pump()

    # 安全读取摇杆和扳机的值
    num_axes = joystick.get_numaxes()

    # ── Axis 读取 ─────────────────────────────────────────────
    # 左摇杆  Axis 0/1  -> X/Y 平移
    ax0 = joystick.get_axis(0) if num_axes > 0 else 0.0
    ax1 = joystick.get_axis(1) if num_axes > 1 else 0.0
    # 扳机    LT=Axis2  RT=Axis5 -> Z 高度
    lt  = joystick.get_axis(2) if num_axes > 2 else -1.0
    rt  = joystick.get_axis(5) if num_axes > 5 else -1.0
    # 右摇杆  Axis 3=X(Yaw)  Axis 4=Y(Roll)
    ax3 = joystick.get_axis(3) if num_axes > 3 else 0.0   # 右摇杆左右 -> Yaw
    ax4 = joystick.get_axis(4) if num_axes > 4 else 0.0   # 右摇杆上下 -> Roll
    # D-pad  Hat (pygame hat 0)
    num_hats = joystick.get_numhats()
    hat_x, hat_y = joystick.get_hat(0) if num_hats > 0 else (0, 0)

    # ── 平移增量 ──────────────────────────────────────────────
    move_dt = MOVE_SPEED * model.opt.timestep
    rot_dt  = ROT_SPEED  * model.opt.timestep

    pos_dx = (abs(ax0) > DEADZONE) *  ax0 * move_dt
    pos_dy = (abs(ax1) > DEADZONE) * -ax1 * move_dt
    lt_val = (lt + 1.0) / 2.0 if lt > -0.9 else 0.0
    rt_val = (rt + 1.0) / 2.0 if rt > -0.9 else 0.0
    pos_dz = (rt_val - lt_val) * move_dt

    new_pos = data.mocap_pos[mocap_id] + np.array([pos_dx, pos_dy, pos_dz])
    data.mocap_pos[mocap_id] = np.clip(new_pos, MOCAP_POS_MIN, MOCAP_POS_MAX)

    # ── 旋转增量 ──────────────────────────────────────────────
    # 右摇杆 X (ax3)  -> Yaw  (绕世界 Z 轴)
    d_yaw  = -(abs(ax3) > DEADZONE) * ax3 * rot_dt
    # 右摇杆 Y (ax4)  -> Roll (绕世界 X 轴)
    d_roll = -(abs(ax4) > DEADZONE) * ax4 * rot_dt
    # D-pad 左/右 (hat_x) -> Pitch (绕世界 Y 轴，对应原有旋转逻辑)
    d_pitch = hat_x * rot_dt * 1.5   # D-pad 幅度

    apply_rotation(mocap_id, 'z', d_yaw)
    apply_rotation(mocap_id, 'x', d_roll)
    apply_rotation(mocap_id, 'y', d_pitch)

    # ── 按键逻辑 ──────────────────────────────────────────────
    now = time.time()
    num_buttons = joystick.get_numbuttons()

    def safe_get_btn(btn_id):
        return joystick.get_button(btn_id) if num_buttons > btn_id else 0

    # X 键 (Xbox=2) -> 重置环境
    if safe_get_btn(2) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        reset_env()

    # B 键 (Xbox=1) -> 切换录制
    if safe_get_btn(1) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        record_toggle()

    # A 键 (Xbox=0) -> 夹爪开合切换
    if safe_get_btn(0) and (now - buttonCooldown > COOLDOWN_SEC):
        buttonCooldown = now
        gripper_open = not gripper_open
        set_gripper(gripper_open)

    # Start (Xbox=7 或 6) -> 退出
    if safe_get_btn(7) or safe_get_btn(6):
        return True

    return False

# --- Main loop ---
renderer = mujoco.Renderer(model, height=OBS_IMAGES_SHAPE[0], width=OBS_IMAGES_SHAPE[1])
reset_env()
video_time = 0.0
dataset = init_lerobot_dataset(repo_path=REPO_ID)

logger.info("Control ready: X=reset, B=toggle recording, A=gripper, Start=exit")
logger.info("  Left stick: XY move | LT/RT: Z up/down")
logger.info("  Right stick X: Yaw | Right stick Y: Roll | D-pad LR: Pitch")

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # IK simulation and physics stepping
            # act_ids 对应 actuator，qpos 里 joint 的索引不同，需要用 joint id
            joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in act_names]
            data.ctrl[:len(act_ids)] = [data.qpos[model.jnt_qposadr[jid]] for jid in joint_ids]
            if joystick_control():
                break
            mujoco.mj_step(model, data)

            # Check task success: 判断方块是否被抓取（末端与方块距离 < 5cm）
            ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp")]
            block_pos = data.xpos[block_body_id]
            dist = np.linalg.norm(ee_pos - block_pos)
            if dist < 0.05 and not is_success:
                logger.info(f"Task succeeded! EE-Block distance: {dist:.4f}m")
                is_success = True

            # Visual rendering and data sampling
            video_time += model.opt.timestep
            if video_time >= VIDEO_STEP:
                video_time = 0.0
                
                renderer.update_scene(data, camera="top_view")
                img_top = renderer.render().copy()
                renderer.update_scene(data, camera="side_view")
                img_side = renderer.render().copy()

                # OpenCV preview
                disp_img_top = cv2.cvtColor(img_top, cv2.COLOR_RGB2BGR).copy()
                disp_img_top = cv2.resize(disp_img_top, (OBS_IMAGES_SHAPE[1] * 2, OBS_IMAGES_SHAPE[0] * 2))
                disp_img_side = cv2.cvtColor(img_side, cv2.COLOR_RGB2BGR).copy()
                disp_img_side = cv2.resize(disp_img_side, (OBS_IMAGES_SHAPE[1] * 2, OBS_IMAGES_SHAPE[0] * 2))

                # Get current control target in both internal mocap frame and exported TCP frame.
                current_mocap_pose = get_mocap_4d_pose(
                    data.mocap_pos[mocap_id].copy(),
                    data.mocap_quat[mocap_id].copy(),
                    gripper_open
                )
                current_tcp_target = get_tcp_target_pose_from_mocap(gripper_open)
                current_tcp_state = get_current_tcp_state(gripper_open)

                if is_recording:
                    # Read joint angles using qposadr (not actuator IDs)
                    joint_ids_state = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in act_names]
                    joint_state = np.array(
                        [data.qpos[model.jnt_qposadr[jid]] for jid in joint_ids_state],
                        dtype=np.float32,
                    )
                    state = np.concatenate([joint_state, current_tcp_state]).astype(np.float32)
                    record_buffer.append({
                        "cam_top": img_top,
                        "cam_side": img_side,
                        "action": current_tcp_target,
                        "state": state
                    })

                    cv2.circle(disp_img_top, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(
                        disp_img_top, f"REC: {len(record_buffer)}", (50, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                    )
                
                info_texts = [
                    f"TCP target X: {current_tcp_target[0]:.3f}",
                    f"TCP target Y: {current_tcp_target[1]:.3f}",
                    f"TCP target Z: {current_tcp_target[2]:.3f}",
                ]

                for i, text in enumerate(info_texts):
                    cv2.putText(
                        disp_img_top, text, (20, 80 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )

                cv2.imshow("MuJoCo Nero - Top View", disp_img_top)
                cv2.imshow("MuJoCo Nero - Side View", disp_img_side)
                viewer.sync()
                cv2.waitKey(1)

            # Maintain physics rate
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)

finally:
    # Cleanup logic when the program exits
    logger.info("Shutting down safely...")

    if save_thread and save_thread.is_alive():
        logger.info("Waiting for the last batch of data to be saved...")
        save_thread.join()

    logger.info("Saving dataset index and releasing resources...")
    del dataset

    cv2.destroyAllWindows()
    pygame.quit()
    logger.info("Exit complete.")
