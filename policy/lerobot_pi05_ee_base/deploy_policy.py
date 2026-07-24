"""PI0.5 remote inference for 14D Nero xyz/rotvec pose actions."""

from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import time
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


LEROBOT_SRC = Path(
    os.environ.get(
        "ROBOTWIN_LEROBOT_SRC",
        Path(__file__).resolve().parents[2] / "third_party/lerobot_nero_runtime/src",
    )
)
if LEROBOT_SRC.exists():
    sys.path.insert(0, str(LEROBOT_SRC))

try:
    import grpc
except ModuleNotFoundError:
    grpc_site_packages = os.environ.get("ROBOTWIN_GRPC_SITE_PACKAGES")
    if not grpc_site_packages:
        raise
    sys.path.append(grpc_site_packages)
    import grpc
    sys.path.remove(grpc_site_packages)

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


@dataclass
class TimedObservation:
    timestamp: float
    timestep: int
    observation: dict
    must_go: bool = False


@dataclass
class TimedAction:
    timestamp: float
    timestep: int
    action: object

    def get_action(self):
        return self.action


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)


# Preserve the module names expected by the Chenglong async server's pickle wire format.
wire_module = types.ModuleType("lerobot.async_inference.helpers")
for wire_class in (TimedObservation, TimedAction, RemotePolicyConfig):
    wire_class.__module__ = wire_module.__name__
    setattr(wire_module, wire_class.__name__, wire_class)
sys.modules[wire_module.__name__] = wire_module


STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_NAMES = [
    "right_x",
    "right_y",
    "right_z",
    "right_rx",
    "right_ry",
    "right_rz",
    "right_gripper",
    "left_x",
    "left_y",
    "left_z",
    "left_rx",
    "left_ry",
    "left_rz",
    "left_gripper",
]
EE_ACTION_NAMES = [
    "right_ee_x",
    "right_ee_y",
    "right_ee_z",
    "right_ee_rotvec_x",
    "right_ee_rotvec_y",
    "right_ee_rotvec_z",
    "right_gripper_width",
    "left_ee_x",
    "left_ee_y",
    "left_ee_z",
    "left_ee_rotvec_x",
    "left_ee_rotvec_y",
    "left_ee_rotvec_z",
    "left_gripper_width",
]
ACTION_DIM = len(ACTION_NAMES)
CAMERA_TO_ROBOTWIN = {
    "observation.images.front": "head_camera",
    "observation.images.left_wrist": "left_camera",
    "observation.images.right_wrist": "right_camera",
}


def _feature_shape(feature: dict) -> tuple[int, ...]:
    return tuple(int(value) for value in feature.get("shape", ()))


def _postprocessor_decodes_relative(checkpoint: Path) -> bool:
    path = checkpoint / "policy_postprocessor.json"
    if not path.is_file():
        return False
    with path.open(encoding="utf-8") as processor_file:
        processor = json.load(processor_file)
    return any(
        step.get("registry_name") == "ee_so3_absolute_actions_processor"
        and step.get("config", {}).get("enabled", True)
        for step in processor.get("steps", ())
    )


def _load_checkpoint_spec(checkpoint: Path) -> tuple[dict[str, dict], bool]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"PI0.5 checkpoint config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if config.get("type") != "pi05":
        raise ValueError(f"Expected a pi05 checkpoint, got type={config.get('type')!r}")
    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    state_shape = _feature_shape(inputs.get(STATE_KEY, {}))
    action_shape = _feature_shape(outputs.get(ACTION_KEY, {}))
    if state_shape != (ACTION_DIM,) or action_shape != (ACTION_DIM,):
        raise ValueError(
            "The base-frame EE adapter requires 14D state/action features; "
            f"checkpoint has state={state_shape}, action={action_shape}."
        )
    if config.get("relative_action_type") != "ee_so3":
        raise ValueError(
            "Expected relative_action_type='ee_so3' for base-frame xyz/rotvec deltas, got "
            f"{config.get('relative_action_type')!r}."
        )
    if not config.get("use_relative_actions", False):
        raise ValueError("This adapter requires a checkpoint trained with use_relative_actions=true")
    configured_names = list(config.get("action_feature_names") or EE_ACTION_NAMES)
    if configured_names not in (ACTION_NAMES, EE_ACTION_NAMES):
        raise ValueError(
            "Checkpoint action order does not match the Nero EE adapter: "
            f"{configured_names!r}"
        )
    missing_cameras = sorted(set(CAMERA_TO_ROBOTWIN) - set(inputs))
    if missing_cameras:
        raise ValueError("Checkpoint is missing camera features: " + ", ".join(missing_cameras))

    features = {
        STATE_KEY: {"dtype": "float32", "shape": state_shape, "names": configured_names},
        ACTION_KEY: {"dtype": "float32", "shape": action_shape, "names": configured_names},
    }
    for camera_key in CAMERA_TO_ROBOTWIN:
        features[camera_key] = {
            "dtype": "image",
            "shape": _feature_shape(inputs[camera_key]),
            "names": ["height", "width", "channels"],
        }
    server_decodes_relative = _postprocessor_decodes_relative(checkpoint)
    if not server_decodes_relative:
        print(
            "[PI0.5 EE] Checkpoint has no ee_so3 absolute postprocessor; "
            "the client will decode each chunk relative to its observation state."
        )
    return features, server_decodes_relative


def _world_pose_to_xyz_rotvec(pose: Any) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"World EE pose must have shape (7,), got {pose.shape}")
    rotation = Rotation.from_quat(pose[3:7][[1, 2, 3, 0]])
    return np.concatenate((pose[:3], rotation.as_rotvec()))


def _robot_policy_ee_state(
    task_env: Any,
    gripper_max_width: float,
    ik: "NeroCuroboIK",
    pose_reference_frame: str,
) -> np.ndarray:
    robot = task_env.robot
    if pose_reference_frame == "robotwin_world_control":
        right_pose = _world_pose_to_xyz_rotvec(robot.get_right_ee_pose())
        left_pose = _world_pose_to_xyz_rotvec(robot.get_left_ee_pose())
    elif pose_reference_frame == "robot_base_link7":
        right_joints = np.asarray(robot.get_right_arm_real_jointState()[:7], dtype=np.float32)
        left_joints = np.asarray(robot.get_left_arm_real_jointState()[:7], dtype=np.float32)
        right_pose = ik.forward(right_joints)
        left_pose = ik.forward(left_joints)
    else:
        raise ValueError(f"Unsupported pose_reference_frame={pose_reference_frame!r}")
    return np.concatenate(
        (
            right_pose,
            [robot.get_right_gripper_val() * gripper_max_width],
            left_pose,
            [robot.get_left_gripper_val() * gripper_max_width],
        )
    ).astype(np.float32)


def _decode_ee_so3_chunk_action(delta: np.ndarray, base_state: np.ndarray) -> np.ndarray:
    """Decode one prediction relative to the fixed observation at chunk start."""
    delta = np.asarray(delta, dtype=np.float64)
    absolute = np.asarray(base_state, dtype=np.float64).copy()
    for offset in (0, 7):
        absolute[offset : offset + 3] += delta[offset : offset + 3]
        current_rotation = Rotation.from_rotvec(base_state[offset + 3 : offset + 6])
        delta_rotation = Rotation.from_rotvec(delta[offset + 3 : offset + 6])
        absolute[offset + 3 : offset + 6] = (delta_rotation * current_rotation).as_rotvec()
        absolute[offset + 6] = delta[offset + 6]
    return absolute.astype(np.float32)


def _limit_vector_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.asarray(target) - np.asarray(current)
    norm = float(np.linalg.norm(delta))
    if max_step <= 0 or norm <= max_step or norm == 0:
        return np.asarray(target, dtype=np.float64).copy()
    return np.asarray(current, dtype=np.float64) + delta * (max_step / norm)


def _limit_rotvec_step(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    if max_step <= 0:
        return np.asarray(target, dtype=np.float64).copy()
    current_rotation = Rotation.from_rotvec(current)
    target_rotation = Rotation.from_rotvec(target)
    delta_rotation = target_rotation * current_rotation.inv()
    delta_rotvec = delta_rotation.as_rotvec()
    angle = float(np.linalg.norm(delta_rotvec))
    if angle <= max_step or angle == 0:
        return np.asarray(target, dtype=np.float64).copy()
    limited_delta = Rotation.from_rotvec(delta_rotvec * (max_step / angle))
    return (limited_delta * current_rotation).as_rotvec()


def _limit_ee_action(
    target: np.ndarray,
    current: np.ndarray,
    max_position_step_m: float,
    max_rotation_step_rad: float,
) -> np.ndarray:
    limited = np.asarray(target, dtype=np.float64).copy()
    for offset in (0, 7):
        limited[offset : offset + 3] = _limit_vector_step(
            target[offset : offset + 3], current[offset : offset + 3], max_position_step_m
        )
        limited[offset + 3 : offset + 6] = _limit_rotvec_step(
            target[offset + 3 : offset + 6],
            current[offset + 3 : offset + 6],
            max_rotation_step_rad,
        )
    return limited.astype(np.float32)


def _apply_carry_lift(
    target: np.ndarray,
    lift_offset_m: float,
    start_z_m: float,
    full_z_m: float,
    closed_width_m: float,
    open_width_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    lifted = np.asarray(target, dtype=np.float64).copy()
    offsets = np.zeros(2, dtype=np.float32)
    if lift_offset_m <= 0:
        return lifted.astype(np.float32), offsets

    for arm_index, offset in enumerate((0, 7)):
        height_factor = np.clip(
            (lifted[offset + 2] - start_z_m) / (full_z_m - start_z_m), 0.0, 1.0
        )
        gripper_factor = np.clip(
            (open_width_m - lifted[offset + 6]) / (open_width_m - closed_width_m),
            0.0,
            1.0,
        )
        z_offset = float(lift_offset_m * height_factor * gripper_factor)
        lifted[offset + 2] += z_offset
        offsets[arm_index] = z_offset
    return lifted.astype(np.float32), offsets


class NeroCuroboIK:
    """One shared Nero IK solver; both arms use identical base-frame kinematics."""

    def __init__(
        self,
        config_path: Path | str,
        num_seeds: int,
        return_seeds: int,
        position_threshold: float,
        rotation_threshold: float,
        ee_link: str,
    ) -> None:
        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose as CuroboPose
        from curobo.util_file import load_yaml
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        import yaml

        config_path = Path(config_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"cuRobo config does not exist: {config_path}")
        self.torch = torch
        self.CuroboPose = CuroboPose
        self.tensor_args = TensorDeviceType()
        config_text = config_path.read_text(encoding="utf-8")
        if "${ASSETS_PATH}" in config_text:
            repo_root = Path(
                os.environ.get("ROBOTWIN_ROOT", str(config_path.parents[3]))
            ).resolve()
            config_text = config_text.replace("${ASSETS_PATH}", str(repo_root))
            robot_yaml = yaml.safe_load(config_text)
        else:
            robot_yaml = load_yaml(str(config_path))
        robot_config = copy.deepcopy(robot_yaml.get("robot_cfg", robot_yaml))
        solver_config = IKSolverConfig.load_from_robot_config(
            robot_config,
            None,
            tensor_args=self.tensor_args,
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            self_collision_check=False,
            self_collision_opt=False,
            use_cuda_graph=False,
            collision_checker_type=None,
            ee_link_name=ee_link,
        )
        self.solver = IKSolver(solver_config)
        self.return_seeds = min(max(1, return_seeds), num_seeds)
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.ee_link = ee_link
        print(
            f"[PI0.5 EE] cuRobo IK ready: ee_link={ee_link}, seeds={num_seeds}, "
            f"return_seeds={self.return_seeds}"
        )

    def forward(self, joints: np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=np.float32)
        if joints.shape != (7,):
            raise ValueError(f"{self.ee_link} FK joints must have shape (7,), got {joints.shape}")
        q = self.torch.as_tensor(
            joints, device=self.tensor_args.device, dtype=self.torch.float32
        ).reshape(1, -1)
        state = self.solver.kinematics.get_state(q)
        position = state.ee_position.detach().cpu().numpy().reshape(-1, 3)[0]
        quaternion_wxyz = state.ee_quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        rotation = Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]])
        return np.concatenate((position, rotation.as_rotvec())).astype(np.float32)

    def solve(self, pose_xyz_rotvec: np.ndarray, current_joints: np.ndarray, arm: str) -> np.ndarray:
        target = np.asarray(pose_xyz_rotvec, dtype=np.float64)
        current = np.asarray(current_joints, dtype=np.float64)
        if target.shape != (6,) or current.shape != (7,):
            raise ValueError(f"Invalid {arm} IK input shapes: target={target.shape}, seed={current.shape}")

        quat_xyzw = Rotation.from_rotvec(target[3:6]).as_quat()
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        goal = self.CuroboPose.from_list(
            np.concatenate((target[:3], quat_wxyz)).tolist(), tensor_args=self.tensor_args
        )
        seed = self.torch.as_tensor(
            current, device=self.tensor_args.device, dtype=self.torch.float32
        ).reshape(1, -1)
        result = self.solver.solve_single(
            goal,
            retract_config=seed,
            seed_config=seed.reshape(1, 1, -1),
            return_seeds=self.return_seeds,
        )
        solutions = result.solution.detach().cpu().numpy().reshape(-1, 7)
        success = result.success.detach().cpu().numpy().reshape(-1).astype(bool)
        valid = np.flatnonzero(success)
        if len(valid) == 0:
            position_error = float(result.position_error.detach().cpu().numpy().reshape(-1)[0])
            rotation_error = float(result.rotation_error.detach().cpu().numpy().reshape(-1)[0])
            raise RuntimeError(
                f"{arm} {self.ee_link} IK failed for target={target.tolist()}; "
                f"position_error={position_error:.6f}, rotation_error={rotation_error:.6f}"
            )
        nearest = valid[np.argmin(np.linalg.norm(solutions[valid] - current, axis=1))]
        return np.asarray(solutions[nearest], dtype=np.float32)


def _world_control_target_to_base_ik(robot: Any, target: np.ndarray, arm: str) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    quaternion_xyzw = Rotation.from_rotvec(target[3:6]).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    control_pose = np.concatenate((target[:3], quaternion_wxyz))
    endlink_pose = robot._trans_from_gripper_to_endlink(control_pose, arm_tag=arm)
    base_pose = (
        robot.right_entity_origion_pose if arm == "right" else robot.left_entity_origion_pose
    )
    world_from_base = Rotation.from_quat(np.asarray(base_pose.q)[[1, 2, 3, 0]])
    world_from_endlink = Rotation.from_quat(np.asarray(endlink_pose.q)[[1, 2, 3, 0]])
    base_position = world_from_base.inv().apply(
        np.asarray(endlink_pose.p, dtype=np.float64) - np.asarray(base_pose.p, dtype=np.float64)
    )
    base_rotation = world_from_base.inv() * world_from_endlink
    return np.concatenate((base_position, base_rotation.as_rotvec())).astype(np.float32)


class RemotePI05EEBase:
    def __init__(self, usr_args: dict):
        self.checkpoint = Path(usr_args["policy_path"]).expanduser().resolve()
        self.server_address = str(usr_args.get("server_address", "127.0.0.1:8081"))
        self.policy_device = str(usr_args.get("policy_device", "cuda"))
        self.actions_per_chunk = int(usr_args.get("actions_per_chunk", 50))
        self.chunk_size_threshold = float(usr_args.get("chunk_size_threshold", 0.8))
        self.aggregate_fn_name = str(usr_args.get("aggregate_fn_name", "average"))
        self.action_merge_new_weight = float(usr_args.get("action_merge_new_weight", 0.75))
        self.fps = int(usr_args.get("fps", 30))
        self.gripper_max_width = float(usr_args.get("gripper_max_width", 0.1))
        self.pose_reference_frame = str(
            usr_args.get("pose_reference_frame", "robotwin_world_control")
        )
        print(f"[PI0.5 EE] Policy pose reference frame: {self.pose_reference_frame}")
        self.max_ee_position_step_m = float(usr_args.get("max_ee_position_step_m", 0.04))
        self.max_ee_rotation_step_rad = float(usr_args.get("max_ee_rotation_step_rad", 0.15))
        self.carry_lift_offset_m = float(usr_args.get("carry_lift_offset_m", 0.0))
        self.carry_lift_start_z_m = float(usr_args.get("carry_lift_start_z_m", 0.89))
        self.carry_lift_full_z_m = float(usr_args.get("carry_lift_full_z_m", 0.94))
        self.carry_lift_closed_width_m = float(
            usr_args.get("carry_lift_closed_width_m", 0.02)
        )
        self.carry_lift_open_width_m = float(
            usr_args.get("carry_lift_open_width_m", 0.08)
        )
        self.curobo_config = Path(usr_args["curobo_config"]).expanduser().resolve()
        self.curobo_ee_link = str(usr_args.get("curobo_ee_link", "gripper_tcp"))
        self.curobo_num_seeds = int(usr_args.get("curobo_num_seeds", 64))
        self.curobo_return_seeds = int(usr_args.get("curobo_return_seeds", 8))
        self.curobo_position_threshold = float(usr_args.get("curobo_position_threshold", 0.05))
        self.curobo_rotation_threshold = float(usr_args.get("curobo_rotation_threshold", 0.08))
        self.exact_policy_tracking = bool(usr_args.get("exact_policy_tracking", True))
        self.max_policy_step_rad = float(usr_args.get("max_policy_step_rad", 0.05))
        self.max_gripper_step_m = float(usr_args.get("max_gripper_step_m", 0.05))
        self.max_executor_step_rad = float(usr_args.get("max_executor_step_rad", 0.005))
        self.max_executor_gripper_step_m = float(
            usr_args.get("max_executor_gripper_step_m", 0.004)
        )
        self.trace_enabled = bool(usr_args.get("trace_enabled", True))
        self.trace_dir = Path(usr_args.get("trace_dir", "outputs/pi05_ee_base_inference_traces"))
        self.trace_path: Path | None = None
        if self.trace_enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = self.trace_dir / f"trace_{int(time.time())}.jsonl"

        self.features, self.server_decodes_relative = _load_checkpoint_spec(self.checkpoint)
        self.state_names = self.features[STATE_KEY]["names"]
        if self.gripper_max_width <= 0:
            raise ValueError("gripper_max_width must be positive")
        if not 0 <= self.chunk_size_threshold < 1:
            raise ValueError("chunk_size_threshold must be in [0, 1)")
        if self.aggregate_fn_name != "average":
            raise ValueError("Only aggregate_fn_name=average is supported")
        if not 0.0 <= self.action_merge_new_weight <= 1.0:
            raise ValueError("action_merge_new_weight must be in [0, 1]")
        print(
            f"[PI0.5 EE] Overlapping action merge: old="
            f"{1.0 - self.action_merge_new_weight:.2f}, "
            f"new={self.action_merge_new_weight:.2f}"
        )
        if self.carry_lift_offset_m < 0:
            raise ValueError("carry_lift_offset_m must be non-negative")
        if self.carry_lift_full_z_m <= self.carry_lift_start_z_m:
            raise ValueError("carry_lift_full_z_m must be greater than carry_lift_start_z_m")
        if self.carry_lift_open_width_m <= self.carry_lift_closed_width_m:
            raise ValueError(
                "carry_lift_open_width_m must be greater than carry_lift_closed_width_m"
            )
        if self.carry_lift_offset_m > 0:
            print(
                f"[PI0.5 EE] Carry lift enabled: +{self.carry_lift_offset_m:.3f} m "
                f"between z={self.carry_lift_start_z_m:.3f} and "
                f"z={self.carry_lift_full_z_m:.3f}"
            )

        self.replan_stride = max(
            1, int(round(self.actions_per_chunk * (1.0 - self.chunk_size_threshold)))
        )
        self.action_queue: OrderedDict[int, np.ndarray] = OrderedDict()
        self.latest_action_timestep = -1
        self.timestep = 0
        self.ik: NeroCuroboIK | None = None

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff="0.0333s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self._handshake()

    def _handshake(self) -> None:
        try:
            grpc.channel_ready_future(self.channel).result(timeout=10)
            self.stub.Ready(services_pb2.Empty(), timeout=10)
            policy_config = RemotePolicyConfig(
                policy_type="pi05",
                pretrained_name_or_path=str(self.checkpoint),
                lerobot_features=self.features,
                actions_per_chunk=self.actions_per_chunk,
                device=self.policy_device,
            )
            request = services_pb2.PolicySetup(data=pickle.dumps(policy_config))
            self.stub.SendPolicyInstructions(request, timeout=300)
        except grpc.RpcError as error:
            raise RuntimeError(
                f"Could not initialize PI0.5 server at {self.server_address}: {error}"
            ) from error

    def _ensure_ik(self) -> NeroCuroboIK:
        if self.ik is None:
            self.ik = NeroCuroboIK(
                self.curobo_config,
                self.curobo_num_seeds,
                self.curobo_return_seeds,
                self.curobo_position_threshold,
                self.curobo_rotation_threshold,
                self.curobo_ee_link,
            )
        return self.ik

    def reset(self) -> None:
        self.timestep = 0
        self.latest_action_timestep = -1
        self.action_queue.clear()
        try:
            self.stub.Ready(services_pb2.Empty(), timeout=10)
        except grpc.RpcError as error:
            raise RuntimeError(
                f"Could not reset PI0.5 episode at {self.server_address}: {error}"
            ) from error

    def infer(self, observation: dict, task: str, base_state: np.ndarray) -> list[tuple[int, np.ndarray]]:
        raw_observation = {
            name: float(base_state[index]) for index, name in enumerate(self.state_names)
        }
        for feature_key, robotwin_key in CAMERA_TO_ROBOTWIN.items():
            raw_key = feature_key.removeprefix("observation.images.")
            raw_observation[raw_key] = np.asarray(
                observation["observation"][robotwin_key]["rgb"], dtype=np.uint8
            )
        raw_observation["task"] = task
        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=self.timestep,
            observation=raw_observation,
            must_go=True,
        )
        payload = pickle.dumps(timed_observation)
        request_iterator = send_bytes_in_chunks(
            payload, services_pb2.Observation, log_prefix="[RoboTwin EE] Observation", silent=True
        )
        self.stub.SendObservations(request_iterator, timeout=30)
        inference_start = time.perf_counter()
        response = self.stub.GetActions(services_pb2.Empty(), timeout=300)
        inference_elapsed_s = time.perf_counter() - inference_start
        if not response.data:
            raise RuntimeError("PI0.5 server returned an empty EE action chunk")

        timed_actions = pickle.loads(response.data)
        actions = []
        raw_actions = []
        for item in timed_actions:
            predicted = np.asarray(item.get_action().numpy(), dtype=np.float32)
            raw_actions.append(predicted.tolist())
            absolute = (
                predicted
                if self.server_decodes_relative
                else _decode_ee_so3_chunk_action(predicted, base_state)
            )
            actions.append((int(item.timestep), absolute.astype(np.float32)))
        self._trace(
            {
                "event": "inference",
                "timestep": self.timestep,
                "task": task,
                "inference_elapsed_s": inference_elapsed_s,
                "base_ee_state": base_state.tolist(),
                "server_decodes_relative": self.server_decodes_relative,
                "action_merge_new_weight": self.action_merge_new_weight,
                "server_actions": raw_actions,
            }
        )
        return actions

    def refresh_action_queue(self, observation: dict, task: str, base_state: np.ndarray) -> None:
        self.timestep = self.latest_action_timestep + 1
        incoming_actions = self.infer(observation, task, base_state)
        current = dict(self.action_queue)
        merged: OrderedDict[int, np.ndarray] = OrderedDict()
        for timestep, new_action in incoming_actions:
            if timestep <= self.latest_action_timestep:
                continue
            old_action = current.get(timestep)
            merged[timestep] = (
                new_action
                if old_action is None
                else (
                    (1.0 - self.action_merge_new_weight) * old_action
                    + self.action_merge_new_weight * new_action
                )
            ).astype(np.float32)
        if not merged:
            raise RuntimeError("PI0.5 EE replanning produced no future actions")
        self.action_queue = merged

    def pop_actions_until_replan(self) -> list[np.ndarray]:
        action_count = min(self.replan_stride, len(self.action_queue))
        actions = []
        for _ in range(action_count):
            timestep, action = self.action_queue.popitem(last=False)
            self.latest_action_timestep = timestep
            actions.append(action)
        return actions

    def ee_target_to_joint_action(self, task_env: Any, target: np.ndarray) -> np.ndarray:
        ik = self._ensure_ik()
        current_ee = _robot_policy_ee_state(
            task_env, self.gripper_max_width, ik, self.pose_reference_frame
        )
        carry_lifted, carry_lift_offsets = _apply_carry_lift(
            target,
            self.carry_lift_offset_m,
            self.carry_lift_start_z_m,
            self.carry_lift_full_z_m,
            self.carry_lift_closed_width_m,
            self.carry_lift_open_width_m,
        )
        limited = _limit_ee_action(
            carry_lifted,
            current_ee,
            self.max_ee_position_step_m,
            self.max_ee_rotation_step_rad,
        )
        robot = task_env.robot
        right_current = np.asarray(robot.get_right_arm_real_jointState()[:7], dtype=np.float32)
        left_current = np.asarray(robot.get_left_arm_real_jointState()[:7], dtype=np.float32)
        if self.pose_reference_frame == "robotwin_world_control":
            right_ik_target = _world_control_target_to_base_ik(robot, limited[0:6], "right")
            left_ik_target = _world_control_target_to_base_ik(robot, limited[7:13], "left")
        else:
            right_ik_target = limited[0:6]
            left_ik_target = limited[7:13]
        ik_start = time.perf_counter()
        right_joints = ik.solve(right_ik_target, right_current, "right")
        left_joints = ik.solve(left_ik_target, left_current, "left")
        ik_elapsed_s = time.perf_counter() - ik_start
        right_gripper = float(np.clip(limited[6] / self.gripper_max_width, 0.0, 1.0))
        left_gripper = float(np.clip(limited[13] / self.gripper_max_width, 0.0, 1.0))
        joint_action = np.concatenate(
            (left_joints, [left_gripper], right_joints, [right_gripper])
        ).astype(np.float32)
        self._trace(
            {
                "event": "ik",
                "policy_timestep": self.latest_action_timestep,
                "current_ee": current_ee.tolist(),
                "raw_ee_target": np.asarray(target).tolist(),
                "carry_lifted_ee_target": carry_lifted.tolist(),
                "carry_lift_offsets_m": carry_lift_offsets.tolist(),
                "limited_ee_target": limited.tolist(),
                "right_base_ik_target": right_ik_target.tolist(),
                "left_base_ik_target": left_ik_target.tolist(),
                "ik_elapsed_s": ik_elapsed_s,
                "joint_action": joint_action.tolist(),
            }
        )
        return joint_action

    def _trace(self, record: dict) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def get_model(usr_args: dict) -> RemotePI05EEBase:
    return RemotePI05EEBase(usr_args)


def eval(task_env: Any, model: RemotePI05EEBase, observation: dict) -> None:
    base_state = _robot_policy_ee_state(
        task_env,
        model.gripper_max_width,
        model._ensure_ik(),
        model.pose_reference_frame,
    )
    model.refresh_action_queue(observation, task_env.get_instruction(), base_state)
    for ee_target in model.pop_actions_until_replan():
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
            break
        joint_action = model.ee_target_to_joint_action(task_env, ee_target)
        task_env.take_policy_action(
            joint_action,
            fps=model.fps,
            exact_policy_tracking=model.exact_policy_tracking,
            max_policy_step_rad=model.max_policy_step_rad,
            max_gripper_step=model.max_gripper_step_m / model.gripper_max_width,
            max_executor_step_rad=model.max_executor_step_rad,
            max_executor_gripper_step=(
                model.max_executor_gripper_step_m / model.gripper_max_width
            ),
        )


def reset_model(model: RemotePI05EEBase) -> None:
    model.reset()
