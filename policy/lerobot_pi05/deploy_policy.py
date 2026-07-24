"""LeRobot PI0.5 remote-inference adapter for the RoboTwin evaluator."""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
import types
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


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
    # Both environments use Python 3.10; only pushT-pi05 currently has the
    # optional LeRobot async-inference dependency installed.
    grpc_site_packages = os.environ.get("ROBOTWIN_GRPC_SITE_PACKAGES")
    if not grpc_site_packages:
        raise
    sys.path.append(grpc_site_packages)
    import grpc
    sys.path.remove(grpc_site_packages)

from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks


# These wire objects mirror lerobot.async_inference.helpers. Keeping their
# original module identity lets the policy server unpickle them as its native
# classes, without importing all policy and dataset dependencies in RoboTwin.
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
    record_dir: str = ""
    feature_adapter_enabled: bool | None = None
    inference_seed: int | None = None


wire_module = types.ModuleType("lerobot.async_inference.helpers")
for wire_class in (TimedObservation, TimedAction, RemotePolicyConfig):
    wire_class.__module__ = wire_module.__name__
    setattr(wire_module, wire_class.__name__, wire_class)
sys.modules[wire_module.__name__] = wire_module


STATE_KEY = "observation.state"
ACTION_KEY = "action"
EXPECTED_STATE_DIM = 16
EXPECTED_ACTION_DIM = 16

CAMERA_TO_ROBOTWIN = {
    "observation.images.front": "head_camera",
    "observation.images.left_wrist": "left_camera",
    "observation.images.right_wrist": "right_camera",
}


def _parse_feature_adapter_enabled(value: object | None) -> bool | None:
    if value is None:
        value = os.environ.get("ROBOTWIN_PI05_FEATURE_ADAPTER", "auto")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"", "auto"}:
        return None
    if normalized in {"1", "true", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "off", "disable", "disabled"}:
        return False
    raise ValueError(
        "feature_adapter_enabled must be true, false, or auto "
        f"(got {value!r})"
    )


def _feature_shape(feature: dict) -> tuple[int, ...]:
    return tuple(int(value) for value in feature["shape"])


def _load_and_validate_features(checkpoint: Path) -> dict[str, dict]:
    config_path = checkpoint / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"PI0.5 checkpoint config does not exist: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if config.get("type") != "pi05":
        raise ValueError(f"Expected a pi05 checkpoint, got type={config.get('type')!r}")

    inputs = config.get("input_features", {})
    outputs = config.get("output_features", {})
    state_shape = _feature_shape(inputs.get(STATE_KEY, {"shape": []}))
    action_shape = _feature_shape(outputs.get(ACTION_KEY, {"shape": []}))
    if state_shape != (EXPECTED_STATE_DIM,) or action_shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(
            "This RoboTwin Nero adapter requires a 16D state and 16D action checkpoint; "
            f"checkpoint has state={state_shape}, action={action_shape}. Train PI0.5 on "
            "data/place_two_cubes_box_lerobot_v3 or select a matching checkpoint."
        )

    missing_cameras = sorted(set(CAMERA_TO_ROBOTWIN) - set(inputs))
    if missing_cameras:
        raise ValueError(
            "Checkpoint camera features do not match the RoboTwin dataset; missing "
            + ", ".join(missing_cameras)
        )

    features = {
        STATE_KEY: {
            "dtype": "float32",
            "shape": state_shape,
            "names": [f"state_{index}" for index in range(EXPECTED_STATE_DIM)],
        },
        ACTION_KEY: {
            "dtype": "float32",
            "shape": action_shape,
            "names": [f"action_{index}" for index in range(EXPECTED_ACTION_DIM)],
        },
    }
    for camera_key in CAMERA_TO_ROBOTWIN:
        features[camera_key] = {
            "dtype": "image",
            "shape": _feature_shape(inputs[camera_key]),
            "names": ["height", "width", "channels"],
        }
    return features


def _robotwin_state_to_bus(state: np.ndarray, gripper_max_width: float) -> np.ndarray:
    """Convert left7,left-gripper,right7,right-gripper to training bus order."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (EXPECTED_STATE_DIM,):
        raise ValueError(f"Expected RoboTwin state shape (16,), got {state.shape}")
    left_arm, left_gripper = state[:7], state[7]
    right_arm, right_gripper = state[8:15], state[15]
    return np.concatenate(
        (
            right_arm,
            left_arm,
            np.array([right_gripper, left_gripper], dtype=np.float32) * gripper_max_width,
        )
    ).astype(np.float32)


def _bus_action_to_robotwin(action: np.ndarray, gripper_max_width: float) -> np.ndarray:
    """Convert training bus order to RoboTwin left7,left-gripper,right7,right-gripper."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"Expected policy action shape (16,), got {action.shape}")
    right_arm, left_arm = action[:7], action[7:14]
    right_gripper, left_gripper = np.clip(action[14:16] / gripper_max_width, 0.0, 1.0)
    return np.concatenate(
        (left_arm, [left_gripper], right_arm, [right_gripper])
    ).astype(np.float32)


class RemotePI05:
    def __init__(self, usr_args: dict):
        self.checkpoint = Path(usr_args["policy_path"]).expanduser().resolve()
        self.server_address = str(usr_args.get("server_address", "127.0.0.1:8081"))
        self.policy_device = str(usr_args.get("policy_device", "cuda"))
        self.feature_adapter_enabled = _parse_feature_adapter_enabled(
            usr_args.get("feature_adapter_enabled")
        )
        self.actions_per_chunk = int(usr_args.get("actions_per_chunk", 50))
        self.chunk_size_threshold = float(usr_args.get("chunk_size_threshold", 0.8))
        self.aggregate_fn_name = str(usr_args.get("aggregate_fn_name", "average"))
        self.fps = int(usr_args.get("fps", 30))
        self.gripper_max_width = float(usr_args.get("gripper_max_width", 0.1))
        self.use_relative_actions = bool(usr_args.get("use_relative_actions", True))
        self.direct_sim_control = bool(usr_args.get("direct_sim_control", True))
        self.exact_policy_tracking = bool(usr_args.get("exact_policy_tracking", True))
        self.max_policy_step_rad = float(usr_args.get("max_policy_step_rad", 0.05))
        self.max_gripper_step_m = float(usr_args.get("max_gripper_step_m", 0.05))
        self.max_executor_step_rad = float(usr_args.get("max_executor_step_rad", 0.005))
        self.max_executor_gripper_step_m = float(
            usr_args.get("max_executor_gripper_step_m", 0.004)
        )
        self.trace_enabled = bool(usr_args.get("trace_enabled", False))
        inference_seed = usr_args.get("policy_inference_seed")
        self.policy_inference_seed = int(inference_seed) if inference_seed is not None else None
        self.trace_dir = Path(
            usr_args.get("trace_dir", "outputs/pi05_inference_traces")
        )
        self.trace_path = None
        if self.trace_enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.trace_path = self.trace_dir / f"trace_{int(time.time())}.jsonl"
        self.features = _load_and_validate_features(self.checkpoint)
        if self.gripper_max_width <= 0:
            raise ValueError("gripper_max_width must be positive")
        if not 0 <= self.chunk_size_threshold < 1:
            raise ValueError("chunk_size_threshold must be in [0, 1)")
        if self.aggregate_fn_name != "average":
            raise ValueError("The RoboTwin PI0.5 adapter currently supports aggregate_fn_name=average")
        if self.fps <= 0:
            raise ValueError("fps must be positive")

        self.replan_stride = max(
            1,
            int(round(self.actions_per_chunk * (1.0 - self.chunk_size_threshold))),
        )
        self.action_queue: OrderedDict[int, np.ndarray] = OrderedDict()
        self.latest_action_timestep = -1

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff="0.0333s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.timestep = 0
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
                feature_adapter_enabled=self.feature_adapter_enabled,
                inference_seed=self.policy_inference_seed,
            )
            request = services_pb2.PolicySetup(data=pickle.dumps(policy_config))
            self.stub.SendPolicyInstructions(request, timeout=300)
        except grpc.RpcError as error:
            raise RuntimeError(
                f"Could not initialize PI0.5 server at {self.server_address}: {error}"
            ) from error

    def reset(self) -> None:
        """Reset one rollout without reloading the policy weights on the server."""
        self.timestep = 0
        self.latest_action_timestep = -1
        self.action_queue.clear()
        try:
            # Ready flushes server-side observation/action state and calls policy.reset().
            # Re-sending PolicySetup here would reload the PI0.5 checkpoint every rollout.
            self.stub.Ready(services_pb2.Empty(), timeout=10)
        except grpc.RpcError as error:
            raise RuntimeError(
                f"Could not reset PI0.5 episode at {self.server_address}: {error}"
            ) from error

    def infer(self, observation: dict, task: str) -> list[tuple[int, np.ndarray]]:
        raw_observation = {
            f"state_{index}": value
            for index, value in enumerate(
                _robotwin_state_to_bus(
                    observation["joint_action"]["vector"], self.gripper_max_width
                )
            )
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
            payload, services_pb2.Observation, log_prefix="[RoboTwin] Observation", silent=True
        )
        self.stub.SendObservations(request_iterator, timeout=30)
        inference_start = time.perf_counter()
        response = self.stub.GetActions(services_pb2.Empty(), timeout=300)
        inference_elapsed_s = time.perf_counter() - inference_start
        if not response.data:
            raise RuntimeError("PI0.5 server returned an empty action chunk; check the server log")

        timed_actions = pickle.loads(response.data)
        base_bus_state = _robotwin_state_to_bus(
            observation["joint_action"]["vector"], self.gripper_max_width
        )
        actions = []
        for item in timed_actions:
            predicted_action = item.get_action().numpy()
            if self.use_relative_actions:
                # Each action in the 50-step chunk is relative to the state
                # that produced the chunk, not to the preceding prediction.
                bus_action = base_bus_state.copy()
                bus_action[:14] += predicted_action[:14]
                bus_action[14:16] = predicted_action[14:16]
            else:
                bus_action = predicted_action
            actions.append(
                (
                    int(item.timestep),
                    _bus_action_to_robotwin(bus_action, self.gripper_max_width),
                )
            )
        if self.trace_path is not None:
            with self.trace_path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(
                    json.dumps(
                        {
                            "timestep": self.timestep,
                            "task": task,
                            "inference_elapsed_s": inference_elapsed_s,
                            "base_bus_state": base_bus_state.tolist(),
                            "relative_actions": [item.get_action().numpy().tolist() for item in timed_actions],
                            "robotwin_targets": [action.tolist() for _, action in actions],
                        }
                    )
                    + "\n"
                )
        return actions

    def refresh_action_queue(self, observation: dict, task: str) -> None:
        """Replan from the latest observation and merge overlapping future actions."""
        self.timestep = self.latest_action_timestep + 1
        incoming_actions = self.infer(observation, task)
        merged: OrderedDict[int, np.ndarray] = OrderedDict()

        current = dict(self.action_queue)
        for timestep, new_action in incoming_actions:
            if timestep <= self.latest_action_timestep:
                continue
            old_action = current.get(timestep)
            merged[timestep] = (
                new_action
                if old_action is None
                else 0.5 * old_action + 0.5 * new_action
            ).astype(np.float32)

        if not merged:
            raise RuntimeError("PI0.5 replanning produced no future actions")
        self.action_queue = merged

    def pop_actions_until_replan(self) -> list[np.ndarray]:
        action_count = min(self.replan_stride, len(self.action_queue))
        actions = []
        for _ in range(action_count):
            timestep, action = self.action_queue.popitem(last=False)
            self.latest_action_timestep = timestep
            actions.append(action)
        return actions


def get_model(usr_args: dict) -> RemotePI05:
    return RemotePI05(usr_args)


def eval(task_env, model: RemotePI05, observation: dict) -> None:
    model.refresh_action_queue(observation, task_env.get_instruction())
    for action in model.pop_actions_until_replan():
        if task_env.take_action_cnt >= task_env.step_lim or task_env.eval_success:
            break
        if model.direct_sim_control:
            task_env.take_policy_action(
                action,
                fps=model.fps,
                exact_policy_tracking=model.exact_policy_tracking,
                max_policy_step_rad=model.max_policy_step_rad,
                max_gripper_step=model.max_gripper_step_m / model.gripper_max_width,
                max_executor_step_rad=model.max_executor_step_rad,
                max_executor_gripper_step=(
                    model.max_executor_gripper_step_m / model.gripper_max_width
                ),
            )
        else:
            task_env.take_action(action)


def reset_model(model: RemotePI05) -> None:
    model.reset()
