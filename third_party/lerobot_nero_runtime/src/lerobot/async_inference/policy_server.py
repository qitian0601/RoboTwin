# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import json
import math
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import LatencyTracker, reanchor_relative_rtc_prefix
from lerobot.processor import NormalizerProcessorStep, PolicyProcessorPipeline, RelativeActionsProcessorStep
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES
from .helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.inference_seed: int | None = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None
        self._rtc_latency_tracker = LatencyTracker()
        self._rtc_original_actions: torch.Tensor | None = None
        self._rtc_processed_actions: torch.Tensor | None = None
        self._rtc_chunk_start_timestep: int | None = None
        self._rtc_relative_step: RelativeActionsProcessorStep | None = None
        self._rtc_normalizer_step: NormalizerProcessorStep | None = None
        self._rtc_debug_dump_path: Path | None = None
        self._rtc_debug_dump_lock = threading.Lock()

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)
        self.last_processed_obs = None

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        self._reset_rtc_state()
        reset_policy = getattr(self.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()

    def _reset_rtc_state(self) -> None:
        """Clear async RTC chunk state while preserving loaded processor references."""
        self._rtc_original_actions = None
        self._rtc_processed_actions = None
        self._rtc_chunk_start_timestep = None
        self._rtc_latency_tracker.reset()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        if self.inference_seed is not None:
            torch.manual_seed(self.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.inference_seed)
            self.logger.info("Reset policy inference RNG to seed %s", self.inference_seed)
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self.inference_seed = getattr(policy_specs, "inference_seed", None)
        self._configure_rtc_debug_dump(policy_specs.record_dir)

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        feature_adapter_enabled = getattr(policy_specs, "feature_adapter_enabled", None)
        if feature_adapter_enabled is not None and getattr(self.policy.config, "use_feature_adapter", False):
            self.policy.set_feature_adapter_enabled(feature_adapter_enabled)
            self.logger.info(
                "Feature Adapter inference is %s",
                "enabled" if feature_adapter_enabled else "bypassed",
            )
        self._enable_policy_rtc_debug_dump()
        self.policy.to(self.device)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        self._refresh_rtc_processor_steps()
        self._reset_rtc_state()
        self._log_async_rtc_status()

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        if not self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        ):
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(
            obs,
            previous_obs,
            lerobot_features=self.lerobot_features,
            atol=self.config.obs_similarity_atol,
        ):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _refresh_rtc_processor_steps(self) -> None:
        """Find processor steps needed to re-anchor relative-action RTC prefixes."""
        steps = getattr(self.preprocessor, "steps", ())
        self._rtc_relative_step = next(
            (step for step in steps if isinstance(step, RelativeActionsProcessorStep) and step.enabled),
            None,
        )
        self._rtc_normalizer_step = next(
            (step for step in steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )
        if self._rtc_relative_step is not None and self._rtc_relative_step.action_names is None:
            action_names = self._policy_action_names()
            if action_names:
                self._rtc_relative_step.action_names = action_names

    def _configure_rtc_debug_dump(self, record_dir: str) -> None:
        self._rtc_debug_dump_path = None
        if not self.config.async_rtc.debug_dump.enabled or not record_dir:
            return

        record_path = Path(record_dir).expanduser()
        record_path.mkdir(parents=True, exist_ok=True)
        self._rtc_debug_dump_path = record_path / "rtc_debug.jsonl"
        self._rtc_debug_dump_path.write_text("", encoding="utf-8")
        self.logger.info("Writing async RTC debug dump to %s", self._rtc_debug_dump_path)

    def _enable_policy_rtc_debug_dump(self) -> None:
        if self._rtc_debug_dump_path is None:
            return

        policy_config = getattr(self.policy, "config", None)
        rtc_config = getattr(policy_config, "rtc_config", None)
        if rtc_config is None:
            self.logger.warning("RTC debug dump requested, but loaded policy has no rtc_config")
            return

        rtc_config.debug = True
        rtc_config.debug_maxlen = max(int(getattr(rtc_config, "debug_maxlen", 100)), 100)
        init_rtc_processor = getattr(self.policy, "init_rtc_processor", None)
        if callable(init_rtc_processor):
            init_rtc_processor()

    @staticmethod
    def _tensor_stats(tensor: torch.Tensor | None) -> dict[str, Any] | None:
        if tensor is None:
            return None

        tensor_cpu = tensor.detach().float().cpu()
        return {
            "shape": list(tensor_cpu.shape),
            "mean": float(tensor_cpu.mean().item()),
            "std": float(tensor_cpu.std().item()) if tensor_cpu.numel() > 1 else 0.0,
            "min": float(tensor_cpu.min().item()),
            "max": float(tensor_cpu.max().item()),
            "norm": float(torch.linalg.norm(tensor_cpu).item()),
        }

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return PolicyServer._tensor_stats(value)
        return value

    def _rtc_debug_summary(
        self,
        *,
        observation_timestep: int,
        chunk_start_timestep: int,
        chunk_len: int,
        total_latency_s: float,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
    ) -> dict[str, Any] | None:
        rtc_processor = getattr(self.policy, "rtc_processor", None)
        get_steps = getattr(rtc_processor, "get_all_debug_steps", None)
        if not callable(get_steps):
            return None

        steps = get_steps()
        if not steps:
            return None

        step_summaries = []
        for step in steps:
            weights = getattr(step, "weights", None)
            weights_cpu = weights.detach().float().cpu() if weights is not None else None
            weights_summary = self._tensor_stats(weights_cpu)
            if weights_cpu is not None:
                weights_summary = {
                    **weights_summary,
                    "nonzero": int(torch.count_nonzero(weights_cpu).item()),
                }

            step_summaries.append(
                {
                    "step_idx": int(getattr(step, "step_idx", 0)),
                    "time": self._json_scalar(getattr(step, "time", None)),
                    "guidance_weight": self._json_scalar(getattr(step, "guidance_weight", None)),
                    "inference_delay": getattr(step, "inference_delay", None),
                    "execution_horizon": getattr(step, "execution_horizon", None),
                    "weights": weights_summary,
                    "correction": self._tensor_stats(getattr(step, "correction", None)),
                    "err": self._tensor_stats(getattr(step, "err", None)),
                    "x_t": self._tensor_stats(getattr(step, "x_t", None)),
                    "v_t": self._tensor_stats(getattr(step, "v_t", None)),
                    "x1_t": self._tensor_stats(getattr(step, "x1_t", None)),
                }
            )

        reset_tracker = getattr(rtc_processor, "reset_tracker", None)
        if callable(reset_tracker):
            reset_tracker()

        return {
            "event": "rtc_debug",
            "timestamp_s": time.time(),
            "observation_timestep": observation_timestep,
            "chunk_start_timestep": chunk_start_timestep,
            "chunk_len": chunk_len,
            "total_latency_s": total_latency_s,
            "inference_delay": inference_delay,
            "leftover_len": 0 if prev_chunk_left_over is None else int(prev_chunk_left_over.shape[0]),
            "steps": step_summaries,
        }

    def _write_rtc_debug_dump(
        self,
        *,
        observation_timestep: int,
        chunk_start_timestep: int,
        chunk_len: int,
        total_latency_s: float,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
    ) -> None:
        if self._rtc_debug_dump_path is None:
            return

        summary = self._rtc_debug_summary(
            observation_timestep=observation_timestep,
            chunk_start_timestep=chunk_start_timestep,
            chunk_len=chunk_len,
            total_latency_s=total_latency_s,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
        )
        if summary is None:
            return

        with self._rtc_debug_dump_lock:
            with self._rtc_debug_dump_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(summary, sort_keys=True) + "\n")

    def _policy_action_names(self) -> list[str] | None:
        """Return policy action dimension names when the loaded config exposes them."""
        policy_config = getattr(self.policy, "config", None)
        action_names = getattr(policy_config, "action_feature_names", None)
        if action_names:
            return list(action_names)

        output_features = getattr(policy_config, "output_features", None)
        action_feature = output_features.get("action") if isinstance(output_features, dict) else None
        feature_names = getattr(action_feature, "names", None)
        if feature_names:
            return list(feature_names)
        if isinstance(action_feature, dict) and action_feature.get("names"):
            return list(action_feature["names"])

        return None

    def _policy_rtc_config_enabled(self) -> bool:
        policy_config = getattr(self.policy, "config", None)
        rtc_config = getattr(policy_config, "rtc_config", None)
        return bool(rtc_config is not None and getattr(rtc_config, "enabled", False))

    def _async_rtc_enabled(self) -> bool:
        return bool(self.config.async_rtc.enabled and self.policy is not None and self._policy_rtc_config_enabled())

    def _log_async_rtc_status(self) -> None:
        if self.config.async_rtc.enabled and self._policy_rtc_config_enabled():
            self.logger.info("Async RTC guidance enabled for policy server")
        elif self.config.async_rtc.enabled:
            self.logger.warning("Async RTC requested, but loaded policy rtc_config is missing or disabled")
        elif self._policy_rtc_config_enabled():
            self.logger.info("Policy rtc_config is enabled; async RTC runtime switch is disabled")

    def _rtc_inference_delay(self) -> int:
        latency = self._rtc_latency_tracker.percentile(self.config.async_rtc.latency_quantile)
        if not latency:
            return 0
        return max(0, math.ceil(latency / self.config.environment_dt))

    def _policy_device(self) -> torch.device | str:
        policy_config = getattr(self.policy, "config", None)
        return self.device or getattr(policy_config, "device", None) or "cpu"

    def _rtc_leftover_for_observation(self, obs_timestep: int) -> torch.Tensor | None:
        if self._rtc_original_actions is None or self._rtc_chunk_start_timestep is None:
            return None

        # The client reports latest_action as the observation timestep, so the
        # next unexecuted action is obs_timestep + 1.
        offset = max(0, obs_timestep + 1 - self._rtc_chunk_start_timestep)
        if offset >= self._rtc_original_actions.shape[0]:
            return None

        leftover = self._rtc_original_actions[offset:].clone()

        if (
            self._rtc_relative_step is not None
            and self._rtc_processed_actions is not None
            and offset < self._rtc_processed_actions.shape[0]
        ):
            get_cached_state = getattr(self._rtc_relative_step, "get_cached_state", None)
            current_state = get_cached_state() if callable(get_cached_state) else None
            processed_leftover = self._rtc_processed_actions[offset:].clone()
            if current_state is not None and processed_leftover.numel() > 0:
                try:
                    leftover = reanchor_relative_rtc_prefix(
                        prev_actions_absolute=processed_leftover,
                        current_state=current_state,
                        relative_step=self._rtc_relative_step,
                        normalizer_step=self._rtc_normalizer_step,
                        policy_device=self._policy_device(),
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("Failed to re-anchor RTC relative prefix; using original prefix: %s", exc)

        return leftover.to(self._policy_device())

    def _update_rtc_state(
        self,
        original_actions: torch.Tensor,
        processed_actions: torch.Tensor,
        chunk_start_timestep: int,
        total_latency_s: float,
        inference_delay: int,
        prev_chunk_left_over: torch.Tensor | None,
    ) -> None:
        self._rtc_original_actions = original_actions.squeeze(0).detach().cpu()
        self._rtc_processed_actions = processed_actions.detach().cpu()
        self._rtc_chunk_start_timestep = chunk_start_timestep
        self._rtc_latency_tracker.add(total_latency_s)

        leftover_len = 0 if prev_chunk_left_over is None else int(prev_chunk_left_over.shape[0])
        self.logger.info(
            "Async RTC chunk state updated | obs_timestep=%s | inference_delay=%s | "
            "leftover_len=%s | chunk_len=%s | latency=%.3fs",
            chunk_start_timestep,
            inference_delay,
            leftover_len,
            int(self._rtc_original_actions.shape[0]),
            total_latency_s,
        )

    def _trim_stale_rtc_actions(
        self,
        original_actions: torch.Tensor,
        processed_actions: torch.Tensor,
        *,
        chunk_start_timestamp: float,
        chunk_start_timestep: int,
        total_latency_s: float,
    ) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        """Discard RTC actions that expired while this chunk was being generated."""
        delay_steps = max(0, math.ceil(total_latency_s / self.config.environment_dt))
        chunk_len = int(processed_actions.shape[0])
        clamped_delay = min(delay_steps, chunk_len)

        if clamped_delay == 0:
            return original_actions, processed_actions, chunk_start_timestamp, chunk_start_timestep

        self.logger.info(
            "Async RTC trimming stale actions | delay_steps=%s | clamped_delay=%s | chunk_len=%s",
            delay_steps,
            clamped_delay,
            chunk_len,
        )
        trimmed_timestamp = chunk_start_timestamp + clamped_delay * self.config.environment_dt
        trimmed_timestep = chunk_start_timestep + clamped_delay
        return (
            original_actions[:, clamped_delay:, :],
            processed_actions[clamped_delay:, :],
            trimmed_timestamp,
            trimmed_timestep,
        )

    def _get_action_chunk(
        self,
        observation: dict[str, torch.Tensor],
        *,
        inference_delay: int | None = None,
        prev_chunk_left_over: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        if self._async_rtc_enabled():
            chunk = self.policy.predict_action_chunk(
                observation,
                inference_delay=0 if inference_delay is None else inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
            )
        else:
            chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        rtc_active = self._async_rtc_enabled()
        rtc_prev_chunk_left_over = (
            self._rtc_leftover_for_observation(observation_t.get_timestep()) if rtc_active else None
        )
        rtc_inference_delay = self._rtc_inference_delay() if rtc_active else None

        start_inference = time.perf_counter()
        if rtc_active:
            action_tensor = self._get_action_chunk(
                observation,
                inference_delay=rtc_inference_delay,
                prev_chunk_left_over=rtc_prev_chunk_left_over,
            )
        else:
            action_tensor = self._get_action_chunk(observation)
        original_action_tensor = action_tensor.detach().clone() if rtc_active else None
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess
        total_latency_s = postprocess_stops - start_prepare
        chunk_start_timestamp = observation_t.get_timestamp()
        chunk_start_timestep = observation_t.get_timestep()

        if rtc_active and original_action_tensor is not None:
            original_action_tensor, action_tensor, chunk_start_timestamp, chunk_start_timestep = (
                self._trim_stale_rtc_actions(
                    original_action_tensor,
                    action_tensor,
                    chunk_start_timestamp=chunk_start_timestamp,
                    chunk_start_timestep=chunk_start_timestep,
                    total_latency_s=total_latency_s,
                )
            )

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(chunk_start_timestamp, list(action_tensor), chunk_start_timestep)
        if rtc_active and original_action_tensor is not None:
            self._update_rtc_state(
                original_actions=original_action_tensor,
                processed_actions=action_tensor,
                chunk_start_timestep=chunk_start_timestep,
                total_latency_s=total_latency_s,
                inference_delay=0 if rtc_inference_delay is None else rtc_inference_delay,
                prev_chunk_left_over=rtc_prev_chunk_left_over,
            )
            self._write_rtc_debug_dump(
                observation_timestep=observation_t.get_timestep(),
                chunk_start_timestep=chunk_start_timestep,
                chunk_len=len(action_chunk),
                total_latency_s=total_latency_s,
                inference_delay=0 if rtc_inference_delay is None else rtc_inference_delay,
                prev_chunk_left_over=rtc_prev_chunk_left_over,
            )

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * total_latency_s:.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * total_latency_s:.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
