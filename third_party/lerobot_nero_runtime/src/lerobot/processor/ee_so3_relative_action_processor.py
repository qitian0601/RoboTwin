#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""SO(3)/SE(3)-correct relative action transforms for dual-arm EE pose actions.

Expected 14-D layout:

right: [x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]
left:  [x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]

Expected 16-D local-SE(3) layout:

right: [x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]
left:  [x, y, z, rotvec_x, rotvec_y, rotvec_z, gripper]
base:  [x, y]
"""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE

from .pipeline import ProcessorStep, ProcessorStepRegistry

EE_POSE_DIM = 7
DUAL_ARM_EE_DIM = 14
DUAL_ARM_EE_WITH_BASE_DIM = 16
_EPS = 1e-7


def _skew(vectors: Tensor) -> Tensor:
    zeros = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((zeros, -z, y), dim=-1),
            torch.stack((z, zeros, -x), dim=-1),
            torch.stack((-y, x, zeros), dim=-1),
        ),
        dim=-2,
    )


def rotvec_to_matrix(rotvec: Tensor) -> Tensor:
    """Convert rotation vectors to rotation matrices with Rodrigues' formula."""
    if rotvec.shape[-1] != 3:
        raise ValueError(f"rotvec last dimension must be 3, got {rotvec.shape[-1]}")

    original_shape = rotvec.shape[:-1]
    flat = rotvec.reshape(-1, 3)
    theta = torch.linalg.norm(flat, dim=-1, keepdim=True)
    k = _skew(flat)
    k2 = k @ k

    theta2 = theta * theta
    small = theta < 1e-4
    sin_over_theta = torch.where(
        small,
        1 - theta2 / 6 + theta2 * theta2 / 120,
        torch.sin(theta) / theta.clamp_min(_EPS),
    )
    one_minus_cos_over_theta2 = torch.where(
        small,
        0.5 - theta2 / 24 + theta2 * theta2 / 720,
        (1 - torch.cos(theta)) / theta2.clamp_min(_EPS),
    )

    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(flat.shape[0], 3, 3)
    matrix = eye + sin_over_theta[..., None] * k + one_minus_cos_over_theta2[..., None] * k2
    return matrix.reshape(*original_shape, 3, 3)


def matrix_to_rotvec(matrix: Tensor) -> Tensor:
    """Convert rotation matrices to rotation vectors via the SO(3) logarithm map."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix last dimensions must be 3x3, got {matrix.shape[-2:]}")

    original_shape = matrix.shape[:-2]
    flat = matrix.reshape(-1, 3, 3)
    quat = _matrix_to_quat_xyzw(flat)
    rotvec = _quat_xyzw_to_rotvec(quat)
    return rotvec.reshape(*original_shape, 3)


def _matrix_to_quat_xyzw(matrix: Tensor) -> Tensor:
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1 + matrix[..., 0, 0] - matrix[..., 1, 1] - matrix[..., 2, 2],
                    1 - matrix[..., 0, 0] + matrix[..., 1, 1] - matrix[..., 2, 2],
                    1 - matrix[..., 0, 0] - matrix[..., 1, 1] + matrix[..., 2, 2],
                    1 + trace,
                ),
                dim=-1,
            ),
            min=0,
        )
    )

    quat_by_component = torch.stack(
        (
            torch.stack(
                (
                    q_abs[..., 0] ** 2,
                    matrix[..., 0, 1] + matrix[..., 1, 0],
                    matrix[..., 0, 2] + matrix[..., 2, 0],
                    matrix[..., 2, 1] - matrix[..., 1, 2],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    matrix[..., 0, 1] + matrix[..., 1, 0],
                    q_abs[..., 1] ** 2,
                    matrix[..., 1, 2] + matrix[..., 2, 1],
                    matrix[..., 0, 2] - matrix[..., 2, 0],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    matrix[..., 0, 2] + matrix[..., 2, 0],
                    matrix[..., 1, 2] + matrix[..., 2, 1],
                    q_abs[..., 2] ** 2,
                    matrix[..., 1, 0] - matrix[..., 0, 1],
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    matrix[..., 2, 1] - matrix[..., 1, 2],
                    matrix[..., 0, 2] - matrix[..., 2, 0],
                    matrix[..., 1, 0] - matrix[..., 0, 1],
                    q_abs[..., 3] ** 2,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )

    quat_candidates = quat_by_component / (2 * q_abs.clamp_min(_EPS)[..., None])
    best = q_abs.argmax(dim=-1)
    quat = quat_candidates[torch.arange(matrix.shape[0], device=matrix.device), best]
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(_EPS)
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    return quat


def _quat_xyzw_to_rotvec(quat: Tensor) -> Tensor:
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(_EPS)
    quat = torch.where(quat[..., 3:4] < 0, -quat, quat)
    xyz = quat[..., :3]
    w = quat[..., 3:4]
    sin_half = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, w.clamp_min(_EPS))
    scale = torch.where(sin_half < 1e-6, 2 + angle * angle / 12, angle / sin_half.clamp_min(_EPS))
    return xyz * scale


def _validate_ee_tensor(name: str, tensor: Tensor) -> None:
    if tensor.shape[-1] != DUAL_ARM_EE_DIM:
        raise ValueError(f"{name} must have last dimension {DUAL_ARM_EE_DIM}, got {tensor.shape[-1]}")


def _validate_ee_local_se3_tensor(name: str, tensor: Tensor) -> None:
    if tensor.shape[-1] != DUAL_ARM_EE_WITH_BASE_DIM:
        raise ValueError(
            f"{name} must have last dimension {DUAL_ARM_EE_WITH_BASE_DIM}, got {tensor.shape[-1]}"
        )


def _split_arms(tensor: Tensor) -> tuple[Tensor, Tensor]:
    return tensor[..., :EE_POSE_DIM], tensor[..., EE_POSE_DIM:DUAL_ARM_EE_DIM]


def _split_ee_and_base(tensor: Tensor) -> tuple[Tensor, Tensor]:
    return tensor[..., :DUAL_ARM_EE_DIM], tensor[..., DUAL_ARM_EE_DIM:DUAL_ARM_EE_WITH_BASE_DIM]


def _relative_arm_action(action_arm: Tensor, state_arm: Tensor) -> Tensor:
    pos_delta = action_arm[..., :3] - state_arm[..., :3]
    r_target = rotvec_to_matrix(action_arm[..., 3:6])
    r_current = rotvec_to_matrix(state_arm[..., 3:6])
    r_delta = r_target @ r_current.transpose(-1, -2)
    rot_delta = matrix_to_rotvec(r_delta)
    gripper = action_arm[..., 6:7]
    return torch.cat((pos_delta, rot_delta, gripper), dim=-1)


def _absolute_arm_action(relative_arm: Tensor, state_arm: Tensor) -> Tensor:
    pos = state_arm[..., :3] + relative_arm[..., :3]
    r_delta = rotvec_to_matrix(relative_arm[..., 3:6])
    r_current = rotvec_to_matrix(state_arm[..., 3:6])
    r_target = r_delta @ r_current
    rot = matrix_to_rotvec(r_target)
    gripper = relative_arm[..., 6:7]
    return torch.cat((pos, rot, gripper), dim=-1)


def _relative_local_se3_arm_action(action_arm: Tensor, state_arm: Tensor) -> Tensor:
    r_current = rotvec_to_matrix(state_arm[..., 3:6])
    r_target = rotvec_to_matrix(action_arm[..., 3:6])
    pos_delta = (r_current.transpose(-1, -2) @ (action_arm[..., :3] - state_arm[..., :3])[..., None]).squeeze(
        -1
    )
    r_delta = r_current.transpose(-1, -2) @ r_target
    rot_delta = matrix_to_rotvec(r_delta)
    gripper = action_arm[..., 6:7]
    return torch.cat((pos_delta, rot_delta, gripper), dim=-1)


def _absolute_local_se3_arm_action(relative_arm: Tensor, state_arm: Tensor) -> Tensor:
    r_current = rotvec_to_matrix(state_arm[..., 3:6])
    pos = state_arm[..., :3] + (r_current @ relative_arm[..., :3][..., None]).squeeze(-1)
    r_delta = rotvec_to_matrix(relative_arm[..., 3:6])
    r_target = r_current @ r_delta
    rot = matrix_to_rotvec(r_target)
    gripper = relative_arm[..., 6:7]
    return torch.cat((pos, rot, gripper), dim=-1)


def _broadcast_state_for_actions(state: Tensor, actions: Tensor) -> Tensor:
    if actions.ndim == state.ndim + 1:
        return state.unsqueeze(-2)
    if actions.ndim == state.ndim:
        return state
    raise ValueError(f"actions ndim must match state ndim or state ndim + 1, got {actions.ndim} and {state.ndim}")


def to_relative_ee_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert absolute EE target actions to SO(3)-relative EE deltas."""
    _validate_ee_tensor("actions", actions)
    _validate_ee_tensor("state", state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state = _broadcast_state_for_actions(state, actions)

    action_right, action_left = _split_arms(actions)
    state_right, state_left = _split_arms(state)
    return torch.cat(
        (
            _relative_arm_action(action_right, state_right),
            _relative_arm_action(action_left, state_left),
        ),
        dim=-1,
    )


def to_absolute_ee_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert SO(3)-relative EE deltas back to absolute EE target actions."""
    _validate_ee_tensor("actions", actions)
    _validate_ee_tensor("state", state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state = _broadcast_state_for_actions(state, actions)

    action_right, action_left = _split_arms(actions)
    state_right, state_left = _split_arms(state)
    return torch.cat(
        (
            _absolute_arm_action(action_right, state_right),
            _absolute_arm_action(action_left, state_left),
        ),
        dim=-1,
    )


def to_relative_ee_local_se3_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert 16-D absolute egocentric EE/base targets to local-SE(3) relative deltas.

    EE pose deltas use ``T_delta = inv(T_current) @ T_target`` for each arm.
    The final 2 base/head dimensions use chunk-start relative deltas:
    ``base_delta = base_target - base_current``.
    """
    _validate_ee_local_se3_tensor("actions", actions)
    _validate_ee_local_se3_tensor("state", state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state = _broadcast_state_for_actions(state, actions)

    action_ee, action_base = _split_ee_and_base(actions)
    state_ee, state_base = _split_ee_and_base(state)
    action_right, action_left = _split_arms(action_ee)
    state_right, state_left = _split_arms(state_ee)
    base_delta = action_base - state_base
    return torch.cat(
        (
            _relative_local_se3_arm_action(action_right, state_right),
            _relative_local_se3_arm_action(action_left, state_left),
            base_delta,
        ),
        dim=-1,
    )


def to_absolute_ee_local_se3_actions(actions: Tensor, state: Tensor) -> Tensor:
    """Convert 16-D local-SE(3) EE/base deltas back to absolute egocentric targets."""
    _validate_ee_local_se3_tensor("actions", actions)
    _validate_ee_local_se3_tensor("state", state)
    if state.device != actions.device or state.dtype != actions.dtype:
        state = state.to(device=actions.device, dtype=actions.dtype)
    state = _broadcast_state_for_actions(state, actions)

    relative_ee, relative_base = _split_ee_and_base(actions)
    state_ee, state_base = _split_ee_and_base(state)
    relative_right, relative_left = _split_arms(relative_ee)
    state_right, state_left = _split_arms(state_ee)
    base_target = state_base + relative_base
    return torch.cat(
        (
            _absolute_local_se3_arm_action(relative_right, state_right),
            _absolute_local_se3_arm_action(relative_left, state_left),
            base_target,
        ),
        dim=-1,
    )


@ProcessorStepRegistry.register("ee_so3_relative_actions_processor")
@dataclass
class EERelativeActionsProcessorStep(ProcessorStep):
    """Convert absolute dual-arm EE actions to SO(3)-relative deltas.

    The paired EEAbsoluteActionsProcessorStep uses the cached state to convert
    policy outputs back to absolute EE targets during inference.
    """

    enabled: bool = False
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        new_transition[TransitionKey.ACTION] = to_relative_ee_actions(action, state)
        return new_transition

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ee_so3_absolute_actions_processor")
@dataclass
class EEAbsoluteActionsProcessorStep(ProcessorStep):
    """Convert SO(3)-relative EE deltas back to absolute EE target actions."""

    enabled: bool = False
    relative_step: EERelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "EEAbsoluteActionsProcessorStep requires a paired EERelativeActionsProcessorStep "
                "but relative_step is None."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "EEAbsoluteActionsProcessorStep requires state from EERelativeActionsProcessorStep "
                "but no state has been cached."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        new_transition[TransitionKey.ACTION] = to_absolute_ee_actions(action, cached_state)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ee_local_se3_relative_actions_processor")
@dataclass
class EELocalSE3RelativeActionsProcessorStep(ProcessorStep):
    """Convert 16-D absolute egocentric EE/base actions to local-SE(3) relative deltas."""

    enabled: bool = False
    _last_state: Tensor | None = field(default=None, init=False, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION, {})
        state = observation.get(OBS_STATE) if observation else None
        if state is not None:
            self._last_state = state

        if not self.enabled:
            return transition

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None or state is None:
            return new_transition

        new_transition[TransitionKey.ACTION] = to_relative_ee_local_se3_actions(action, state)
        return new_transition

    def get_cached_state(self) -> Tensor | None:
        return self._last_state

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("ee_local_se3_absolute_actions_processor")
@dataclass
class EELocalSE3AbsoluteActionsProcessorStep(ProcessorStep):
    """Convert local-SE(3) EE/base deltas back to 16-D absolute egocentric targets."""

    enabled: bool = False
    relative_step: EELocalSE3RelativeActionsProcessorStep | None = field(default=None, repr=False)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition

        if self.relative_step is None:
            raise RuntimeError(
                "EELocalSE3AbsoluteActionsProcessorStep requires a paired "
                "EELocalSE3RelativeActionsProcessorStep but relative_step is None."
            )

        cached_state = self.relative_step.get_cached_state()
        if cached_state is None:
            raise RuntimeError(
                "EELocalSE3AbsoluteActionsProcessorStep requires state from "
                "EELocalSE3RelativeActionsProcessorStep but no state has been cached."
            )

        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            return new_transition

        new_transition[TransitionKey.ACTION] = to_absolute_ee_local_se3_actions(action, cached_state)
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
