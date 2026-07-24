#!/usr/bin/env python

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

from dataclasses import dataclass
from typing import TypeAlias

from ..config import TeleoperatorConfig


@dataclass
class SOLeaderConfig:
    """Base configuration class for SO Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True


@TeleoperatorConfig.register_subclass("so101_leader")
@TeleoperatorConfig.register_subclass("so100_leader")
@dataclass
class SOLeaderTeleopConfig(TeleoperatorConfig, SOLeaderConfig):
    pass


@TeleoperatorConfig.register_subclass("so101_8dof_leader")
@dataclass
class SO1018DofLeaderConfig(SOLeaderTeleopConfig):
    """Custom SO101 leader with 7 joints + 1 gripper.

    Motor IDs:
      joint_1 -> 1
      joint_2 -> 2
      joint_3 -> 3
      joint_4 -> 4
      joint_5 -> 5
      joint_6 -> 6
      joint_7 -> 7
      gripper -> 8
    """
    pass


SO100LeaderConfig: TypeAlias = SOLeaderTeleopConfig
SO101LeaderConfig: TypeAlias = SOLeaderTeleopConfig
