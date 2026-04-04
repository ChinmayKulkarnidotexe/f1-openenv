# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the F1 Openenv Environment.

The f1_openenv environment is a simple test environment that echoes back messages.
"""
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Literal


class F1OpenenvAction(Action):
    """Action for F1 race engineer decisions"""

    pit: bool = Field(..., description="Whether to pit this lap")
    tire_choice: Literal["soft", "medium", "hard"] = Field(...)
    push_level: Literal["low", "medium", "high"] = Field(...)


class F1OpenenvObservation(Observation):
    """Observation of F1 race state"""

    lap: int = Field(...)
    position: int = Field(...)
    tire_type: Literal["soft", "medium", "hard"] = Field(...)
    tire_wear: float = Field(...)
    fuel: float = Field(...)
    weather: Literal["dry", "rain"] = Field(...)
    rain_probability: float = Field(...)
    gap_ahead: float = Field(...)
    gap_behind: float = Field(...)
    safety_car: bool = Field(...)
