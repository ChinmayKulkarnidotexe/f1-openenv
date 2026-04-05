# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the F1 OpenEnv Environment.

Supports all 5 real F1 tire compounds (soft, medium, hard, intermediate, wet)
and realistic race state observations for RL training.
"""
from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Literal, List

# All 5 F1 tire compounds
TIRE_COMPOUNDS = Literal["soft", "medium", "hard", "intermediate", "wet"]

# Dry-weather compounds (count toward mandatory 2-compound rule)
DRY_COMPOUNDS = {"soft", "medium", "hard"}

# Wet-weather compounds
WET_COMPOUNDS = {"intermediate", "wet"}


class F1OpenenvAction(Action):
    """Action for F1 race engineer decisions."""

    pit: bool = Field(..., description="Whether to pit this lap")
    tire_choice: TIRE_COMPOUNDS = Field(
        ..., description="Tire compound to fit (only used if pit=True)"
    )
    push_level: Literal["low", "medium", "high"] = Field(
        ..., description="Engine mode / push level for this lap"
    )


class F1OpenenvObservation(Observation):
    """Observation of F1 race state — provides all info an RL agent needs."""

    lap: int = Field(..., description="Current lap number (1-indexed)")
    total_laps: int = Field(..., description="Total laps in the race")
    position: int = Field(..., description="Current race position (1-20)")
    tire_type: TIRE_COMPOUNDS = Field(..., description="Currently fitted compound")
    tire_wear: float = Field(..., description="0.0 = fresh, 1.0 = destroyed")
    tire_age: int = Field(..., description="Laps on current tire set")
    fuel: float = Field(..., description="Fuel remaining in kg (starts ~110)")
    weather: Literal["dry", "light_rain", "heavy_rain"] = Field(
        ..., description="Current weather condition"
    )
    rain_probability: float = Field(
        ..., description="Probability of weather change next lap"
    )
    gap_ahead: float = Field(..., description="Gap to car ahead in seconds")
    gap_behind: float = Field(..., description="Gap to car behind in seconds")
    safety_car: bool = Field(..., description="Whether any SC/VSC is active")
    laps_since_pit: int = Field(
        ..., description="Laps since last pit stop (same as tire_age)"
    )
    track_status: Literal["green", "vsc", "safety_car", "red_flag"] = Field(
        ..., description="Current track status flag"
    )
    incident: str = Field(
        default="none",
        description="Current incident type (none, crash, spin, debris, mechanical)",
    )
    sc_laps_remaining: int = Field(
        default=0, description="Laps remaining under current SC/VSC period"
    )
    compounds_used: List[str] = Field(
        ..., description="All tire compounds used so far in this race"
    )
    pit_stops_made: int = Field(..., description="Total pit stops made so far")
    is_wet_race: bool = Field(
        ...,
        description="If True, the mandatory 2 dry-compound rule is waived",
    )
