# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
F1 OpenEnv Environment Implementation.

A reinforcement learning environment simulating F1 race strategy decisions.
An agent acts as a race engineer, deciding pit stops, tire choices, and push levels.
"""

from uuid import uuid4
from typing import Any, Literal, Optional
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import F1OpenenvAction, F1OpenenvObservation


class F1OpenenvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        # defaults — overridden by reset() kwargs from task_config
        self.total_laps = 50
        self.rain_probability = 0.1
        self.safety_car_probability = 0.05
        self._init_weather = "dry"

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        # task_config fields
        laps: int = 50,
        weather: Literal["dry", "mixed", "dynamic"] = "dry",
        rain_probability: float = 0.1,
        safety_car_probability: float = 0.05,
        **kwargs: Any,
    ) -> F1OpenenvObservation:
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        # seed randomness
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        # apply task config
        self.total_laps = laps
        self.rain_probability = rain_probability
        self.safety_car_probability = safety_car_probability
        self._init_weather = weather

        # initial race state
        self.lap = 1
        self.position = 10
        self.tire_type = "medium"
        self.tire_wear = 0.0
        self.fuel = 100.0
        self.safety_car = False
        self.done = False

        # resolve starting weather
        if weather in ("mixed", "dynamic"):
            self.weather = "rain" if random.random() < rain_probability else "dry"
        else:
            self.weather = "dry"

        return self._get_obs(reward=0.0, pit=False)

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action: F1OpenenvAction, **kwargs: Any) -> F1OpenenvObservation:
        self._state.step_count += 1

        # -----------------------------
        # PIT STOP
        # -----------------------------
        if action.pit:
            self.tire_type = action.tire_choice
            self.tire_wear = 0.0
            pit_penalty = 20
        else:
            pit_penalty = 0

        # -----------------------------
        # TIRE WEAR
        # -----------------------------
        push_factor = {
            "low": 0.8,
            "medium": 1.0,
            "high": 1.3,
        }[action.push_level]

        tire_deg = {
            "soft": 0.035,
            "medium": 0.025,
            "hard": 0.018,
        }[self.tire_type]

        self.tire_wear += tire_deg * push_factor

        # rain makes tires degrade faster
        if self.weather == "rain":
            self.tire_wear += 0.01

        # -----------------------------
        # WEATHER CHANGE
        # -----------------------------
        if self._init_weather == "dynamic":
            if random.random() < self.rain_probability:
                self.weather = "rain" if self.weather == "dry" else "dry"
        elif self._init_weather == "mixed":
            if random.random() < self.rain_probability * 0.5:
                self.weather = "rain" if self.weather == "dry" else "dry"
        # "dry" → weather stays dry

        # -----------------------------
        # SAFETY CAR
        # -----------------------------
        self.safety_car = random.random() < self.safety_car_probability

        # -----------------------------
        # LAP TIME
        # -----------------------------
        base_time = 90

        lap_time = (
            base_time
            + self.tire_wear * 12
            + self.fuel * 0.04
            + (12 if self.weather == "rain" else 0)
            + pit_penalty
            + (-5 if self.safety_car else 0)  # safety car bunches field up
        )

        # -----------------------------
        # POSITION UPDATE
        # -----------------------------
        performance_score = 100 - lap_time
        random_factor = random.uniform(-2, 2)
        delta = int((performance_score + random_factor) / 10)

        if self.safety_car:
            # under safety car positions are mostly frozen
            delta = 0

        self.position = max(1, min(20, self.position - delta))

        # penalty for worn tires
        if self.tire_wear > 0.8:
            self.position = min(20, self.position + 1)

        # wrong tires for conditions penalty
        if self.weather == "rain" and self.tire_type != "soft":
            self.position = min(20, self.position + 1)

        # -----------------------------
        # FUEL + LAP
        # -----------------------------
        fuel_consumption = {"low": 1.5, "medium": 2.0, "high": 2.5}[action.push_level]
        self.fuel = max(0.0, self.fuel - fuel_consumption)
        self.lap += 1

        if self.lap > self.total_laps:
            self.done = True

        # -----------------------------
        # REWARD
        # -----------------------------
        reward = 0.0

        # position reward (most important)
        reward += (20 - self.position) * 0.6

        # efficiency
        reward -= lap_time * 0.01

        # tire management
        reward -= self.tire_wear * 2.5

        # pit penalty
        if action.pit:
            reward -= 2

        # bad tire punishment
        if self.tire_wear > 0.9:
            reward -= 5

        # wrong tire for weather
        if self.weather == "rain" and self.tire_type != "soft":
            reward -= 3

        # fuel management — running low
        if self.fuel < 10:
            reward -= 2

        return self._get_obs(reward, action.pit, lap_time)

    # -----------------------------
    # OBSERVATION
    # -----------------------------
    def _get_obs(self, reward, pit, lap_time=None):
        return F1OpenenvObservation(
            lap=self.lap,
            position=self.position,
            tire_type=self.tire_type,
            tire_wear=round(self.tire_wear, 4),
            fuel=round(self.fuel, 2),
            weather=self.weather,
            rain_probability=self.rain_probability,
            gap_ahead=round(random.uniform(0.5, 4.0), 2),
            gap_behind=round(random.uniform(0.5, 4.0), 2),
            safety_car=self.safety_car if hasattr(self, "safety_car") else False,
            done=self.done,
            reward=round(reward, 4),
            metadata={
                "lap_time": round(lap_time, 2) if lap_time else None,
                "pitted": pit,
                "total_laps": self.total_laps,
            },
        )

    @property
    def state(self) -> State:
        return self._state