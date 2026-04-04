# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
F1 Openenv Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import F1OpenenvAction, F1OpenenvObservation


class F1OpenenvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.total_laps = 50

    # -----------------------------
    # SAFE INIT (IMPORTANT)
    # -----------------------------
    def _ensure_initialized(self):
        if not hasattr(self, "tire_type"):
            random.seed(42)
            self.lap = 1
            self.position = 10
            self.tire_type = "medium"
            self.tire_wear = 0.0
            self.fuel = 100.0
            self.weather = "dry"
            self.done = False

    # -----------------------------
    # RESET
    # -----------------------------
    def reset(self) -> F1OpenenvObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        random.seed(42)

        self.lap = 1
        self.position = 10
        self.tire_type = "medium"
        self.tire_wear = 0.0
        self.fuel = 100.0
        self.weather = "dry"
        self.done = False

        return self._get_obs(0.0)

    # -----------------------------
    # STEP
    # -----------------------------
    def step(self, action: F1OpenenvAction) -> F1OpenenvObservation:
        self._state.step_count += 1

        self._ensure_initialized()

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

        # -----------------------------
        # WEATHER CHANGE
        # -----------------------------
        if random.random() < 0.1:
            self.weather = "rain" if self.weather == "dry" else "dry"

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
        )

        # -----------------------------
        # POSITION UPDATE (FIXED LOGIC)
        # -----------------------------
        performance_score = 100 - lap_time

        random_factor = random.uniform(-2, 2)

        delta = int((performance_score + random_factor) / 10)

        self.position = max(1, min(20, self.position - delta))

        # penalty for worn tires
        if self.tire_wear > 0.8:
            self.position = min(20, self.position + 1)

        # -----------------------------
        # FUEL + LAP
        # -----------------------------
        self.fuel -= 2
        self.lap += 1

        if self.lap > self.total_laps:
            self.done = True

        # -----------------------------
        # REWARD (IMPROVED)
        # -----------------------------
        reward = 0

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

        return self._get_obs(reward, lap_time)

    # -----------------------------
    # OBSERVATION
    # -----------------------------
    def _get_obs(self, reward, lap_time=None):
        return F1OpenenvObservation(
            lap=self.lap,
            position=self.position,
            tire_type=self.tire_type,
            tire_wear=self.tire_wear,
            fuel=self.fuel,
            weather=self.weather,
            rain_probability=0.1,
            gap_ahead=round(random.uniform(1.0, 3.0), 2),
            gap_behind=round(random.uniform(1.0, 3.0), 2),
            safety_car=random.random() < 0.05,
            done=self.done,
            reward=reward,
            metadata={"lap_time": lap_time},
        )

    @property
    def state(self) -> State:
        return self._state