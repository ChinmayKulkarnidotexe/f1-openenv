# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
F1 OpenEnv Environment -- Realistic Race Strategy Simulation.

A reinforcement learning environment simulating F1 race strategy decisions.
An agent acts as a race engineer, deciding pit stops, tire choices, and push
levels each lap. The simulation models:
  - 5 tire compounds with non-linear degradation and cliff behavior
  - Dynamic weather (dry / light rain / heavy rain) with Markov transitions
  - Safety Car & Virtual Safety Car with proper pit-cost discounting
  - 20-car field with gap tracking and position changes
  - Fuel load affecting lap time
  - FIA regulation: must use >= 2 different dry compounds in a dry race

Architecture note -- Multi-agent expansion:
  To run 20 agents each controlling a different car, you would:
  1. Move the per-car state (tire, fuel, position, pit history) into a CarState
     dataclass and store a list of 20 CarState objects.
  2. Change step() to accept (car_index, action) -- each agent submits its
     action for its car each lap.
  3. Process all 20 actions, compute all 20 lap times, then rank-sort to
     determine positions. Return per-agent observations.
  4. Optionally use a vectorized EnvClient that sends all 20 actions in a
     single request and receives 20 observations back.
  The shared state (weather, SC, incidents) stays global; only per-car state
  is replicated. This file is structured to make that refactor straightforward
  by keeping car-state in clearly separated instance variables.
"""

from uuid import uuid4
from typing import Any, Literal, Optional, List, Dict
import random
import math

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import F1OpenenvAction, F1OpenenvObservation, DRY_COMPOUNDS


# ====================================================================
# TIRE PHYSICS CONSTANTS
# ====================================================================

TIRE_BASE_DEGRADATION = {
    "soft":         0.040,
    "medium":       0.025,
    "hard":         0.017,
    "intermediate": 0.030,
    "wet":          0.025,
}

TIRE_CLIFF = {
    "soft":         0.60,
    "medium":       0.75,
    "hard":         0.85,
    "intermediate": 0.70,
    "wet":          0.80,
}

# Pace advantage vs medium (seconds per lap, negative = faster)
TIRE_PACE_OFFSET = {
    "soft":         -1.5,
    "medium":        0.0,
    "hard":         +1.2,
    "intermediate": +4.0,
    "wet":          +8.0,
}

PUSH_WEAR_FACTOR = {"low": 0.75, "medium": 1.0, "high": 1.40}
PUSH_PACE_OFFSET = {"low": +1.0, "medium": 0.0, "high": -1.5}

FUEL_CONSUMPTION = {"low": 1.5, "medium": 1.85, "high": 2.4}

# Fuel weight effect (seconds per kg) -- realistic F1 is ~0.035s/kg
FUEL_WEIGHT_EFFECT = 0.035

# Pit stop time loss by track status (seconds)
PIT_TIME_LOSS = {"green": 22.0, "vsc": 13.0, "safety_car": 8.0, "red_flag": 0.0}

SC_DURATION_RANGE = (3, 6)


# ====================================================================
# WEATHER TRANSITIONS (Markov chain)
# ====================================================================
WEATHER_TRANSITIONS = {
    "dry":        {"dry": 0.90, "light_rain": 0.10, "heavy_rain": 0.00},
    "light_rain": {"dry": 0.25, "light_rain": 0.55, "heavy_rain": 0.20},
    "heavy_rain": {"dry": 0.00, "light_rain": 0.30, "heavy_rain": 0.70},
}


# ====================================================================
# FIELD SIMULATION (19 AI opponents)
# ====================================================================

def _compute_expected_player_laptime(fuel: float = 110.0) -> float:
    """
    Compute what the player's lap time would be on lap 1 with medium tires,
    0% wear, full fuel, medium push, green track, dry weather.
    This is used to calibrate the AI field so positions are realistic.
    """
    base = 90.0
    compound_offset = TIRE_PACE_OFFSET["medium"]  # 0.0
    wear_offset = 0.0  # fresh tires
    fuel_offset = fuel * FUEL_WEIGHT_EFFECT
    push_offset = PUSH_PACE_OFFSET["medium"]  # 0.0
    return base + compound_offset + wear_offset + fuel_offset + push_offset


def _generate_field(n_cars: int = 20, start_position: int = 10, start_fuel: float = 110.0) -> List[Dict]:
    """
    Generate a 20-car field calibrated to the player's expected performance.

    The field is spaced so that:
      - P1 car is ~2s/lap faster than the player's initial pace
      - P20 car is ~2s/lap slower
      - Player's base pace slots in at their starting position
    """
    expected_player = _compute_expected_player_laptime(start_fuel)

    # Total spread across the field: 4s (P1 is 2s faster, P20 is 2s slower)
    spread = 4.0
    fastest = expected_player - spread * (start_position - 1) / (n_cars - 1)

    field = []
    for i in range(n_cars):
        # Linear spread from fastest to slowest
        base_pace = fastest + (i / (n_cars - 1)) * spread
        consistency = random.uniform(0.2, 0.5)
        field.append({
            "base_pace": base_pace,
            "consistency": consistency,
            "cumulative_time": 0.0,
            "tire_age": 0,
            "fuel": start_fuel,
            "has_pitted": False,
            "retired": False,
            "pitting_cooldown": 0,
        })
    return field


class F1OpenenvEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    # ----------------------------------------------------------------
    # RESET
    # ----------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        laps: int = 50,
        weather: str = "dry",
        rain_probability: float = 0.1,
        safety_car_probability: float = 0.05,
        start_position: int = 10,
        start_tire: str = "medium",
        start_fuel: float = 110.0,
        **kwargs: Any,
    ) -> F1OpenenvObservation:
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        # Task config
        self.total_laps = laps
        self.rain_probability = rain_probability
        self.safety_car_probability = safety_car_probability
        self._weather_mode = weather

        # Player car state
        self.lap = 0
        self.position = max(1, min(20, start_position))
        self.tire_type = start_tire
        self.tire_wear = 0.0
        self.tire_age = 0
        self.fuel = start_fuel
        self.done = False

        # Pit / regulation tracking
        self.pit_stops_made = 0
        self.compounds_used: List[str] = [self.tire_type]
        self.pit_history: List[int] = []
        self.consecutive_pit_count = 0

        # Weather state
        if weather == "wet":
            self.weather = "heavy_rain"
        elif weather in ("mixed", "dynamic"):
            self.weather = random.choices(
                ["dry", "light_rain", "heavy_rain"],
                weights=[0.6, 0.3, 0.1],
            )[0]
        else:
            self.weather = "dry"
        self.rain_laps = 0
        self.track_wetness = 1.0 if self.weather != "dry" else 0.0

        # Safety Car state
        self.track_status: str = "green"
        self.incident: str = "none"
        self.sc_laps_remaining = 0
        self.safety_car = False

        # Field -- calibrated to player's expected lap time
        self.field = _generate_field(20, self.position, start_fuel)
        self.player_cumulative_time = 0.0

        # Lap time tracking
        self.last_lap_time: Optional[float] = None
        self.position_history: List[int] = [self.position]

        return self._build_observation(reward=0.0, pitted=False)

    # ----------------------------------------------------------------
    # STEP (one lap of racing)
    # ----------------------------------------------------------------
    def step(self, action: F1OpenenvAction, **kwargs: Any) -> F1OpenenvObservation:
        self._state.step_count += 1
        self.lap += 1

        prev_position = self.position
        pitted_this_lap = False

        # == 1. PIT STOP =============================================
        pit_time_loss = 0.0
        if action.pit:
            pitted_this_lap = True
            pit_time_loss = PIT_TIME_LOSS.get(self.track_status, 22.0)
            self.tire_type = action.tire_choice
            self.tire_wear = 0.0
            self.tire_age = 0
            self.pit_stops_made += 1
            self.pit_history.append(self.lap)

            if self.tire_type not in self.compounds_used:
                self.compounds_used.append(self.tire_type)

            # Back-to-back tracking
            if len(self.pit_history) >= 2 and self.pit_history[-2] == self.lap - 1:
                self.consecutive_pit_count += 1
            else:
                self.consecutive_pit_count = 0
        else:
            if self.consecutive_pit_count > 0:
                self.consecutive_pit_count = 0

        # == 2. TIRE DEGRADATION =====================================
        base_deg = TIRE_BASE_DEGRADATION.get(self.tire_type, 0.025)
        push_factor = PUSH_WEAR_FACTOR.get(action.push_level, 1.0)

        weather_wear_mult = 1.0
        if self.weather == "light_rain" and self.tire_type in DRY_COMPOUNDS:
            weather_wear_mult = 2.5
        elif self.weather == "heavy_rain" and self.tire_type in DRY_COMPOUNDS:
            weather_wear_mult = 4.0
        elif self.weather == "dry" and self.tire_type in ("intermediate", "wet"):
            weather_wear_mult = 2.0

        degradation = base_deg * push_factor * weather_wear_mult
        degradation *= random.uniform(0.85, 1.15)

        cliff = TIRE_CLIFF.get(self.tire_type, 0.75)
        if self.tire_wear >= cliff:
            overshoot = (self.tire_wear - cliff) / (1.0 - cliff + 0.01)
            cliff_mult = 3.0 + 3.0 * overshoot
            degradation *= cliff_mult

        self.tire_wear = min(1.0, self.tire_wear + degradation)
        self.tire_age += 1

        # == 3. FUEL ==================================================
        fuel_burn = FUEL_CONSUMPTION.get(action.push_level, 1.85)
        if self.track_status in ("safety_car", "vsc"):
            fuel_burn *= 0.6
        self.fuel = max(0.0, self.fuel - fuel_burn)

        # == 4. WEATHER ===============================================
        self._update_weather()
        if self.weather != "dry":
            self.rain_laps += 1

        # == 5. SAFETY CAR ============================================
        self._update_safety_car()

        # == 6. LAP TIME (player) =====================================
        player_lap_time = self._compute_lap_time(
            tire_type=self.tire_type,
            tire_wear=self.tire_wear,
            fuel=self.fuel,
            push_level=action.push_level,
            pit_loss=pit_time_loss,
        )
        self.player_cumulative_time += player_lap_time

        # == 7. FIELD + POSITION ======================================
        self._update_field()
        self._compute_position()
        self.position_history.append(self.position)

        self.last_lap_time = player_lap_time

        # == 8. RACE END ==============================================
        if self.lap >= self.total_laps:
            self.done = True

        # == 9. REWARD ================================================
        reward = self._compute_reward(
            prev_position=prev_position,
            pitted=pitted_this_lap,
            pit_time_loss=pit_time_loss,
        )

        return self._build_observation(reward=reward, pitted=pitted_this_lap)

    # ----------------------------------------------------------------
    # WEATHER
    # ----------------------------------------------------------------
    def _update_weather(self):
        if self._weather_mode == "dry":
            self.weather = "dry"
            self.track_wetness = max(0.0, self.track_wetness - 0.3)
            return

        probs = WEATHER_TRANSITIONS[self.weather].copy()

        if self.weather == "dry":
            rain_prob = probs["light_rain"] * (self.rain_probability / 0.1)
            probs["light_rain"] = min(rain_prob, 0.4)
            probs["dry"] = 1.0 - probs["light_rain"] - probs["heavy_rain"]

        states = list(probs.keys())
        weights = [probs[s] for s in states]
        self.weather = random.choices(states, weights=weights)[0]

        if self.weather == "heavy_rain":
            self.track_wetness = min(1.0, self.track_wetness + 0.3)
        elif self.weather == "light_rain":
            self.track_wetness = min(1.0, self.track_wetness + 0.1)
        else:
            self.track_wetness = max(0.0, self.track_wetness - 0.35)

    # ----------------------------------------------------------------
    # SAFETY CAR
    # ----------------------------------------------------------------
    def _update_safety_car(self):
        if self.sc_laps_remaining > 0:
            self.sc_laps_remaining -= 1
            if self.sc_laps_remaining == 0:
                self.track_status = "green"
                self.incident = "none"
                self.safety_car = False
            return

        if self.track_status == "green" and random.random() < self.safety_car_probability:
            self.incident = random.choice(["crash", "spin", "debris", "mechanical"])

            if self.incident == "crash":
                sc_type = random.choices(["safety_car", "vsc"], weights=[0.75, 0.25])[0]
            elif self.incident == "debris":
                sc_type = random.choices(["safety_car", "vsc"], weights=[0.40, 0.60])[0]
            else:
                sc_type = random.choices(["safety_car", "vsc"], weights=[0.30, 0.70])[0]

            self.track_status = sc_type
            self.safety_car = True
            self.sc_laps_remaining = random.randint(*SC_DURATION_RANGE)
            self._handle_incident_retirement()
        else:
            self.track_status = "green"
            self.incident = "none"
            self.safety_car = False

    def _handle_incident_retirement(self):
        active_cars = [i for i, c in enumerate(self.field) if i > 0 and not c["retired"]]
        if active_cars and random.random() < 0.4:
            retiree = random.choice(active_cars)
            self.field[retiree]["retired"] = True

    # ----------------------------------------------------------------
    # LAP TIME
    # ----------------------------------------------------------------
    def _compute_lap_time(self, tire_type, tire_wear, fuel, push_level, pit_loss=0.0):
        base = 90.0

        compound_offset = TIRE_PACE_OFFSET.get(tire_type, 0.0)

        weather_offset = 0.0
        if self.weather == "light_rain":
            if tire_type in DRY_COMPOUNDS:
                weather_offset = +3.5
            elif tire_type == "intermediate":
                weather_offset = -1.0
            elif tire_type == "wet":
                weather_offset = +1.5
        elif self.weather == "heavy_rain":
            if tire_type in DRY_COMPOUNDS:
                weather_offset = +8.0
            elif tire_type == "intermediate":
                weather_offset = +2.0
            elif tire_type == "wet":
                weather_offset = -1.5

        cliff = TIRE_CLIFF.get(tire_type, 0.75)
        if tire_wear < cliff:
            wear_offset = (tire_wear / cliff) * 2.0
        else:
            overshoot = (tire_wear - cliff) / (1.0 - cliff + 0.01)
            wear_offset = 2.0 + 13.0 * (overshoot ** 1.5)

        fuel_offset = fuel * FUEL_WEIGHT_EFFECT

        push_offset = PUSH_PACE_OFFSET.get(push_level, 0.0)

        sc_offset = 0.0
        if self.track_status == "safety_car":
            sc_offset = +25.0
        elif self.track_status == "vsc":
            sc_offset = +12.0

        noise = random.gauss(0, 0.4)

        lap_time = (
            base + compound_offset + weather_offset + wear_offset
            + fuel_offset + push_offset + sc_offset + noise + pit_loss
        )

        return max(80.0, lap_time)

    # ----------------------------------------------------------------
    # FIELD
    # ----------------------------------------------------------------
    def _update_field(self):
        for i, car in enumerate(self.field):
            if i == 0 or car["retired"]:
                continue

            # Base time + noise
            lap_time = car["base_pace"] + random.gauss(0, car["consistency"])

            # AI fuel burn (same rate as player medium push)
            car["fuel"] = max(0.0, car["fuel"] - 1.85)

            # AI fuel weight effect (same physics as player)
            lap_time += car["fuel"] * FUEL_WEIGHT_EFFECT

            # Weather penalty for AI (they run medium compounds)
            if self.weather == "light_rain":
                lap_time += 2.5
            elif self.weather == "heavy_rain":
                lap_time += 5.0

            # SC/VSC: everyone goes the same speed
            if self.track_status == "safety_car":
                lap_time = 115.0
            elif self.track_status == "vsc":
                lap_time = 102.0

            # AI tire age penalty (simplified linear)
            car["tire_age"] += 1
            age_penalty = max(0, (car["tire_age"] - 18)) * 0.12
            lap_time += age_penalty

            # AI pit stops
            if car["pitting_cooldown"] > 0:
                car["pitting_cooldown"] -= 1
            elif car["tire_age"] > random.randint(18, 30):
                pit_cost = PIT_TIME_LOSS.get(self.track_status, 22.0)
                lap_time += pit_cost
                car["tire_age"] = 0
                car["fuel"] = max(0.0, car["fuel"])
                car["has_pitted"] = True
                car["pitting_cooldown"] = 8

            car["cumulative_time"] += lap_time

    def _compute_position(self):
        times = []
        for i, car in enumerate(self.field):
            if car["retired"]:
                continue
            if i == 0:
                times.append((self.player_cumulative_time, 0))
            else:
                times.append((car["cumulative_time"], i))

        times.sort(key=lambda x: x[0])

        for pos, (_, car_idx) in enumerate(times, 1):
            if car_idx == 0:
                self.position = pos
                break

        player_rank = self.position
        self._gap_ahead = 0.0
        self._gap_behind = 0.0

        if player_rank > 1 and len(times) >= player_rank:
            ahead_time = times[player_rank - 2][0]
            self._gap_ahead = round(self.player_cumulative_time - ahead_time, 2)

        if player_rank < len(times):
            behind_time = times[player_rank][0]
            self._gap_behind = round(behind_time - self.player_cumulative_time, 2)

        # Gaps compress under SC
        if self.track_status == "safety_car":
            self._gap_ahead = min(self._gap_ahead, 1.5)
            self._gap_behind = min(self._gap_behind, 1.5)
        elif self.track_status == "vsc":
            self._gap_ahead = min(self._gap_ahead, 3.0)
            self._gap_behind = min(self._gap_behind, 3.0)

    # ----------------------------------------------------------------
    # REWARD
    # ----------------------------------------------------------------
    def _compute_reward(self, prev_position, pitted, pit_time_loss):
        """
        Reward signal balanced so that a car holding position mid-pack
        gets a small positive reward (~1-3 per lap). Penalties for bad
        decisions subtract from this baseline.
        """
        reward = 0.0

        # -- Position: baseline positive reward --
        # P1 = +3.0, P10 = +1.5, P20 = 0.0 per lap
        reward += (20 - self.position) * 0.15

        # -- Position delta: gained = bonus, lost = penalty --
        pos_delta = prev_position - self.position
        reward += pos_delta * 1.5

        # -- Tire management --
        cliff = TIRE_CLIFF.get(self.tire_type, 0.75)
        if self.tire_wear > cliff:
            overshoot = self.tire_wear - cliff
            reward -= 2.0 + overshoot * 8.0  # past cliff penalty
        elif self.tire_wear > (cliff - 0.08):
            reward -= 0.3  # gentle nudge near cliff

        # -- Pit decisions --
        if pitted:
            reward -= 0.5  # small cost for pitting (lost time)

            # Back-to-back pit: severe penalty
            if self.consecutive_pit_count > 0:
                reward -= 4.0 * self.consecutive_pit_count

            # Bonus for pitting under SC/VSC
            if self.track_status in ("safety_car", "vsc"):
                saved = PIT_TIME_LOSS["green"] - pit_time_loss
                reward += saved * 0.2

        # Excessive total pits (beyond 3)
        if self.pit_stops_made > 3:
            reward -= (self.pit_stops_made - 3) * 1.0

        # -- Weather mismatch --
        if self.weather == "heavy_rain" and self.tire_type in DRY_COMPOUNDS:
            reward -= 3.0
        elif self.weather == "light_rain" and self.tire_type in DRY_COMPOUNDS:
            reward -= 1.5
        elif self.weather == "dry" and self.tire_type in ("intermediate", "wet"):
            reward -= 2.0

        # -- Fuel --
        if self.fuel < 3.0:
            reward -= 5.0
        elif self.fuel < 8.0:
            reward -= 1.0

        # -- Regulation warning --
        dry_used = {c for c in self.compounds_used if c in DRY_COMPOUNDS}
        is_wet_race = (self.rain_laps / max(1, self.lap)) > 0.30
        laps_remaining = self.total_laps - self.lap

        if not is_wet_race and len(dry_used) < 2:
            if laps_remaining <= 3:
                reward -= 3.0
            elif laps_remaining <= 8:
                reward -= 0.5

        # -- Race finish --
        if self.done:
            if self.position <= 3:
                reward += 15.0
            elif self.position <= 10:
                reward += 8.0
            elif self.position <= 15:
                reward += 3.0
            else:
                reward += 1.0

            if not is_wet_race and len(dry_used) < 2:
                reward -= 10.0

            if self.pit_stops_made == 0:
                reward -= 8.0
            elif self.pit_stops_made == 1:
                reward -= 2.0
            elif self.pit_stops_made in (2, 3):
                reward += 3.0
            elif self.pit_stops_made >= 5:
                reward -= 5.0

        return round(reward, 4)

    # ----------------------------------------------------------------
    # OBSERVATION
    # ----------------------------------------------------------------
    def _build_observation(self, reward: float, pitted: bool) -> F1OpenenvObservation:
        is_wet_race = (self.rain_laps / max(1, self.lap + 1)) > 0.30

        return F1OpenenvObservation(
            lap=max(1, self.lap),
            total_laps=self.total_laps,
            position=self.position,
            tire_type=self.tire_type,
            tire_wear=round(self.tire_wear, 4),
            tire_age=self.tire_age,
            fuel=round(self.fuel, 2),
            weather=self.weather,
            rain_probability=self.rain_probability,
            gap_ahead=round(getattr(self, "_gap_ahead", random.uniform(0.5, 3.0)), 2),
            gap_behind=round(getattr(self, "_gap_behind", random.uniform(0.5, 3.0)), 2),
            safety_car=self.safety_car,
            laps_since_pit=self.tire_age,
            track_status=self.track_status,
            incident=self.incident,
            sc_laps_remaining=self.sc_laps_remaining,
            compounds_used=list(self.compounds_used),
            pit_stops_made=self.pit_stops_made,
            is_wet_race=is_wet_race,
            done=self.done,
            reward=round(reward, 4),
            metadata={
                "lap_time": round(self.last_lap_time, 2) if self.last_lap_time else None,
                "pitted": pitted,
                "total_laps": self.total_laps,
                "regulation_violation": (
                    not is_wet_race
                    and len({c for c in self.compounds_used if c in DRY_COMPOUNDS}) < 2
                    and self.done
                ),
                "rain_laps": self.rain_laps,
                "track_wetness": round(self.track_wetness, 2),
                "pit_history": list(self.pit_history),
                "consecutive_pits": self.consecutive_pit_count,
            },
        )

    @property
    def state(self) -> State:
        return self._state