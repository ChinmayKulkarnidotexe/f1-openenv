"""
F1 OpenEnv Task Configurations.

Each task defines a race scenario with specific parameters.
These are designed to test different aspects of race strategy:
  - Dry race: pure tire strategy and pit timing
  - Mixed weather: adapting to rain mid-race
  - Chaos race: long race with frequent incidents and weather changes
"""

from typing import Dict


def dry_race_config() -> Dict:
    """
    Clean dry race (Bahrain-style).
    Focus: tire strategy, 2-3 pit stops, compound regulation.
    No rain, very low SC probability.
    """
    return {
        "name": "bahrain_dry",
        "laps": 30,
        "weather": "dry",
        "rain_probability": 0.0,
        "safety_car_probability": 0.03,
        "start_position": 10,
        "start_tire": "medium",
        "start_fuel": 110.0,
    }


def mixed_weather_config() -> Dict:
    """
    Variable weather race (Silverstone-style).
    Focus: adapting strategy when rain arrives, tire compound switches.
    Moderate SC probability, rain can come and go.
    """
    return {
        "name": "silverstone_mixed",
        "laps": 40,
        "weather": "mixed",
        "rain_probability": 0.25,
        "safety_car_probability": 0.05,
        "start_position": 10,
        "start_tire": "medium",
        "start_fuel": 110.0,
    }


def chaos_race_config() -> Dict:
    """
    High-chaos race (Monaco-style).
    Focus: SC/VSC exploitation, frequent incidents, dynamic weather.
    Tests reactive strategy under pressure.
    """
    return {
        "name": "monaco_chaos",
        "laps": 50,
        "weather": "dynamic",
        "rain_probability": 0.35,
        "safety_car_probability": 0.08,
        "start_position": 10,
        "start_tire": "soft",
        "start_fuel": 110.0,
    }


TASKS = [
    dry_race_config,
    mixed_weather_config,
    chaos_race_config,
]