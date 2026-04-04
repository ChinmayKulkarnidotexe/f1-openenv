from typing import Dict

def easy_task_config() -> Dict:
    return {
        "name": "dry_race",
        "laps": 30,
        "weather": "dry",
        "rain_probability": 0.0,
        "safety_car_probability": 0.0,
    }

def medium_task_config() -> Dict:
    return {
        "name": "variable_weather",
        "laps": 40,
        "weather": "mixed",
        "rain_probability": 0.3,
        "safety_car_probability": 0.1,
    }

def hard_task_config() -> Dict:
    return {
        "name": "chaos_race",
        "laps": 50,
        "weather": "dynamic",
        "rain_probability": 0.5,
        "safety_car_probability": 0.3,
    }

TASKS = [
    easy_task_config,
    medium_task_config,
    hard_task_config
]