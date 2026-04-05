# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""F1 OpenEnv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import F1OpenenvAction, F1OpenenvObservation


class F1OpenenvEnv(
    EnvClient[F1OpenenvAction, F1OpenenvObservation, State]
):
    def _step_payload(self, action: F1OpenenvAction) -> Dict:
        return {
            "pit": action.pit,
            "tire_choice": action.tire_choice,
            "push_level": action.push_level,
        }

    def _parse_result(self, payload: Dict) -> StepResult[F1OpenenvObservation]:
        obs_data = payload.get("observation", {})

        observation = F1OpenenvObservation(
            lap=obs_data.get("lap", 1),
            total_laps=obs_data.get("total_laps", 50),
            position=obs_data.get("position", 10),
            tire_type=obs_data.get("tire_type", "medium"),
            tire_wear=obs_data.get("tire_wear", 0.0),
            tire_age=obs_data.get("tire_age", 0),
            fuel=obs_data.get("fuel", 100.0),
            weather=obs_data.get("weather", "dry"),
            rain_probability=obs_data.get("rain_probability", 0.1),
            gap_ahead=obs_data.get("gap_ahead", 1.0),
            gap_behind=obs_data.get("gap_behind", 1.0),
            safety_car=obs_data.get("safety_car", False),
            laps_since_pit=obs_data.get("laps_since_pit", 0),
            track_status=obs_data.get("track_status", "green"),
            incident=obs_data.get("incident", "none"),
            sc_laps_remaining=obs_data.get("sc_laps_remaining", 0),
            compounds_used=obs_data.get("compounds_used", ["medium"]),
            pit_stops_made=obs_data.get("pit_stops_made", 0),
            is_wet_race=obs_data.get("is_wet_race", False),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# TEST RUN
if __name__ == "__main__":
    with F1OpenenvEnv(base_url="http://localhost:8000").sync() as client:
        # pass task config via reset kwargs
        obs = client.reset(
            laps=30, weather="dry", rain_probability=0.0,
            safety_car_probability=0.0, start_position=10,
        )
        print("RESET:", obs)

        for i in range(30):
            result = client.step(
                F1OpenenvAction(
                    pit=(i == 14),  # pit once on lap 15
                    tire_choice="hard" if i == 14 else "medium",
                    push_level="medium",
                )
            )
            print(
                f"Lap {i+1}: P{result.observation.position} | "
                f"{result.observation.tire_type} ({result.observation.tire_wear:.0%}) | "
                f"fuel {result.observation.fuel:.0f}kg | "
                f"reward {result.reward:.2f}"
            )
            if result.done:
                print(f"Race finished at lap {i+1}")
                break