# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""F1 Openenv Environment Client."""

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
            lap=obs_data.get("lap"),
            position=obs_data.get("position"),
            tire_type=obs_data.get("tire_type"),
            tire_wear=obs_data.get("tire_wear"),
            fuel=obs_data.get("fuel"),
            weather=obs_data.get("weather"),
            rain_probability=obs_data.get("rain_probability"),
            gap_ahead=obs_data.get("gap_ahead"),
            gap_behind=obs_data.get("gap_behind"),
            safety_car=obs_data.get("safety_car"),
            done=payload.get("done"),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done"),
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
        print("RESET:", client.reset(laps=30, weather="dry", rain_probability=0.0, safety_car_probability=0.0))

        for i in range(30):
            result = client.step(
                F1OpenenvAction(
                    pit=False,
                    tire_choice="medium",
                    push_level="medium",
                )
            )
            print(f"Step {i+1}:", result)
            if result.done:
                print(f"Race finished at step {i+1}")
                break