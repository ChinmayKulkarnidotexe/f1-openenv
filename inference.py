from openai import OpenAI
from client import F1OpenenvEnv
from models import F1OpenenvAction
from grader import grade_episode
from dotenv import load_dotenv
from tasks import TASKS
import json
import re
import os

load_dotenv()

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)


def run_task(task_config):
    with F1OpenenvEnv(base_url="http://localhost:8000").sync() as env:
        # pass task config to server via reset kwargs
        obs = env.reset(**task_config)

        total_laps = task_config["laps"]
        history = []

        for step in range(total_laps):
            prompt = f"""You are an F1 race engineer AI. Make strategic decisions for this lap.

Race: {task_config['name']} | Lap {step + 1}/{total_laps} | Weather mode: {task_config['weather']}

Current state:
{obs}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"pit": True/False, "tire_choice": "soft"/"medium"/"hard", "push_level": "low"/"medium"/"high"}}"""

            response = client.chat.completions.create(
                model="qwen3:0.6b",
                messages=[{"role": "user", "content": prompt}],
            )

            raw_content = response.choices[0].message.content
            action_json = parse_action(raw_content)

            print(f"\n\n[Lap {step + 1}/{total_laps}] Action: {action_json}")
            action = F1OpenenvAction(**action_json)

            result = env.step(action)

            history.append({
                "observation": result.observation,
                "reward": result.reward,
                "info": result.observation.metadata
            })

            obs = result
            print(f"\nStep {step} results:", result)

            if result.done:
                print(f"  Race finished at lap {step + 1}")
                break

        score = grade_episode(history, total_laps)
        return score


def parse_action(text):
    """Parse LLM output into an action dict, handling common formatting issues."""
    if not text:
        return _default_action()

    # strip qwen3 <think>...</think> tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.strip("`").strip()

    # try direct JSON parse
    try:
        parsed = json.loads(text)
        return _validate_action(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # try to find JSON object in the text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group())
            return _validate_action(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    return _default_action()


def _validate_action(parsed):
    """Validate and normalize a parsed action dict."""
    pit = parsed.get("pit", False)
    if isinstance(pit, str):
        pit = pit.lower() in ("true", "yes", "1")

    tire = parsed.get("tire_choice", "medium")
    if tire not in ("soft", "medium", "hard"):
        tire = "medium"

    push = parsed.get("push_level", "medium")
    if push not in ("low", "medium", "high"):
        push = "medium"

    return {"pit": bool(pit), "tire_choice": tire, "push_level": push}


def _default_action():
    return {"pit": False, "tire_choice": "medium", "push_level": "medium"}


if __name__ == "__main__":
    for task in TASKS:
        task_config = task()
        print(f"\n{'='*50}")
        print(f"Starting task: {task_config['name']} ({task_config['laps']} laps)")
        print(f"{'='*50}")
        score = run_task(task_config)
        print(f"\n{task_config['name']} → Score: {score}")
