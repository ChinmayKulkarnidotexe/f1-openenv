import os
import re
import json
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

from client import F1OpenenvEnv
from models import F1OpenenvAction
from grader import grade_episode
from tasks import TASKS

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "qwen3:0.6b")
BENCHMARK    = os.getenv("BENCHMARK", "f1-openenv")

TEMPERATURE = 0.2
MAX_TOKENS = 200
FALLBACK_ACTION = {"pit": False, "tire_choice": "medium", "push_level": "medium"}

DEBUG = True


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def parse_action(text):
    """Parse LLM output into an action dict, handling common formatting issues."""
    if not text:
        return dict(FALLBACK_ACTION)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = text.strip("`").strip()

    try:
        parsed = json.loads(text)
        return _validate_action(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            parsed = json.loads(match.group())
            return _validate_action(parsed)
        except (json.JSONDecodeError, ValueError):
            pass

    return dict(FALLBACK_ACTION)


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


def action_to_str(action_dict: dict) -> str:
    """Convert action dict to a compact string for logging."""
    return json.dumps(action_dict, separators=(",", ":"))


def run_task(task_config):
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_name = task_config["name"]
    total_laps = task_config["laps"]

    history = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = F1OpenenvEnv(base_url="http://localhost:8000").sync()

    try:
        obs = env.reset(**task_config)

        for step in range(1, total_laps + 1):
            prompt = f"""You are an F1 race engineer AI. Make strategic decisions for this lap.

Race: {task_name} | Lap {step}/{total_laps} | Weather mode: {task_config['weather']}

Current state:
{obs}

Respond with ONLY a JSON object (no markdown, no explanation):
{{"pit": True/False, "tire_choice": "soft"/"medium"/"hard", "push_level": "low"/"medium"/"high"}}"""

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                raw_content = response.choices[0].message.content or ""
            except Exception as exc:
                raw_content = ""
                if DEBUG:
                    print(f"[DEBUG] Model request failed: {exc}", flush=True)

            action_json = parse_action(raw_content)
            action = F1OpenenvAction(**action_json)
            action_str = action_to_str(action_json)

            result = env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append({
                "observation": result.observation,
                "reward": reward,
                "info": result.observation.metadata
            })

            obs = result.observation

            if done:
                success = reward > 0.0
                if DEBUG:
                    print(f"[DEBUG] Race finished at lap {step}", flush=True)
                break

        else:
            success = False

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)

    score = grade_episode(history, total_laps)
    return score


if __name__ == "__main__":
    for task in TASKS:
        task_config = task()
        print(f"\n{'='*50}")
        print(f"Starting task: {task_config['name']} ({task_config['laps']} laps)")
        print(f"{'='*50}")
        score = run_task(task_config)
        print(f"\n{task_config['name']} → Score: {score}")
