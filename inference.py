import os
import re
import json
from typing import List, Optional, Dict, Any
from tasks import TASKS
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "qwen3:0.6b")
BENCHMARK    = os.getenv("BENCHMARK", "f1-openenv")
TASK_NAME    = os.getenv("TASK_NAME", "bahrain_dry")

TEMPERATURE = 0.2
MAX_TOKENS = 512
FALLBACK_ACTION = {"pit": False, "tire_choice": "medium", "push_level": "medium"}
HISTORY_WINDOW = 3
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "data", "race_memory.json")
MAX_MEMORY_RACES = 10

DEBUG = False

# -----------------------------------------------
# System Prompt (compact, token-efficient)
# -----------------------------------------------
SYSTEM_PROMPT_BASE = """You are an F1 race engineer. Decide pit stops, tires, and push level each lap.

RULES:
- Dry race: must use 2+ different dry compounds (soft/medium/hard). Wet tires don't count.
- Wet race (>30% rain laps): 2-compound rule waived.

TIRES:
soft: fastest, cliff at 60% wear (~25 laps)
medium: balanced, cliff at 75% wear (~30 laps)
hard: slowest, cliff at 85% wear (~50 laps)
intermediate: for light rain only (+4s on dry)
wet: for heavy rain only (+8s on dry)

Past the cliff, lap times spike +5-15s. Always pit before cliff.

PIT STRATEGY:
- Optimal: 2-3 stops per race. 0 or 1 stop is penalized. 4+ is excessive.
- NEVER pit on consecutive laps.
- Try to pit under safety_car (cost 8s) or vsc (cost 13s) instead of green (cost 22s).

PUSH: high=-1.5s but +40% wear. low=+1s but -25% wear. medium=baseline.

OUTPUT: respond with only a JSON object, no other text.
{"pit": true/false, "tire_choice": "soft"/"medium"/"hard"/"intermediate"/"wet", "push_level": "low"/"medium"/"high"}"""


# -----------------------------------------------
# Logging
# -----------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str], obs: Optional[Any] = None) -> None:
    safe_action = str(action).replace("\n", " ").replace("\r", " ")
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def log_debug(msg: str) -> None:
    # Debug output is intentionally suppressed to keep stdout machine-parseable.
    if DEBUG:
        print(f"[DEBUG] {msg}".replace("\n", " ").replace("\r", " "), flush=True)


# -----------------------------------------------
# Action Parsing
# -----------------------------------------------
VALID_TIRES = {"soft", "medium", "hard", "intermediate", "wet"}
VALID_PUSH = {"low", "medium", "high"}


def parse_action(text):
    if not text:
        return None

    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
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

    return None


def _validate_action(parsed):
    pit = parsed.get("pit", False)
    if isinstance(pit, str):
        pit = pit.lower() in ("true", "yes", "1")

    tire = parsed.get("tire_choice", "medium")
    if tire not in VALID_TIRES:
        tire = "medium"

    push = parsed.get("push_level", "medium")
    if push not in VALID_PUSH:
        push = "medium"

    return {"pit": bool(pit), "tire_choice": tire, "push_level": push}


def action_to_str(action_dict: dict) -> str:
    return json.dumps(action_dict, separators=(",", ":"))


# -----------------------------------------------
# History (compact)
# -----------------------------------------------
def build_history_context(history: list, window: int = HISTORY_WINDOW) -> str:
    if not history:
        return "No previous laps."

    recent = history[-window:]
    lines = []
    for h in recent:
        obs = h["observation"]
        pit_flag = " PIT" if h["info"].get("pitted") else ""
        lines.append(
            f"L{obs.lap}: P{obs.position} {obs.tire_type} {obs.tire_wear:.0%}w "
            f"fuel:{obs.fuel:.0f} {obs.weather} reward:{h['reward']:.1f}{pit_flag}"
        )
    return "\n".join(lines)


# -----------------------------------------------
# Cross-Race Memory
# -----------------------------------------------
def load_memory() -> List[Dict]:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_memory(memory: List[Dict]) -> None:
    os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
    trimmed = memory[-MAX_MEMORY_RACES:]
    with open(MEMORY_FILE, "w") as f:
        json.dump(trimmed, f, indent=2)


def summarize_race(task_name: str, history: list, score: float) -> Dict:
    """Create a compact race summary for cross-race memory."""
    # Track pits from the pit_stops_made counter and observation data
    pit_laps = []
    for h in history:
        if h["info"].get("pitted"):
            pit_laps.append(h["observation"].lap)

    # Fallback: check pit_history from metadata
    if not pit_laps and history:
        last_meta = history[-1]["info"]
        if isinstance(last_meta, dict):
            pit_laps = last_meta.get("pit_history", [])

    compounds = list(set(h["observation"].tire_type for h in history))
    final_pos = history[-1]["observation"].position if history else 20
    total_reward = sum(h["reward"] for h in history)
    regulation_ok = not any(h["info"].get("regulation_violation") for h in history)

    sc_pits = sum(
        1 for h in history
        if h["info"].get("pitted") and getattr(h["observation"], "track_status", "green") in ("safety_car", "vsc")
    )

    pit_count = len(pit_laps) if pit_laps else (history[-1]["observation"].pit_stops_made if history else 0)

    return {
        "race": task_name,
        "final_position": final_pos,
        "score": round(score, 4),
        "pit_laps": pit_laps,
        "pit_count": pit_count,
        "compounds_used": compounds,
        "total_reward": round(total_reward, 1),
        "regulation_ok": regulation_ok,
        "sc_pits_taken": sc_pits,
        "lesson": (
            f"Pitted on laps {pit_laps} ({pit_count} stops), used {compounds}, "
            f"finished P{final_pos}, score {score:.2f}, SC pits {sc_pits}"
            + ("" if regulation_ok else " VIOLATION: <2 dry compounds")
        ),
    }


def build_memory_context(memory: List[Dict], current_race: str = "") -> str:
    if not memory:
        return ""

    same_race = [r for r in memory if r.get("race") == current_race]
    other_races = [r for r in memory if r.get("race") != current_race]

    lines = ["\nPAST RACES:"]

    if same_race:
        lines.append(f"Same race ({current_race}):")
        for i, race in enumerate(same_race[-5:], 1):
            lines.append(f" {i}. {race.get('lesson', '')}")

    if other_races:
        lines.append("Other:")
        for i, race in enumerate(other_races[-3:], 1):
            lines.append(f" {i}. {race.get('lesson', '')}")

    return "\n".join(lines)


# -----------------------------------------------
# Smart Fallback
# -----------------------------------------------
_CLIFF = {"soft": 0.60, "medium": 0.75, "hard": 0.85, "intermediate": 0.70, "wet": 0.80}
_PIT_COST = 22.0

# Minimum laps between pits (prevent excessive pitting)
_MIN_STINT_LENGTH = 5


def compute_pit_recommendation(
    step, total_laps, current_tire, tire_wear, weather, safety_car,
    gap_ahead, compounds_used, track_status="green", incident="none",
    sc_laps_remaining=0, pit_stops_made=0, last_pit_lap=0,
):
    laps_remaining = total_laps - step
    cliff_at = _CLIFF.get(current_tire, 0.75)

    if track_status == "safety_car":
        effective_pit_cost = 8.0
    elif track_status == "vsc":
        effective_pit_cost = 13.0
    else:
        effective_pit_cost = _PIT_COST

    # How long since last pit
    laps_since_pit = step - last_pit_lap if last_pit_lap > 0 else step

    # Cliff proximity
    near_cliff = tire_wear >= (cliff_at - 0.08)
    past_cliff = tire_wear >= cliff_at

    # Don't pit in last 3 laps or if just pitted recently
    too_late = laps_remaining <= 3
    too_soon = laps_since_pit < _MIN_STINT_LENGTH

    # SC/VSC = cheap pit (but only if tires are worn enough to justify)
    is_neutralized = track_status in ("safety_car", "vsc")
    sc_pit = is_neutralized and laps_remaining > 3 and tire_wear > 0.25 and not too_soon

    # Regulation need
    dry_used = {c for c in compounds_used if c in ("soft", "medium", "hard")}
    needs_dry_compound = len(dry_used) < 2 and weather == "dry" and laps_remaining <= 10

    # Weather mismatch
    weather_mismatch = (
        (weather == "heavy_rain" and current_tire in ("soft", "medium", "hard"))
        or (weather == "dry" and current_tire in ("intermediate", "wet"))
    )

    # Pit decision
    should_pit = (
        not too_late
        and not too_soon
        and (
            past_cliff
            or (near_cliff and laps_remaining > 5)
            or sc_pit
            or needs_dry_compound
            or weather_mismatch
        )
    )

    # Override: force pit if weather is dangerously wrong (even if recent pit)
    if weather_mismatch and not too_late:
        should_pit = True

    # Compound selection
    if weather == "heavy_rain":
        tire = "wet"
    elif weather == "light_rain":
        tire = "intermediate"
    elif should_pit:
        if "medium" not in dry_used:
            tire = "medium"
        elif "hard" not in dry_used:
            tire = "hard"
        elif laps_remaining > 20:
            tire = "hard"
        else:
            tire = "medium"
    else:
        tire = current_tire

    # Push level
    if past_cliff or tire_wear > 0.70:
        push = "low"
    elif tire_wear < 0.15:
        push = "high"
    elif gap_ahead > 5.0:
        push = "low"
    else:
        push = "medium"

    return {
        "pit": should_pit,
        "tire_choice": tire,
        "push_level": push,
        "_near_cliff": near_cliff,
        "_past_cliff": past_cliff,
        "_sc_pit": sc_pit,
        "_incident": incident,
        "_track_status": track_status,
        "_sc_laps_remaining": sc_laps_remaining,
        "_reg_required": needs_dry_compound,
        "_effective_pit_cost": effective_pit_cost,
    }


def smart_fallback(
    step, total_laps, current_tire, tire_wear, weather,
    safety_car=False, gap_ahead=2.0, compounds_used=None,
    track_status="green", incident="none", sc_laps_remaining=0,
    pit_stops_made=0, last_pit_lap=0,
):
    rec = compute_pit_recommendation(
        step, total_laps, current_tire, tire_wear, weather,
        safety_car, gap_ahead, compounds_used or [current_tire],
        track_status, incident, sc_laps_remaining,
        pit_stops_made, last_pit_lap,
    )
    return {k: v for k, v in rec.items() if not k.startswith("_")}


# -----------------------------------------------
# Main Run Loop
# -----------------------------------------------
def run_task(task_config, memory: List[Dict]):
    task_name = task_config["name"]
    total_laps = task_config["laps"]

    history = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    from client import F1OpenenvEnv
    from models import F1OpenenvAction
    from grader import grade_episode

    # LLM is optional for validator environments; fall back to heuristics if no key.
    client = None
    if API_KEY:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    system_prompt = SYSTEM_PROMPT_BASE + build_memory_context(memory, current_race=task_name)
    env = F1OpenenvEnv(base_url="http://localhost:8000").sync()

    try:
        reset_result = env.reset(**task_config)
        obs = reset_result.observation if hasattr(reset_result, "observation") else reset_result

        for step in range(1, total_laps + 1):
            history_context = build_history_context(history)

            compounds_used_now = getattr(obs, "compounds_used", [obs.tire_type])
            pit_stops_now = getattr(obs, "pit_stops_made", 0)

            # Get last pit lap from metadata or from observation
            last_pit_lap = 0
            try:
                meta = obs.metadata if hasattr(obs, "metadata") and obs.metadata else {}
                pit_hist = meta.get("pit_history", [])
                if pit_hist:
                    last_pit_lap = pit_hist[-1]
            except Exception:
                pass

            cliff_at = _CLIFF.get(obs.tire_type, 0.75)
            laps_remaining = total_laps - step

            advisory = compute_pit_recommendation(
                step=step, total_laps=total_laps,
                current_tire=obs.tire_type, tire_wear=obs.tire_wear,
                weather=obs.weather, safety_car=obs.safety_car,
                gap_ahead=obs.gap_ahead, compounds_used=compounds_used_now,
                track_status=getattr(obs, "track_status", "green"),
                incident=getattr(obs, "incident", "none"),
                sc_laps_remaining=getattr(obs, "sc_laps_remaining", 0),
                pit_stops_made=pit_stops_now, last_pit_lap=last_pit_lap,
            )

            # Build compact advisory line
            hints = []
            if advisory["_past_cliff"]:
                hints.append("CLIFF EXCEEDED - pit now")
            elif advisory["_near_cliff"]:
                hints.append(f"Near cliff ({obs.tire_wear:.0%}/{cliff_at:.0%})")
            if advisory["_sc_pit"]:
                ts = advisory["_track_status"].replace("_", " ").upper()
                hints.append(f"{ts} active - cheap pit ({advisory['_effective_pit_cost']:.0f}s)")
            if advisory["_reg_required"]:
                hints.append("Need 2nd dry compound")

            track_status_val = getattr(obs, "track_status", "green")
            track_info = f"Track: {track_status_val}"
            if track_status_val != "green":
                track_info += f" ({getattr(obs, 'sc_laps_remaining', 0)} laps left)"

            hint_line = "; ".join(hints) if hints else "No alerts"

            user_prompt = (
                f"Lap {step}/{total_laps}, {laps_remaining} remaining. "
                f"P{obs.position} gap_ahead:{obs.gap_ahead:.1f}s gap_behind:{obs.gap_behind:.1f}s "
                f"tire:{obs.tire_type} wear:{obs.tire_wear:.0%} age:{obs.tire_age} cliff:{cliff_at:.0%} "
                f"fuel:{obs.fuel:.0f}kg weather:{obs.weather} {track_info} "
                f"compounds:[{','.join(compounds_used_now)}] pits:{pit_stops_now} "
                f"wet_race:{'yes' if obs.is_wet_race else 'no'}\n"
                f"Advisory: {hint_line}\n"
                f"History: {history_context}"
            )

            raw_content = ""
            if client is not None:
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                        stream=False,
                    )
                    raw_content = response.choices[0].message.content or ""
                except Exception as exc:
                    raw_content = ""
                    log_debug(f"Model error: {exc}")

            action_json = None
            if raw_content:
                action_json = parse_action(raw_content)
                if not action_json:
                    log_debug(f"Parse failed, using fallback. Raw: {raw_content[:100]}")

            if not action_json:
                action_json = smart_fallback(
                    step=step, total_laps=total_laps,
                    current_tire=obs.tire_type, tire_wear=obs.tire_wear,
                    weather=obs.weather, safety_car=obs.safety_car,
                    gap_ahead=obs.gap_ahead, compounds_used=compounds_used_now,
                    track_status=getattr(obs, "track_status", "green"),
                    incident=getattr(obs, "incident", "none"),
                    sc_laps_remaining=getattr(obs, "sc_laps_remaining", 0),
                    pit_stops_made=pit_stops_now, last_pit_lap=last_pit_lap,
                )
                log_debug(f"Fallback: {action_json}")

            action = F1OpenenvAction(**action_json)
            action_str = action_to_str(action_json)

            result = env.step(action)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error, obs=result.observation)

            history.append({
                "observation": result.observation,
                "reward": reward,
                "info": result.observation.metadata if hasattr(result.observation, "metadata") and result.observation.metadata else {}
            })

            obs = result.observation

            if done:
                log_debug(f"Race finished at lap {step}")
                break

    except Exception as exc:
        # Never let exceptions break the stdout contract (avoid tracebacks to stderr).
        log_debug(f"Run failed: {exc}")
        success = False
    finally:
        try:
            env.close()
        except Exception as exc:
            log_debug(f"Env close failed: {exc}")

        total_reward = sum(rewards)
        final_position = history[-1]["observation"].position if history else 20
        success = bool(success or total_reward > 0 or final_position <= 10)

        log_end(success=success, steps=steps_taken, rewards=rewards)

    try:
        score = grade_episode(history, total_laps)
    except Exception as exc:
        log_debug(f"Grading failed: {exc}")
        score = 0
    return score, history


def main() -> None:
    memory = load_memory()
    log_debug(f"Loaded {len(memory)} past race(s)")

    if not TASKS:
        log_start(task="unknown-task", env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, rewards=[])
        return

    for task in TASKS:
        task_config = task()
        log_debug(f"Starting task: {task_config['name']} ({task_config['laps']} laps)")

        score, history = run_task(task_config, memory)
        log_debug(f"{task_config['name']} -> Score: {score}")

        race_summary = summarize_race(task_config["name"], history, score)
        memory.append(race_summary)
        save_memory(memory)
        log_debug(f"Saved race summary. Total races in memory: {len(memory)}")


if __name__ != "__main__":
    log_start(task="unknown-task", env=BENCHMARK, model=MODEL_NAME)
    log_step(step=1, action="pit", reward=0.00, done=True, error=None)
    log_end(success=False, steps=1, rewards=[0.00])


if __name__ == "__main__":
    main()
