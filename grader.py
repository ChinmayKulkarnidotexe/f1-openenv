def grade_episode(history: list, total_laps: int = 50) -> float:
    """
    Grade an episode based on final position, cumulative reward,
    and pit strategy efficiency.

    Args:
        history: list of step dicts with 'observation', 'reward', 'info'
        total_laps: total laps in the race (used to scale pit strategy score)
    """
    if not history:
        return 0.0

    final_state = history[-1]["observation"]

    # position score: P1 = 1.0, P20 = 0.0
    final_position = final_state.position
    avg_position_score = 1 - (final_position / 20)

    # cumulative reward score (normalized)
    total_reward = sum(step["reward"] for step in history)
    max_possible_reward = total_laps * 12  # rough upper bound
    reward_score = min(max(total_reward / max_possible_reward, 0), 1)

    # strategy efficiency: penalize excessive pitting relative to race length
    pit_count = sum(1 for step in history if step["info"].get("pitted", False))
    expected_pits = max(1, total_laps // 15)  # ~1 pit per 15 laps is reasonable
    strategy_score = max(0, 1 - (pit_count / (expected_pits * 3)))

    score = (
        0.5 * avg_position_score +
        0.3 * reward_score +
        0.2 * strategy_score
    )
    print(f"  Grading: pos={final_position} ({avg_position_score:.2f}), "
          f"reward={total_reward:.1f} ({reward_score:.2f}), "
          f"pits={pit_count}/{expected_pits} expected ({strategy_score:.2f}) → {score:.4f}")

    return round(min(max(score, 0), 1), 4)