"""
F1 OpenEnv Episode Grader.

Multi-criteria scoring system that evaluates a completed race episode across:
  - Final position (30%)
  - Pit strategy optimality (20%)
  - FIA regulation compliance (15%)
  - Tire management quality (15%)
  - SC/VSC exploitation (10%)
  - Consistency / racecraft (10%)

Returns a score in [0.0, 1.0].
"""

from typing import List, Dict, Set


def grade_episode(history: list, total_laps: int = 50) -> float:
    """
    Grade a completed race episode.

    Args:
        history: list of step dicts with keys:
            - 'observation': F1OpenenvObservation
            - 'reward': float
            - 'info': dict (from observation.metadata)
        total_laps: total laps in the race

    Returns:
        Score in [0.0, 1.0]
    """
    if not history:
        return 0.0

    final_obs = history[-1]["observation"]
    all_infos = [h["info"] for h in history]

    # ═══════════════════════════════════════════════
    # 1. FINAL POSITION SCORE  (30%)
    # ═══════════════════════════════════════════════
    final_position = final_obs.position
    # P1 = 1.0, P10 = 0.5, P20 = 0.0
    position_score = max(0.0, (20 - final_position) / 19)

    # ═══════════════════════════════════════════════
    # 2. PIT STRATEGY SCORE  (20%)
    # ═══════════════════════════════════════════════
    pit_laps = [i + 1 for i, info in enumerate(all_infos) if info.get("pitted")]
    pit_count = len(pit_laps)

    # Optimal: 2-3 pit stops for a standard race
    # Bell curve: peak at 2-3, drops off for 0-1 or 4+
    if pit_count == 0:
        pit_score = 0.05  # no pit = almost certainly a regulation violation
    elif pit_count == 1:
        pit_score = 0.35  # possible but usually suboptimal
    elif pit_count == 2:
        pit_score = 1.0   # ideal
    elif pit_count == 3:
        pit_score = 0.90  # very good, common in chaotic races
    elif pit_count == 4:
        pit_score = 0.55  # acceptable in wet/SC races
    else:
        # 5+ pits: drops sharply
        pit_score = max(0.0, 0.40 - (pit_count - 4) * 0.15)

    # Penalize back-to-back pitting
    consecutive_pits = 0
    for i in range(1, len(pit_laps)):
        if pit_laps[i] == pit_laps[i - 1] + 1:
            consecutive_pits += 1
    pit_score *= max(0.3, 1.0 - consecutive_pits * 0.25)

    # Bonus for well-spaced pits (not all clustered together)
    if len(pit_laps) >= 2:
        spacings = [pit_laps[i] - pit_laps[i - 1] for i in range(1, len(pit_laps))]
        avg_spacing = sum(spacings) / len(spacings)
        ideal_spacing = total_laps / (pit_count + 1)
        spacing_ratio = min(avg_spacing, ideal_spacing) / max(avg_spacing, ideal_spacing, 1)
        pit_score *= (0.7 + 0.3 * spacing_ratio)  # up to 30% bonus/penalty for spacing

    # ═══════════════════════════════════════════════
    # 3. REGULATION COMPLIANCE  (15%)
    # ═══════════════════════════════════════════════
    # Must use ≥2 different dry compounds unless it's a wet race
    is_wet_race = getattr(final_obs, "is_wet_race", False)
    compounds_used = getattr(final_obs, "compounds_used", [])
    dry_compounds_used = {c for c in compounds_used if c in {"soft", "medium", "hard"}}

    if is_wet_race:
        regulation_score = 1.0  # rule waived
    elif len(dry_compounds_used) >= 2:
        regulation_score = 1.0  # compliant
    elif len(dry_compounds_used) == 1 and pit_count > 0:
        regulation_score = 0.1  # pitted but used same compound = bad
    else:
        regulation_score = 0.0  # violated

    # ═══════════════════════════════════════════════
    # 4. TIRE MANAGEMENT  (15%)
    # ═══════════════════════════════════════════════
    # How well did the agent manage tire wear?
    cliff_thresholds = {
        "soft": 0.60, "medium": 0.75, "hard": 0.85,
        "intermediate": 0.70, "wet": 0.80,
    }

    laps_past_cliff = 0
    total_wear_at_pit = []

    for i, h in enumerate(history):
        obs = h["observation"]
        tire_type = obs.tire_type
        cliff = cliff_thresholds.get(tire_type, 0.75)

        if obs.tire_wear > cliff:
            laps_past_cliff += 1

        # Track wear level when pitting (lower = pitting too early)
        if h["info"].get("pitted") and i > 0:
            prev_wear = history[i - 1]["observation"].tire_wear
            total_wear_at_pit.append(prev_wear)

    # Laps past cliff as fraction of total: 0% past cliff = 1.0
    cliff_fraction = laps_past_cliff / max(1, len(history))
    tire_health_score = max(0.0, 1.0 - cliff_fraction * 4.0)  # 25% past cliff = 0.0

    # Pit timing: pitting at 50-80% wear is ideal
    if total_wear_at_pit:
        avg_pit_wear = sum(total_wear_at_pit) / len(total_wear_at_pit)
        if 0.45 <= avg_pit_wear <= 0.85:
            pit_timing_score = 1.0  # sweet spot
        elif avg_pit_wear < 0.25:
            pit_timing_score = 0.3  # pitting too early, wasting tire life
        else:
            pit_timing_score = 0.5  # pitting too late or just outside sweet spot
    else:
        pit_timing_score = 0.3 if pit_count == 0 else 0.5

    tire_score = 0.6 * tire_health_score + 0.4 * pit_timing_score

    # ═══════════════════════════════════════════════
    # 5. SC/VSC EXPLOITATION  (10%)
    # ═══════════════════════════════════════════════
    sc_laps = 0
    sc_pit_opportunities = 0
    sc_pits_taken = 0

    for i, h in enumerate(history):
        obs = h["observation"]
        track_status = getattr(obs, "track_status", "green")

        if track_status in ("safety_car", "vsc"):
            sc_laps += 1

            # Was this a pit opportunity? (not too late in race, tire not brand new)
            laps_left = total_laps - (i + 1)
            if laps_left > 3 and obs.tire_wear > 0.15:
                sc_pit_opportunities += 1
                if h["info"].get("pitted"):
                    sc_pits_taken += 1

    if sc_pit_opportunities > 0:
        # Did the agent take advantage of cheap pit windows?
        sc_score = min(1.0, sc_pits_taken / max(1, min(3, sc_pit_opportunities)))
    elif sc_laps == 0:
        sc_score = 0.5  # no SC occurred, neutral
    else:
        sc_score = 0.5  # SC but no real opportunity — neutral

    # ═══════════════════════════════════════════════
    # 6. CONSISTENCY / RACECRAFT  (10%)
    # ═══════════════════════════════════════════════
    # Lower position variance = more consistent racing
    positions = [h["observation"].position for h in history]
    if len(positions) >= 2:
        pos_diffs = [abs(positions[i] - positions[i - 1]) for i in range(1, len(positions))]
        avg_pos_change = sum(pos_diffs) / len(pos_diffs)
        # 0 average change = perfect consistency
        # 3+ average change = very inconsistent
        consistency_score = max(0.0, 1.0 - avg_pos_change / 3.0)
    else:
        consistency_score = 0.5

    # ═══════════════════════════════════════════════
    # WEIGHTED TOTAL
    # ═══════════════════════════════════════════════
    score = (
        0.30 * position_score
        + 0.20 * pit_score
        + 0.15 * regulation_score
        + 0.15 * tire_score
        + 0.10 * sc_score
        + 0.10 * consistency_score
    )

    print(f"\n  ┌─ GRADING BREAKDOWN ─────────────────────────────────────────")
    print(f"  │ Position:    P{final_position:>2}  →  {position_score:.2f}  (weight 30%)")
    print(f"  │ Pit Strategy: {pit_count} stops  →  {pit_score:.2f}  (weight 20%)")
    print(f"  │   └─ back-to-back pits: {consecutive_pits}, spacing: {'good' if pit_score > 0.7 else 'poor'}")
    print(f"  │ Regulation: {len(dry_compounds_used)} dry compounds {'(waived)' if is_wet_race else ''}  →  {regulation_score:.2f}  (weight 15%)")
    print(f"  │ Tire Mgmt:  {laps_past_cliff} laps past cliff  →  {tire_score:.2f}  (weight 15%)")
    print(f"  │ SC Exploit: {sc_pits_taken}/{sc_pit_opportunities} opportunities taken  →  {sc_score:.2f}  (weight 10%)")
    print(f"  │ Consistency: avg ΔP = {avg_pos_change if len(positions) >= 2 else 'N/A'}  →  {consistency_score:.2f}  (weight 10%)")
    print(f"  ├──────────────────────────────────────────────────────────────")
    print(f"  │ TOTAL SCORE: {score:.4f}")
    print(f"  └──────────────────────────────────────────────────────────────")

    return round(min(max(score, 0.0), 1.0), 4)