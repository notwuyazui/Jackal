"""Default reward parameters and recursive configuration merging."""

from typing import Any, Dict, Mapping, MutableMapping


def default_reward_config() -> Dict[str, Dict[str, float]]:
    """Return a new reward configuration so environments never share mutation."""
    return {
        "auto_aim": {
            "enemy_limit_scale": 0.1,
            "agent_limit_scale": 0.1,
            "enemy_kill_bonus": 20.0,
            "agent_killed_penalty": 10.0,
            "approach_scale": 6.0,
            "alive_adv_scale": 8.0,
            "step_penalty": 0.01,
            "reward_clip_abs": 5.0,
            "win_bonus": 30.0,
            "fast_win_bonus": 0.0,
            "fast_win_reference_steps": 0,
            "lose_penalty": 30.0,
            "timeout_penalty": 30.0,
            "no_kill_timeout_penalty": 0.0,
            "timeout_return_penalty_scale": 0.0,
        },
        "manual_aim": {
            "enemy_limit_scale": 0.12,
            "agent_limit_scale": 0.12,
            "enemy_kill_bonus": 20.0,
            "agent_killed_penalty": 12.0,
            "aim_good_angle": 8.0,
            "aim_ok_angle": 20.0,
            "aim_good_reward": 0.04,
            "aim_ok_reward": 0.015,
            "aim_bad_penalty": 0.01,
            "fire_good_angle": 12.0,
            "fire_good_reward": 0.08,
            "fire_bad_penalty": 0.06,
            "step_penalty": 0.01,
            "fire_action_id": 27,
            "win_bonus": 30.0,
            "fast_win_bonus": 0.0,
            "fast_win_reference_steps": 0,
            "lose_penalty": 30.0,
            "timeout_penalty": 30.0,
            "no_kill_timeout_penalty": 0.0,
            "timeout_return_penalty_scale": 0.0,
        },
    }


def merge_reward_config(
    base: MutableMapping[str, Any],
    update: Mapping[str, Any],
) -> None:
    """Recursively apply an experiment override to a reward configuration."""
    for key, value in update.items():
        current = base.get(key)
        if isinstance(value, Mapping) and isinstance(current, MutableMapping):
            merge_reward_config(current, value)
        else:
            base[key] = value
