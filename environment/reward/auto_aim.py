"""Reward function for the automatic-aim action space."""

from typing import TYPE_CHECKING, Any, Dict, Mapping, Tuple

import numpy as np

from environment.reward.common import BattleStats, fast_win_bonus, timeout_return_penalty

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv


def calculate_auto_aim_reward(
    env: "JackalEnv",
    config: Mapping[str, Any],
    before: BattleStats,
    after: BattleStats,
) -> Tuple[float, Dict[str, Any]]:
    enemy_damage = before["enemy_health"] - after["enemy_health"]
    agent_damage = before["agent_health"] - after["agent_health"]
    enemies_killed = before["enemy_alive"] - after["enemy_alive"]
    agents_killed = before["agent_alive"] - after["agent_alive"]
    if enemies_killed > 0:
        env.has_enemy_kill = True

    approach_progress = (
        before.get("avg_nearest_enemy_dist", 0.0)
        - after.get("avg_nearest_enemy_dist", 0.0)
    )
    alive_advantage_delta = (
        after.get("alive_advantage", 0.0)
        - before.get("alive_advantage", 0.0)
    )

    reward = 0.0
    reward += enemy_damage * float(config.get("enemy_limit_scale", 0.1))
    reward -= agent_damage * float(config.get("agent_limit_scale", 0.1))
    reward += enemies_killed * float(config.get("enemy_kill_bonus", 20.0))
    reward -= agents_killed * float(config.get("agent_killed_penalty", 10.0))
    reward += approach_progress * float(config.get("approach_scale", 0.0))
    reward += alive_advantage_delta * float(config.get("alive_adv_scale", 0.0))
    reward -= float(config.get("step_penalty", 0.0))

    info: Dict[str, Any] = {
        "battle_won": False,
        "reward_mode": "auto_aim",
        "approach_progress": round(float(approach_progress), 5),
        "alive_adv_delta": round(float(alive_advantage_delta), 5),
        "no_kill_timeout": False,
        "episode_limit": False,
    }

    terminal = False
    if after["enemy_alive"] == 0:
        reward += float(config.get("win_bonus", 30.0))
        reward += fast_win_bonus(env, config)
        info["battle_won"] = True
        terminal = True
    elif after["agent_alive"] == 0:
        reward -= float(config.get("lose_penalty", 30.0))
        terminal = True
    elif env.steps >= env.max_steps:
        reward -= float(config.get("timeout_penalty", 30.0))
        if not env.has_enemy_kill:
            reward -= float(config.get("no_kill_timeout_penalty", 0.0))
            info["no_kill_timeout"] = True
        return_penalty = timeout_return_penalty(env, reward, config)
        if return_penalty > 0.0:
            reward -= return_penalty
            info["timeout_return_penalty"] = round(float(return_penalty), 5)
        info["episode_limit"] = True
        terminal = True

    if not terminal:
        clip_abs = float(config.get("reward_clip_abs", 0.0))
        if clip_abs > 0.0:
            reward = float(np.clip(reward, -clip_abs, clip_abs))
    return reward, info
