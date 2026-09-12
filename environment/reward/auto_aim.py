"""Reward function for the automatic-aim action space."""

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from environment.reward.common import (
    BattleStats,
    CombatSummary,
    RewardContext,
    fast_win_bonus,
    timeout_return_penalty,
)


def calculate_auto_aim_reward(
    config: Mapping[str, Any],
    before: BattleStats,
    after: BattleStats,
    combat: CombatSummary,
    context: RewardContext,
) -> Tuple[float, Dict[str, Any]]:
    approach_progress = (
        before.get("avg_nearest_enemy_dist", 0.0)
        - after.get("avg_nearest_enemy_dist", 0.0)
    )
    alive_advantage_delta = (
        after.get("alive_advantage", 0.0)
        - before.get("alive_advantage", 0.0)
    )

    reward = 0.0
    reward += combat.enemy_damage * float(config.get("enemy_limit_scale", 0.1))
    reward -= combat.agent_damage * float(config.get("agent_limit_scale", 0.1))
    reward += combat.enemies_destroyed * float(config.get("enemy_kill_bonus", 20.0))
    reward -= combat.agents_destroyed * float(config.get("agent_killed_penalty", 10.0))
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
        reward += fast_win_bonus(context.step, context.max_steps, config)
        info["battle_won"] = True
        terminal = True
    elif after["agent_alive"] == 0:
        reward -= float(config.get("lose_penalty", 30.0))
        terminal = True
    elif context.step >= context.max_steps:
        reward -= float(config.get("timeout_penalty", 30.0))
        if not context.has_enemy_kill:
            reward -= float(config.get("no_kill_timeout_penalty", 0.0))
            info["no_kill_timeout"] = True
        return_penalty = timeout_return_penalty(
            context.episode_return, reward, config
        )
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
