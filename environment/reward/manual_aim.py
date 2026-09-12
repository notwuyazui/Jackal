"""Reward function for the manual turret-control action space."""

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from environment.reward.common import (
    BattleStats,
    CombatSummary,
    RewardContext,
    fast_win_bonus,
    timeout_return_penalty,
)
from game.BattleState import WorldSnapshot


def calculate_manual_aim_reward(
    config: Mapping[str, Any],
    before: BattleStats,
    after: BattleStats,
    combat: CombatSummary,
    context: RewardContext,
    snapshot: WorldSnapshot,
    actions: Sequence[int],
) -> Tuple[float, Dict[str, Any]]:
    reward = 0.0
    reward += combat.enemy_damage * config["enemy_limit_scale"]
    reward -= combat.agent_damage * config["agent_limit_scale"]
    reward += combat.enemies_destroyed * config["enemy_kill_bonus"]
    reward -= combat.agents_destroyed * config["agent_killed_penalty"]

    aim_shaping = 0.0
    fire_shaping = 0.0
    for agent_id, agent in enumerate(snapshot.allies):
        if not agent.alive:
            continue
        visible_enemies = [
            enemy
            for enemy in snapshot.enemies
            if enemy.alive and agent.can_see_unit(enemy.unit_id)
        ]
        if not visible_enemies:
            continue

        closest_enemy = min(
            visible_enemies,
            key=lambda enemy: math.hypot(
                enemy.position[0] - agent.position[0],
                enemy.position[1] - agent.position[1],
            ),
        )
        dx = closest_enemy.position[0] - agent.position[0]
        dy = closest_enemy.position[1] - agent.position[1]
        target_angle = (math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0
        angle_difference = abs(
            (target_angle - agent.turret_direction_angle + 180.0) % 360.0
            - 180.0
        )

        if angle_difference <= config["aim_good_angle"]:
            aim_shaping += config["aim_good_reward"]
        elif angle_difference <= config["aim_ok_angle"]:
            aim_shaping += config["aim_ok_reward"]
        else:
            aim_shaping -= config["aim_bad_penalty"]

        if agent_id < len(actions) and actions[agent_id] == config["fire_action_id"]:
            if angle_difference <= config["fire_good_angle"]:
                fire_shaping += config["fire_good_reward"]
            else:
                fire_shaping -= config["fire_bad_penalty"]

    reward += aim_shaping
    reward += fire_shaping
    reward -= config["step_penalty"]

    info: Dict[str, Any] = {
        "battle_won": False,
        "reward_mode": "manual_aim",
        "aim_shaping": round(float(aim_shaping), 4),
        "fire_shaping": round(float(fire_shaping), 4),
        "no_kill_timeout": False,
        "episode_limit": False,
    }
    if after["enemy_alive"] == 0:
        reward += config["win_bonus"]
        reward += fast_win_bonus(context.step, context.max_steps, config)
        info["battle_won"] = True
    elif after["agent_alive"] == 0:
        reward -= config["lose_penalty"]
    elif context.step >= context.max_steps:
        reward -= config["timeout_penalty"]
        if not context.has_enemy_kill:
            reward -= config.get("no_kill_timeout_penalty", 0.0)
            info["no_kill_timeout"] = True
        return_penalty = timeout_return_penalty(
            context.episode_return, reward, config
        )
        if return_penalty > 0.0:
            reward -= return_penalty
            info["timeout_return_penalty"] = round(float(return_penalty), 5)
        info["episode_limit"] = True
    return reward, info
