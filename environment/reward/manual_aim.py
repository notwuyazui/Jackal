"""Reward function for the manual turret-control action space."""

import math
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence, Tuple

from environment.reward.common import BattleStats, fast_win_bonus, timeout_return_penalty

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv


def calculate_manual_aim_reward(
    env: "JackalEnv",
    config: Mapping[str, Any],
    before: BattleStats,
    after: BattleStats,
    actions: Sequence[int],
) -> Tuple[float, Dict[str, Any]]:
    enemy_damage = before["enemy_health"] - after["enemy_health"]
    agent_damage = before["agent_health"] - after["agent_health"]
    enemies_killed = before["enemy_alive"] - after["enemy_alive"]
    agents_killed = before["agent_alive"] - after["agent_alive"]
    if enemies_killed > 0:
        env.has_enemy_kill = True

    reward = 0.0
    reward += enemy_damage * config["enemy_limit_scale"]
    reward -= agent_damage * config["agent_limit_scale"]
    reward += enemies_killed * config["enemy_kill_bonus"]
    reward -= agents_killed * config["agent_killed_penalty"]

    aim_shaping = 0.0
    fire_shaping = 0.0
    for agent_id, agent in enumerate(env.agents):
        if not agent.is_alive:
            continue

        visible_enemies = [
            enemy
            for enemy in env.enemies
            if enemy.is_alive and env.is_visible_to_agent(agent, enemy)
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
        target_angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
        angle_difference = abs(
            agent.get_angle_difference(agent.turret_direction_angle, target_angle)
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
        reward += fast_win_bonus(env, config)
        info["battle_won"] = True
    elif after["agent_alive"] == 0:
        reward -= config["lose_penalty"]
    elif env.steps >= env.max_steps:
        reward -= config["timeout_penalty"]
        if not env.has_enemy_kill:
            reward -= config.get("no_kill_timeout_penalty", 0.0)
            info["no_kill_timeout"] = True
        return_penalty = timeout_return_penalty(env, reward, config)
        if return_penalty > 0.0:
            reward -= return_penalty
            info["timeout_return_penalty"] = round(float(return_penalty), 5)
        info["episode_limit"] = True
    return reward, info
