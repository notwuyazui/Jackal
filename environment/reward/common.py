"""Reward calculations shared by automatic and manual aiming modes."""

import math
from typing import TYPE_CHECKING, Any, Dict, Mapping

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv

BattleStats = Dict[str, float]


def collect_battle_stats(env: "JackalEnv") -> BattleStats:
    enemy_alive = sum(1 for enemy in env.enemies if enemy.is_alive)
    agent_alive = sum(1 for agent in env.agents if agent.is_alive)

    alive_agents = [agent for agent in env.agents if agent.is_alive]
    alive_enemies = [enemy for enemy in env.enemies if enemy.is_alive]
    if alive_agents and alive_enemies:
        nearest_sum = 0.0
        for agent in alive_agents:
            nearest_sum += min(
                math.hypot(
                    enemy.position[0] - agent.position[0],
                    enemy.position[1] - agent.position[1],
                )
                for enemy in alive_enemies
            )
        average_distance = nearest_sum / len(alive_agents)
        average_distance /= max(
            1.0,
            math.hypot(env.screen_width, env.screen_height),
        )
    else:
        average_distance = 0.0

    alive_advantage = (
        agent_alive / max(1, env.n_agents)
        - enemy_alive / max(1, env.n_enemies)
    )
    return {
        "enemy_health": sum(enemy.health for enemy in env.enemies),
        "agent_health": sum(agent.health for agent in env.agents),
        "enemy_alive": enemy_alive,
        "agent_alive": agent_alive,
        "avg_nearest_enemy_dist": average_distance,
        "alive_advantage": alive_advantage,
    }


def timeout_return_penalty(
    env: "JackalEnv",
    current_step_reward: float,
    config: Mapping[str, Any],
) -> float:
    scale = float(config.get("timeout_return_penalty_scale", 0.0))
    if scale <= 0.0:
        return 0.0
    projected_return = env.episode_reward_so_far + float(current_step_reward)
    return scale * max(0.0, projected_return)


def fast_win_bonus(env: "JackalEnv", config: Mapping[str, Any]) -> float:
    bonus = float(config.get("fast_win_bonus", 0.0))
    if bonus <= 0.0:
        return 0.0

    reference_steps = int(config.get("fast_win_reference_steps", 0))
    if reference_steps <= 0:
        reference_steps = env.max_steps
    return bonus * max(0.0, 1.0 - env.steps / max(1, reference_steps))
