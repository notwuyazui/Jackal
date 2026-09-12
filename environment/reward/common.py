"""Shared reward inputs derived from immutable world data."""

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

from game.BattleState import CombatEvent, WorldSnapshot
from game.Parameter import Team

BattleStats = Dict[str, float]


@dataclass(frozen=True, slots=True)
class CombatSummary:
    enemy_damage: float = 0.0
    agent_damage: float = 0.0
    enemies_destroyed: int = 0
    agents_destroyed: int = 0


@dataclass(frozen=True, slots=True)
class RewardContext:
    step: int
    max_steps: int
    episode_return: float
    has_enemy_kill: bool


def collect_battle_stats(
    snapshot: WorldSnapshot,
    arena_size: tuple[float, float],
) -> BattleStats:
    agents = snapshot.allies
    enemies = snapshot.enemies
    alive_agents = [agent for agent in agents if agent.alive]
    alive_enemies = [enemy for enemy in enemies if enemy.alive]

    if alive_agents and alive_enemies:
        nearest_sum = sum(
            min(
                math.hypot(
                    enemy.position[0] - agent.position[0],
                    enemy.position[1] - agent.position[1],
                )
                for enemy in alive_enemies
            )
            for agent in alive_agents
        )
        average_distance = nearest_sum / len(alive_agents)
        average_distance /= max(1.0, math.hypot(*arena_size))
    else:
        average_distance = 0.0

    agent_alive = len(alive_agents)
    enemy_alive = len(alive_enemies)
    return {
        "enemy_alive": float(enemy_alive),
        "agent_alive": float(agent_alive),
        "avg_nearest_enemy_dist": average_distance,
        "alive_advantage": (
            agent_alive / max(1, len(agents))
            - enemy_alive / max(1, len(enemies))
        ),
    }


def summarize_combat_events(events: Sequence[CombatEvent]) -> CombatSummary:
    enemy_damage = 0.0
    agent_damage = 0.0
    enemies_destroyed = 0
    agents_destroyed = 0
    for event in events:
        if event.event_type == "damage":
            if event.target_team == Team.ENEMY:
                enemy_damage += event.amount
            elif event.target_team == Team.PLAYER:
                agent_damage += event.amount
        elif event.event_type == "destroyed":
            if event.target_team == Team.ENEMY:
                enemies_destroyed += 1
            elif event.target_team == Team.PLAYER:
                agents_destroyed += 1
    return CombatSummary(
        enemy_damage=enemy_damage,
        agent_damage=agent_damage,
        enemies_destroyed=enemies_destroyed,
        agents_destroyed=agents_destroyed,
    )


def timeout_return_penalty(
    episode_return: float,
    current_step_reward: float,
    config: Mapping[str, Any],
) -> float:
    scale = float(config.get("timeout_return_penalty_scale", 0.0))
    if scale <= 0.0:
        return 0.0
    projected_return = episode_return + float(current_step_reward)
    return scale * max(0.0, projected_return)


def fast_win_bonus(
    step: int,
    max_steps: int,
    config: Mapping[str, Any],
) -> float:
    bonus = float(config.get("fast_win_bonus", 0.0))
    if bonus <= 0.0:
        return 0.0
    reference_steps = int(config.get("fast_win_reference_steps", 0)) or max_steps
    return bonus * max(0.0, 1.0 - step / max(1, reference_steps))
