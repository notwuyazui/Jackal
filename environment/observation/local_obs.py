"""Per-agent local observation encoder."""

import math
from typing import TYPE_CHECKING, List

import numpy as np

from environment.observation.feature_normalizer import (
    bullet_time_to_impact,
    normalize_angular_speed,
    normalize_speed,
)
from environment.observation.map_encoder import MapFeatureEncoder

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv


class LocalObservationEncoder:
    """Build fixed-size, partially observable features for every allied unit."""

    def __init__(self, env: "JackalEnv", map_encoder: MapFeatureEncoder) -> None:
        self.env = env
        self.map_encoder = map_encoder

    def dimension(self) -> int:
        env = self.env
        return (
            (10 + env.unit_type_dim)
            + (env.n_agents - 1) * (11 + env.unit_type_dim)
            + env.n_enemies * (13 + env.unit_type_dim)
            + env.max_obs_bullets * 9
            + self.map_encoder.observation_dim()
            + 1
        )

    def encode(self) -> List[np.ndarray]:
        env = self.env
        observations: List[np.ndarray] = []
        sight_range = env.obs_sight_range
        time_ratio = env.steps / env.max_steps

        for agent_id, agent in enumerate(env.agents):
            if not agent.is_alive:
                observations.append(np.zeros(self.dimension(), dtype=np.float32))
                continue

            features: List[float] = []
            features.extend([
                agent.position[0] / env.screen_width,
                agent.position[1] / env.screen_height,
                agent.health / agent.max_health,
                math.cos(math.radians(agent.direction_angle)),
                math.sin(math.radians(agent.direction_angle)),
                math.cos(math.radians(agent.turret_direction_angle)),
                math.sin(math.radians(agent.turret_direction_angle)),
                agent.fire_cooldown_ratio(),
                normalize_speed(agent),
                normalize_angular_speed(agent),
            ])
            features.extend(env._unit_type_onehot(agent))

            allies = [
                other
                for other_id, other in enumerate(env.agents)
                if other_id != agent_id
            ]
            for ally in allies:
                alive = bool(ally.is_alive)
                visible = alive and env.is_visible_to_agent(agent, ally)
                if visible:
                    rel_x = ally.position[0] - agent.position[0]
                    rel_y = ally.position[1] - agent.position[1]
                    distance = math.hypot(rel_x, rel_y) / sight_range
                    features.extend([
                        1.0,
                        1.0,
                        rel_x / sight_range,
                        rel_y / sight_range,
                        distance,
                        ally.health / ally.max_health,
                        math.cos(math.radians(ally.direction_angle)),
                        math.sin(math.radians(ally.direction_angle)),
                        normalize_speed(ally),
                        ally.fire_cooldown_ratio(),
                        1.0 if env.check_raycast_unblocked(agent, ally) else 0.0,
                    ])
                    features.extend(env._unit_type_onehot(ally))
                else:
                    features.extend([0.0, 1.0 if alive else 0.0] + [0.0] * 9)
                    features.extend(env._unit_type_onehot(ally))

            for enemy in env.enemies:
                alive = bool(enemy.is_alive)
                visible = alive and env.is_visible_to_agent(agent, enemy)
                if visible:
                    rel_x = enemy.position[0] - agent.position[0]
                    rel_y = enemy.position[1] - agent.position[1]
                    raw_distance = math.hypot(rel_x, rel_y)
                    line_of_fire, in_range, alignment = env._target_geometry_features(
                        agent,
                        enemy,
                        raw_distance,
                    )
                    features.extend([
                        1.0,
                        1.0,
                        rel_x / sight_range,
                        rel_y / sight_range,
                        raw_distance / sight_range,
                        enemy.health / enemy.max_health,
                        math.cos(math.radians(enemy.direction_angle)),
                        math.sin(math.radians(enemy.direction_angle)),
                        math.cos(math.radians(enemy.turret_direction_angle)),
                        math.sin(math.radians(enemy.turret_direction_angle)),
                        line_of_fire,
                        in_range,
                        alignment,
                    ])
                    features.extend(env._unit_type_onehot(enemy))
                else:
                    features.extend([0.0, 1.0 if alive else 0.0] + [0.0] * 11)
                    features.extend(env._unit_type_onehot(enemy))

            visible_bullets = []
            for bullet in agent.visible_bullets.bullets:
                if not env.check_raycast_unblocked(agent, bullet):
                    continue
                dx = bullet.position[0] - agent.position[0]
                dy = bullet.position[1] - agent.position[1]
                distance = math.hypot(dx, dy)
                hostile = (
                    hasattr(bullet, "shooter_team")
                    and bullet.shooter_team != agent.team
                )
                visible_bullets.append(
                    (
                        0 if hostile else 1,
                        bullet_time_to_impact(agent, bullet),
                        distance,
                        bullet,
                    )
                )

            visible_bullets.sort(key=lambda item: (item[0], item[1]))
            for slot in range(env.max_obs_bullets):
                if slot >= len(visible_bullets):
                    features.extend([0.0] * 9)
                    continue

                _, impact_time, raw_distance, bullet = visible_bullets[slot]
                rel_x = bullet.position[0] - agent.position[0]
                rel_y = bullet.position[1] - agent.position[1]
                vel_x, vel_y = getattr(bullet, "velocity", (0.0, 0.0))
                hostile = (
                    hasattr(bullet, "shooter_team")
                    and bullet.shooter_team != agent.team
                )
                features.extend([
                    1.0,
                    1.0 if getattr(bullet, "is_active", True) else 0.0,
                    rel_x / sight_range,
                    rel_y / sight_range,
                    vel_x / env.bullet_norm_speed,
                    vel_y / env.bullet_norm_speed,
                    raw_distance / sight_range,
                    1.0 if hostile else 0.0,
                    impact_time,
                ])

            features.extend(self.map_encoder.local_features(agent))
            features.append(time_ratio)
            observations.append(np.array(features, dtype=np.float32))

        return observations
