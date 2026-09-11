"""Centralized global-state encoder for value mixing and critics."""

import math
from typing import TYPE_CHECKING, List

import numpy as np

from environment.observation.feature_normalizer import (
    normalize_angular_speed,
    normalize_speed,
)
from environment.observation.map_encoder import MapFeatureEncoder

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv


class GlobalStateEncoder:
    """Build the fixed-size centralized state used during training."""

    def __init__(self, env: "JackalEnv", map_encoder: MapFeatureEncoder) -> None:
        self.env = env
        self.map_encoder = map_encoder

    def dimension(self) -> int:
        env = self.env
        return (
            env.n_agents * (11 + env.unit_type_dim)
            + env.n_enemies * (10 + env.unit_type_dim)
            + env.max_state_bullets * 6
            + self.map_encoder.state_dim()
            + 1
        )

    def encode(self) -> np.ndarray:
        env = self.env
        features: List[float] = []

        for agent_id, agent in enumerate(env.agents):
            if agent.is_alive:
                features.extend([
                    1.0,
                    agent.position[0] / env.screen_width,
                    agent.position[1] / env.screen_height,
                    agent.health / agent.max_health,
                    math.cos(math.radians(agent.direction_angle)),
                    math.sin(math.radians(agent.direction_angle)),
                    math.cos(math.radians(agent.turret_direction_angle)),
                    math.sin(math.radians(agent.turret_direction_angle)),
                    env._cooldown_ratio(agent, env.agent_fire_cooldowns[agent_id]),
                    normalize_speed(agent),
                    normalize_angular_speed(agent),
                ])
                features.extend(env._unit_type_onehot(agent))
            else:
                features.extend([0.0] * 11)
                features.extend(env._unit_type_onehot(agent))

        for enemy in env.enemies:
            if enemy.is_alive:
                features.extend([
                    1.0,
                    enemy.position[0] / env.screen_width,
                    enemy.position[1] / env.screen_height,
                    enemy.health / enemy.max_health,
                    math.cos(math.radians(enemy.direction_angle)),
                    math.sin(math.radians(enemy.direction_angle)),
                    math.cos(math.radians(enemy.turret_direction_angle)),
                    math.sin(math.radians(enemy.turret_direction_angle)),
                    normalize_speed(enemy),
                    normalize_angular_speed(enemy),
                ])
                features.extend(env._unit_type_onehot(enemy))
            else:
                features.extend([0.0] * 10)
                features.extend(env._unit_type_onehot(enemy))

        bullets = env.bullet_manager.bullets
        for index in range(env.max_state_bullets):
            if index >= len(bullets):
                features.extend([0.0] * 6)
                continue

            bullet = bullets[index]
            vel_x, vel_y = getattr(bullet, "velocity", (0.0, 0.0))
            player_team = (
                hasattr(bullet, "shooter_team")
                and bullet.shooter_team.name == "PLAYER"
            )
            features.extend([
                1.0 if getattr(bullet, "is_active", True) else 0.0,
                bullet.position[0] / env.screen_width,
                bullet.position[1] / env.screen_height,
                vel_x / env.bullet_norm_speed,
                vel_y / env.bullet_norm_speed,
                1.0 if player_team else -1.0,
            ])

        features.extend(self.map_encoder.global_features())
        features.append(env.steps / env.max_steps)
        return np.array(features, dtype=np.float32)
