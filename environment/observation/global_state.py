"""Centralized global-state encoder for value mixing and critics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

import numpy as np

from environment.observation.feature_normalizer import (
    normalize_angular_speed,
    normalize_speed,
)
from environment.observation.map_encoder import MapFeatureEncoder
from game.BattleState import WorldSnapshot

if TYPE_CHECKING:
    from environment.observation.manager import ObservationConfig


class GlobalStateEncoder:
    """Build the fixed-size centralized state from an immutable snapshot."""

    def __init__(
        self,
        config: ObservationConfig,
        map_encoder: MapFeatureEncoder,
    ) -> None:
        self.config = config
        self.map_encoder = map_encoder

    def dimension(self) -> int:
        config = self.config
        return (
            config.n_agents * (11 + config.unit_type_dim)
            + config.n_enemies * (10 + config.unit_type_dim)
            + config.max_state_bullets * 6
            + self.map_encoder.state_dim()
            + 1
        )

    def encode(self, snapshot: WorldSnapshot) -> np.ndarray:
        config = self.config
        features: List[float] = []

        for agent in snapshot.allies:
            if agent.alive:
                features.extend([
                    1.0,
                    agent.position[0] / config.screen_width,
                    agent.position[1] / config.screen_height,
                    agent.health / agent.max_health,
                    math.cos(math.radians(agent.direction_angle)),
                    math.sin(math.radians(agent.direction_angle)),
                    math.cos(math.radians(agent.turret_direction_angle)),
                    math.sin(math.radians(agent.turret_direction_angle)),
                    agent.fire_cooldown_ratio,
                    normalize_speed(agent),
                    normalize_angular_speed(agent),
                ])
            else:
                features.extend([0.0] * 11)
            features.extend(config.unit_type_onehot(agent.unit_type))

        for enemy in snapshot.enemies:
            if enemy.alive:
                features.extend([
                    1.0,
                    enemy.position[0] / config.screen_width,
                    enemy.position[1] / config.screen_height,
                    enemy.health / enemy.max_health,
                    math.cos(math.radians(enemy.direction_angle)),
                    math.sin(math.radians(enemy.direction_angle)),
                    math.cos(math.radians(enemy.turret_direction_angle)),
                    math.sin(math.radians(enemy.turret_direction_angle)),
                    normalize_speed(enemy),
                    normalize_angular_speed(enemy),
                ])
            else:
                features.extend([0.0] * 10)
            features.extend(config.unit_type_onehot(enemy.unit_type))

        for index in range(config.max_state_bullets):
            if index >= len(snapshot.bullets):
                features.extend([0.0] * 6)
                continue
            bullet = snapshot.bullets[index]
            features.extend([
                1.0 if bullet.active else 0.0,
                bullet.position[0] / config.screen_width,
                bullet.position[1] / config.screen_height,
                bullet.velocity[0] / config.bullet_norm_speed,
                bullet.velocity[1] / config.bullet_norm_speed,
                1.0 if bullet.shooter_team.name == "PLAYER" else -1.0,
            ])

        features.extend(self.map_encoder.global_features(snapshot.game_map))
        features.append(snapshot.tick / config.max_steps)
        return np.asarray(features, dtype=np.float32)
