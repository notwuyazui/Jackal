"""Per-agent local observation encoder."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List

import numpy as np

from environment.observation.feature_normalizer import (
    bullet_time_to_impact,
    normalize_angular_speed,
    normalize_speed,
)
from environment.observation.map_encoder import MapFeatureEncoder
from game.BattleState import UnitSnapshot, WorldSnapshot

if TYPE_CHECKING:
    from environment.observation.manager import ObservationConfig


class LocalObservationEncoder:
    """Build fixed-size, partially observable features from a world snapshot."""

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
            (10 + config.unit_type_dim)
            + (config.n_agents - 1) * (11 + config.unit_type_dim)
            + config.n_enemies * (13 + config.unit_type_dim)
            + config.max_obs_bullets * 9
            + self.map_encoder.observation_dim()
            + 1
        )

    @staticmethod
    def _target_geometry(
        observer: UnitSnapshot,
        target: UnitSnapshot,
        distance: float,
    ) -> tuple[float, float, float]:
        dx = target.position[0] - observer.position[0]
        dy = target.position[1] - observer.position[1]
        target_angle = (math.degrees(math.atan2(dy, dx)) + 90.0) % 360.0
        angle_diff = (
            target_angle - observer.turret_direction_angle + 180.0
        ) % 360.0 - 180.0
        alignment = 1.0 - min(abs(angle_diff), 180.0) / 180.0
        return 1.0, 1.0 if distance <= observer.weapon_range else 0.0, alignment

    def encode(self, snapshot: WorldSnapshot) -> List[np.ndarray]:
        config = self.config
        observations: List[np.ndarray] = []
        sight_range = config.sight_range
        time_ratio = snapshot.tick / config.max_steps
        agents = snapshot.allies
        enemies = snapshot.enemies

        for agent_id, agent in enumerate(agents):
            if not agent.alive:
                observations.append(np.zeros(self.dimension(), dtype=np.float32))
                continue

            features: List[float] = [
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
            ]
            features.extend(config.unit_type_onehot(agent.unit_type))

            for other_id, ally in enumerate(agents):
                if other_id == agent_id:
                    continue
                alive = ally.alive
                visible = alive and agent.can_see_unit(ally.unit_id)
                if visible:
                    rel_x = ally.position[0] - agent.position[0]
                    rel_y = ally.position[1] - agent.position[1]
                    features.extend([
                        1.0,
                        1.0,
                        rel_x / sight_range,
                        rel_y / sight_range,
                        math.hypot(rel_x, rel_y) / sight_range,
                        ally.health / ally.max_health,
                        math.cos(math.radians(ally.direction_angle)),
                        math.sin(math.radians(ally.direction_angle)),
                        normalize_speed(ally),
                        ally.fire_cooldown_ratio,
                        1.0,
                    ])
                else:
                    features.extend([0.0, 1.0 if alive else 0.0] + [0.0] * 9)
                features.extend(config.unit_type_onehot(ally.unit_type))

            for enemy in enemies:
                alive = enemy.alive
                visible = alive and agent.can_see_unit(enemy.unit_id)
                if visible:
                    rel_x = enemy.position[0] - agent.position[0]
                    rel_y = enemy.position[1] - agent.position[1]
                    distance = math.hypot(rel_x, rel_y)
                    line_of_fire, in_range, alignment = self._target_geometry(
                        agent, enemy, distance
                    )
                    features.extend([
                        1.0,
                        1.0,
                        rel_x / sight_range,
                        rel_y / sight_range,
                        distance / sight_range,
                        enemy.health / enemy.max_health,
                        math.cos(math.radians(enemy.direction_angle)),
                        math.sin(math.radians(enemy.direction_angle)),
                        math.cos(math.radians(enemy.turret_direction_angle)),
                        math.sin(math.radians(enemy.turret_direction_angle)),
                        line_of_fire,
                        in_range,
                        alignment,
                    ])
                else:
                    features.extend([0.0, 1.0 if alive else 0.0] + [0.0] * 11)
                features.extend(config.unit_type_onehot(enemy.unit_type))

            visible_bullets = []
            for bullet in snapshot.bullets:
                if not agent.can_see_bullet(bullet.index):
                    continue
                dx = bullet.position[0] - agent.position[0]
                dy = bullet.position[1] - agent.position[1]
                hostile = bullet.shooter_team != agent.team
                visible_bullets.append(
                    (
                        0 if hostile else 1,
                        bullet_time_to_impact(agent, bullet),
                        math.hypot(dx, dy),
                        bullet,
                    )
                )

            visible_bullets.sort(key=lambda item: (item[0], item[1]))
            for slot in range(config.max_obs_bullets):
                if slot >= len(visible_bullets):
                    features.extend([0.0] * 9)
                    continue
                _, impact_time, distance, bullet = visible_bullets[slot]
                rel_x = bullet.position[0] - agent.position[0]
                rel_y = bullet.position[1] - agent.position[1]
                features.extend([
                    1.0,
                    1.0 if bullet.active else 0.0,
                    rel_x / sight_range,
                    rel_y / sight_range,
                    bullet.velocity[0] / config.bullet_norm_speed,
                    bullet.velocity[1] / config.bullet_norm_speed,
                    distance / sight_range,
                    1.0 if bullet.shooter_team != agent.team else 0.0,
                    impact_time,
                ])

            features.extend(self.map_encoder.local_features(snapshot.game_map, agent))
            features.append(time_ratio)
            observations.append(np.asarray(features, dtype=np.float32))

        return observations
