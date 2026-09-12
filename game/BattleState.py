"""Read-only data exchanged between the game engine and RL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from game.Parameter import Team


CombatEventType = Literal["damage", "destroyed"]
TerrainFeature = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class CombatEvent:
    """One combat fact emitted by the game engine during a world step."""

    event_type: CombatEventType
    tick: int
    source_id: Optional[int]
    source_team: Optional[Team]
    target_id: int
    target_team: Team
    amount: float = 0.0


@dataclass(frozen=True, slots=True)
class MapSnapshot:
    """Immutable terrain features needed by observation encoders."""

    tile_size: int
    width: int
    height: int
    terrain: tuple[tuple[TerrainFeature, ...], ...]

    def terrain_at(self, x: float, y: float) -> TerrainFeature:
        col = int(float(x) // self.tile_size)
        row = int(float(y) // self.tile_size)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return (0.0, 0.0, 0.0, 0.0)
        return self.terrain[row][col]


@dataclass(frozen=True, slots=True)
class BulletSnapshot:
    """Observation-facing projectile state with no mutable entity reference."""

    index: int
    projectile_id: str
    shooter_team: Team
    position: tuple[float, float]
    velocity: tuple[float, float]
    active: bool


@dataclass(frozen=True, slots=True)
class UnitSnapshot:
    """Observation-facing unit state captured at one simulation tick."""

    unit_id: int
    team: Team
    unit_type: str
    position: tuple[float, float]
    size: tuple[float, float]
    health: float
    max_health: float
    alive: bool
    direction_angle: float
    turret_direction_angle: float
    fire_cooldown_ratio: float
    speed: float
    max_speed: float
    angular_speed: float
    max_angular_speed: float
    weapon_range: float
    visible_unit_ids: frozenset[int]
    visible_bullet_indices: frozenset[int]

    def can_see_unit(self, target_id: int) -> bool:
        return target_id in self.visible_unit_ids

    def can_see_bullet(self, bullet_index: int) -> bool:
        return bullet_index in self.visible_bullet_indices


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    """Immutable boundary object consumed by observations and rewards."""

    tick: int
    elapsed_time: float
    units: tuple[UnitSnapshot, ...]
    bullets: tuple[BulletSnapshot, ...]
    game_map: MapSnapshot
    combat_events: tuple[CombatEvent, ...] = ()

    @property
    def allies(self) -> tuple[UnitSnapshot, ...]:
        return tuple(unit for unit in self.units if unit.team == Team.PLAYER)

    @property
    def enemies(self) -> tuple[UnitSnapshot, ...]:
        return tuple(unit for unit in self.units if unit.team == Team.ENEMY)
