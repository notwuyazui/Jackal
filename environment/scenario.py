"""Episode map and unit construction for the Jackal environment."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Mapping, Sequence

from game.BattleWorld import BattleWorld
from game.Map.GameMap import (
    create_builtin_map,
    create_map_from_file,
    create_map_from_strings,
)
from game.Parameter import Team, UNIT_MIN_SIGHT_RATIO


@dataclass
class ScenarioConfig:
    map_name: str
    map_file: str | None
    map_data: Sequence[str] | None
    map_tile_size: int
    arena_size: tuple[float, float]
    ally_positions: Sequence[Sequence[float]]
    enemy_positions: Sequence[Sequence[float]]
    ally_unit_types: Sequence[str]
    enemy_unit_types: Sequence[str]
    enemy_use_ai: bool
    ally_unit_scales: Mapping[str, Any] = field(default_factory=dict)
    enemy_unit_scales: Mapping[str, Any] = field(default_factory=dict)
    ally_unit_type_scales: Mapping[str, Any] = field(default_factory=dict)
    enemy_unit_type_scales: Mapping[str, Any] = field(default_factory=dict)
    agent_fire_cooldown: float = 1.5
    enemy_fire_cooldown: float | None = None
    agent_fire_cooldown_by_type: Mapping[str, float] = field(default_factory=dict)
    enemy_fire_cooldown_by_type: Mapping[str, float] = field(default_factory=dict)
    projectile_overrides_by_type: Mapping[str, Any] = field(default_factory=dict)
    enemy_fire_angle_tolerance: float | None = None
    ally_initial_headings: Sequence[float] = ()
    enemy_initial_headings: Sequence[float] = ()
    sight_range: float = 400.0
    position_jitter: float = 0.0
    heading_jitter: float = 0.0


@dataclass(frozen=True)
class EpisodeScenario:
    world: BattleWorld
    allies: list[Any]
    enemies: list[Any]


def default_positions(count: int, *, enemy: bool) -> list[tuple[int, int]]:
    """Build deterministic default spawn points for one side."""

    x = 740 if enemy else 220
    return [(x, 220 + index * 100) for index in range(count)]


def normalize_unit_types(
    unit_types: Sequence[str] | None,
    count: int,
    allowed_types: Sequence[str],
) -> list[str]:
    """Normalize a per-unit type list to the requested team size."""

    if unit_types is None:
        normalized = ["tank"] * count
    else:
        normalized = [str(unit_type).lower() for unit_type in unit_types][:count]
        normalized.extend(["tank"] * (count - len(normalized)))

    unknown = sorted(set(normalized) - set(allowed_types))
    if unknown:
        raise ValueError(
            f"Unknown unit types {unknown}; expected values from {list(allowed_types)}"
        )
    return normalized


def build_episode(config: ScenarioConfig) -> EpisodeScenario:
    """Create and configure a fresh battle world without depending on JackalEnv."""

    world = BattleWorld(_create_map(config))
    allies = [
        _create_unit(world, config, Team.PLAYER, index, unit_type, enemy=False)
        for index, unit_type in enumerate(config.ally_unit_types)
    ]
    enemies = [
        _create_unit(world, config, Team.ENEMY, index, unit_type, enemy=True)
        for index, unit_type in enumerate(config.enemy_unit_types)
    ]
    world.refresh_vision()
    return EpisodeScenario(world, allies, enemies)


def _create_map(config: ScenarioConfig):
    if config.map_data:
        return create_map_from_strings(
            config.map_data,
            tile_size=config.map_tile_size,
        )
    if config.map_file:
        game_map = create_map_from_file(
            config.map_file,
            tile_size=config.map_tile_size,
        )
        if game_map is None:
            raise ValueError(f"Failed to load map_file: {config.map_file}")
        return game_map
    return create_builtin_map(config.map_name)


def _create_unit(
    world: BattleWorld,
    config: ScenarioConfig,
    team: Team,
    index: int,
    unit_type: str,
    *,
    enemy: bool,
):
    positions = config.enemy_positions if enemy else config.ally_positions
    default = (740, 220 + index * 100) if enemy else (220, 220 + index * 100)
    position = positions[index] if index < len(positions) else default
    unit = world.unit_manager.create_unit(
        unit_type,
        100 + index if enemy else index + 1,
        team,
        _jitter_position(position, config.position_jitter, config.arena_size),
        using_ai=config.enemy_use_ai if enemy else False,
    )
    unit.usingAI = config.enemy_use_ai if enemy else False

    unit.sight_range = config.sight_range
    unit.min_sight_range = UNIT_MIN_SIGHT_RATIO * unit.sight_range
    unit.communication_range = unit.sight_range
    _apply_scales(unit, config.enemy_unit_scales if enemy else config.ally_unit_scales)
    type_scales = (
        config.enemy_unit_type_scales if enemy else config.ally_unit_type_scales
    )
    _apply_scales(
        unit,
        type_scales.get(unit_type, type_scales.get(unit_type.lower(), {})) or {},
    )
    _configure_weapon(unit, config, unit_type, enemy=enemy)

    headings = config.enemy_initial_headings if enemy else config.ally_initial_headings
    if index < len(headings):
        _set_heading(unit, float(headings[index]))
    _jitter_heading(unit, config.heading_jitter)

    if enemy and config.enemy_fire_angle_tolerance is not None:
        unit.ai_fire_angle_tolerance = float(config.enemy_fire_angle_tolerance)
    world.add_unit(unit)
    return unit


def _jitter_position(
    position: Sequence[float],
    jitter: float,
    arena_size: tuple[float, float],
) -> tuple[float, float]:
    if jitter <= 0.0:
        return float(position[0]), float(position[1])
    margin = 50.0
    x = float(position[0]) + random.uniform(-jitter, jitter)
    y = float(position[1]) + random.uniform(-jitter, jitter)
    return (
        min(max(x, margin), arena_size[0] - margin),
        min(max(y, margin), arena_size[1] - margin),
    )


def _set_heading(unit, heading: float) -> None:
    heading = unit.normalize_angle(heading)
    unit.direction_angle = heading
    unit.turret_direction_angle = heading
    unit.turret_target_angle = heading
    unit.velocity = unit.cal_velocity()
    unit._update_bounding_box()


def _jitter_heading(unit, jitter: float) -> None:
    if jitter <= 0.0:
        return
    delta = random.uniform(-jitter, jitter)
    unit.direction_angle = unit.normalize_angle(unit.direction_angle + delta)
    unit.turret_direction_angle = unit.normalize_angle(
        unit.turret_direction_angle + delta
    )
    unit.turret_target_angle = unit.turret_direction_angle
    unit.velocity = unit.cal_velocity()
    unit._update_bounding_box()


def _apply_scales(unit, scales: Mapping[str, Any]) -> None:
    if not scales:
        return
    unit.max_speed *= float(scales.get("speed", 1.0))
    acceleration = float(scales.get("acceleration", 1.0))
    unit.max_acceleration *= acceleration
    unit.min_acceleration *= acceleration
    unit.max_angular_speed *= float(scales.get("turn", 1.0))
    unit.turret_angular_speed *= float(scales.get("turret_turn", 1.0))
    health = float(scales.get("health", 1.0))
    unit.max_health *= health
    unit.health = min(unit.health * health, unit.max_health)


def _configure_weapon(
    unit,
    config: ScenarioConfig,
    unit_type: str,
    *,
    enemy: bool,
) -> None:
    cooldown: float | None
    if enemy:
        if unit_type in config.enemy_fire_cooldown_by_type:
            cooldown = float(config.enemy_fire_cooldown_by_type[unit_type])
        else:
            cooldown = config.enemy_fire_cooldown
    else:
        cooldown = (
            float(config.agent_fire_cooldown_by_type[unit_type])
            if unit_type in config.agent_fire_cooldown_by_type
            else config.agent_fire_cooldown
        )
    unit.configure_weapon(
        fire_cooldown=None if cooldown is None else float(cooldown),
        projectile_overrides=config.projectile_overrides_by_type.get(unit_type, {}),
    )
