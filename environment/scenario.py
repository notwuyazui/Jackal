"""Episode map and unit construction for the Jackal environment."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Mapping, Sequence

from game.BattleWorld import BattleWorld
from game.Map.GameMap import (
    GameMap,
    create_builtin_map,
    create_map_from_file,
    create_map_from_strings,
)
from game.Parameter import DEFAULT_AI_INTELLIGENCE_LEVEL, Team
from game.Unit.UnitManager import UnitManager


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
    collision_scale: float = 1.0
    enable_unit_collision: bool | None = None
    use_tear_drop_vision: bool | None = None
    auto_communicate: bool | None = None
    enemy_ai_intelligence_level: int = DEFAULT_AI_INTELLIGENCE_LEVEL


@dataclass(frozen=True)
class EpisodeScenario:
    world: BattleWorld
    allies: list[Any]
    enemies: list[Any]


def default_positions(count: int, *, enemy: bool) -> list[tuple[int, int]]:
    """Build deterministic default spawn points for one side."""

    if count <= 0:
        return []
    x = 740 if enemy else 220
    # Preserve the original 1v1--4v4 positions. Larger teams compress the
    # vertical spacing so the final unit remains above the bottom border wall.
    spacing = min(100.0, 320.0 / max(1, count - 1))
    return [(x, round(220 + index * spacing)) for index in range(count)]


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


def build_episode(
    config: ScenarioConfig,
    game_map: GameMap | None = None,
) -> EpisodeScenario:
    """Create and configure a fresh battle world without depending on JackalEnv."""

    unit_manager = UnitManager(
        enable_unit_collision=config.enable_unit_collision,
        use_tear_drop_vision=config.use_tear_drop_vision,
        auto_communicate=config.auto_communicate,
    )
    world = BattleWorld(
        game_map
        if game_map is not None
        else create_scenario_map(
            map_name=config.map_name,
            map_file=config.map_file,
            map_data=config.map_data,
            map_tile_size=config.map_tile_size,
        ),
        unit_manager=unit_manager,
    )
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


def create_scenario_map(
    *,
    map_name: str,
    map_file: str | None,
    map_data: Sequence[str] | None,
    map_tile_size: int,
) -> GameMap:
    """Create the map described by an environment scenario."""

    if map_data:
        return create_map_from_strings(
            list(map_data),
            tile_size=map_tile_size,
        )
    if map_file:
        game_map = create_map_from_file(
            map_file,
            tile_size=map_tile_size,
        )
        if game_map is None:
            raise ValueError(f"Failed to load map_file: {map_file}")
        return game_map
    return create_builtin_map(map_name)


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
    spawn_position = _jitter_position(
        position,
        config.position_jitter,
        config.arena_size,
    )
    team_scales = config.enemy_unit_scales if enemy else config.ally_unit_scales
    type_scales = (
        config.enemy_unit_type_scales if enemy else config.ally_unit_type_scales
    )
    type_specific_scales = (
        type_scales.get(unit_type, type_scales.get(unit_type.lower(), {})) or {}
    )
    headings = config.enemy_initial_headings if enemy else config.ally_initial_headings
    initial_heading = float(headings[index]) if index < len(headings) else None
    if config.heading_jitter > 0.0:
        initial_heading = (initial_heading or 0.0) + random.uniform(
            -config.heading_jitter,
            config.heading_jitter,
        )

    cooldown = _resolve_fire_cooldown(config, unit_type, enemy=enemy)
    ai_options = (
        {"ai_intelligence_level": config.enemy_ai_intelligence_level}
        if enemy
        else {}
    )
    try:
        return world.create_unit(
            unit_type,
            team,
            spawn_position,
            unit_id=100 + index if enemy else index + 1,
            using_ai=config.enemy_use_ai if enemy else False,
            sight_range=config.sight_range,
            communication_range=config.sight_range,
            stat_scale_layers=(team_scales, type_specific_scales),
            fire_cooldown=cooldown,
            projectile_overrides=config.projectile_overrides_by_type.get(unit_type, {}),
            initial_heading=initial_heading,
            ai_fire_angle_tolerance=(
                config.enemy_fire_angle_tolerance if enemy else None
            ),
            collision_scale=config.collision_scale,
            **ai_options,
        )
    except ValueError as exc:
        side = "enemy" if enemy else "ally"
        raise ValueError(
            f"Invalid {side} spawn index {index} ({unit_type}): {exc}"
        ) from exc


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


def _resolve_fire_cooldown(
    config: ScenarioConfig,
    unit_type: str,
    *,
    enemy: bool,
) -> float | None:
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
    return None if cooldown is None else float(cooldown)
