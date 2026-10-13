"""Immutable presentation contract shared by 2D, 3D and replay consumers.

Coordinates and rectangles are Jackal pixels, time is seconds, headings are
degrees (zero points up). Consumers cannot change engine state through these
records. Rendering scale, models, textures and cameras belong to renderers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Protocol, Sequence

if TYPE_CHECKING:
    from engine.battle_world import BattleWorld
    from engine.geometry import Rect

Vector2 = tuple[float, float]
Rectangle = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class Rules:
    unit_collision: bool
    tear_drop_vision: bool
    auto_communicate: bool


@dataclass(frozen=True, slots=True)
class MapFrame:
    size: Vector2
    tile_size: int
    tiles: tuple[str, ...]
    unit_obstacles: tuple[Rectangle, ...]
    bullet_obstacles: tuple[Rectangle, ...]


@dataclass(frozen=True, slots=True)
class UnitFrame:
    id: int
    team: str
    unit_type: str
    position: Vector2
    heading: float
    turret_heading: float
    speed: float
    angular_speed: float
    acceleration: float
    health: float
    max_health: float
    alive: bool
    cooldown: float
    cooldown_duration: float
    hitbox: Rectangle
    collision_box: Rectangle
    size: Vector2
    visible_ids: tuple[int, ...]
    kills: int
    damage_dealt: float
    visible: bool
    sight_range: float
    min_sight_range: float
    attack_range: float


@dataclass(frozen=True, slots=True)
class BulletFrame:
    id: int
    source_id: int
    team: str
    ammunition: str
    position: Vector2
    velocity: Vector2
    size: Vector2
    lifetime: float
    rotation_angle: float
    has_exploded: bool
    is_explosive: bool
    explosion_radius: float
    hitbox: Rectangle


@dataclass(frozen=True, slots=True)
class BattleEvent:
    seq: int
    tick: int
    kind: str
    source_id: int | None = None
    target_id: int | None = None
    bullet_id: int | None = None
    position: Vector2 | None = None
    velocity: Vector2 | None = None
    target_ids: tuple[int, ...] | None = None
    surface: str | None = None
    amount: float | None = None


@dataclass(frozen=True, slots=True)
class Parameters:
    id: int
    unit_type: str
    size: Vector2
    collision_size: Vector2
    max_speed: float
    max_acceleration: float
    min_acceleration: float
    hull_deg_s: float
    turret_deg_s: float
    max_health: float
    sight_range: float
    communication_range: float
    cooldown: float
    shell_speed: float
    shell_lifetime: float
    shell_range: float
    shell_size: Vector2
    shell_base_damage: float
    armor_damage_multipliers: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BattleManifest:
    protocol: str
    dt: float
    map: MapFrame
    rules: Rules
    units: tuple[Parameters, ...]


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    tick: int
    time: float
    units: tuple[UnitFrame, ...]
    bullets: tuple[BulletFrame, ...]
    events: tuple[BattleEvent, ...]
    event_cursor: int
    map: MapFrame
    rules: Rules

    def to_dict(self) -> dict:
        return asdict(self)


class Renderer(Protocol):
    def initialize(self, manifest: BattleManifest) -> None: ...
    def render(self, frame: FrameSnapshot): ...
    def close(self) -> None: ...


def map_frame(world: BattleWorld) -> MapFrame:
    return MapFrame(_vector2(world.game_map.get_map_size()), world.game_map.tile_size,
                    tuple(world.game_map.to_strings()),
                    tuple(_rectangle(r) for r in world.get_unit_obstacles()),
                    tuple(_rectangle(r) for r in world.get_bullet_obstacles()))


def _vector2(value: Sequence[float]) -> Vector2:
    return (value[0], value[1])


def _rectangle(value: Rect) -> Rectangle:
    return (value.x, value.y, value.width, value.height)


def rules_frame(world: BattleWorld) -> Rules:
    manager = world.unit_manager
    return Rules(manager.enable_unit_collision, manager.use_tear_drop_vision,
                 manager.auto_communicate_enabled)


def capture_frame(world: BattleWorld) -> FrameSnapshot:
    manager = world.bullet_manager
    units = tuple(UnitFrame(
        u.id, u.team.name, u.unit_type, _vector2(u.position), u.direction_angle,
        u.turret_direction_angle, u.speed, u.angular_speed, u.acceleration,
        u.health, u.max_health, u.is_alive, u.fire_cooldown,
        u.fire_cooldown_duration(), _rectangle(u.bounding_box), _rectangle(u.collision_box),
        _vector2(u.size), tuple(sorted(u.get_visible_unit_ids())),
        u.destroy_enemy_count, u.damage_dealt, u.visible, u.sight_range,
        u.min_sight_range, u.weapon_range(),
    ) for u in world.iter_units())
    bullets = tuple(BulletFrame(
        b.render_id, b.shooter.id, b.shooter_team.name,
        world._bullet_ammunition(b), _vector2(b.position), _vector2(b.velocity),
        _vector2(b.size), b.lifetime, b.rotation_angle, b.has_exploded,
        b.is_explosive, b.explosion_radius, _rectangle(b.bounding_box),
    ) for b in world.iter_bullets() if b.is_active)
    events = tuple(BattleEvent(**e) for e in manager.events)
    return FrameSnapshot(world.tick, world.elapsed_time, units, bullets, events,
                         manager.event_sequence, map_frame(world), rules_frame(world))


def build_manifest(world: BattleWorld, dt: float) -> BattleManifest:
    from engine.parameters import BULLET_DAMAGE
    parameters = []
    for u in world.iter_units():
        weapon = u.get_weapon_spec() if u.current_ammunition else None
        parameters.append(Parameters(
            u.id, u.unit_type, _vector2(u.size), _vector2(u.collision_size),
            u.max_speed, u.max_acceleration, u.min_acceleration,
            u.max_angular_speed, u.turret_angular_speed, u.max_health,
            u.sight_range, u.communication_range, u.fire_cooldown_duration(weapon),
            weapon.speed if weapon else 0., weapon.lifetime if weapon else 0.,
            weapon.max_range if weapon else 0., weapon.size if weapon else (0., 0.),
            BULLET_DAMAGE * weapon.damage_rate if weapon else 0.,
            weapon.penetration if weapon else (0., 0., 0.),
        ))
    return BattleManifest('jackal-render/1', dt, map_frame(world), rules_frame(world),
                          tuple(parameters))
