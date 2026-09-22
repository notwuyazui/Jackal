"""游戏地图、单位与子弹的统一战场入口。"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from game.BattleState import (
    BulletSnapshot,
    MapSnapshot,
    TerrainFeature,
    UnitSnapshot,
    WorldSnapshot,
)
from game.Bullet.BulletManager import BulletManager
from game.Bullet.weapon_specs import ProjectileSpec
import game.GameMode as GameMode
from game.Map.GameMap import GameMap, create_builtin_map, create_empty_map
from game.Parameter import (
    ACC,
    DEFAULT_AI_INTELLIGENCE_LEVEL,
    Direction,
    Team,
    UNIT_MIN_SIGHT_RATIO,
)
from game.Unit.UnitManager import UnitManager
from game.utils import centered_rect, get_class_from_str

if TYPE_CHECKING:
    from game.Bullet.BaseBullet import BaseBullet
    from game.Unit.BaseUnit import BaseUnit
    from game.utils import Action


class BattleWorld:
    """持有并推进一场战斗，也是交互游戏和强化学习环境的共同入口。"""

    GAME_STATE_FORMAT = "jackal-game-state"
    GAME_STATE_VERSION = 1
    GAME_STATE_DIRECTORY = Path(__file__).resolve().parent / "Map" / "game_states"

    def __init__(
        self,
        game_map: GameMap | None = None,
        unit_manager: UnitManager | None = None,
        bullet_manager: BulletManager | None = None,
    ) -> None:
        self.game_map = game_map if game_map is not None else create_empty_map()
        self.unit_manager = unit_manager if unit_manager is not None else UnitManager()
        self.bullet_manager = (
            bullet_manager if bullet_manager is not None else BulletManager()
        )
        self.camera_offset = [0.0, 0.0]
        self.camera_speed = 5.0
        self.camera_viewport_size = tuple(
            float(value) for value in self.game_map.get_map_size()
        )
        self.tick = 0
        self.elapsed_time = 0.0
        self.print_record_timer = 0.0
        self._map_snapshot: MapSnapshot | None = None

    def step(self, delta_time: float) -> None:
        """按地图、单位/AI、子弹、最终视野的顺序推进一帧。"""

        real_delta_time = float(delta_time)
        if real_delta_time <= 0.0:
            raise ValueError("delta_time must be > 0")
        if ACC < 0.0:
            raise ValueError("ACC must be >= 0")
        delta_time = real_delta_time * ACC

        self.unit_manager.begin_step(self.tick + 1)
        self.game_map.update(delta_time)
        self.unit_manager.update(delta_time, self.bullet_manager, self.game_map)
        self.bullet_manager.update(delta_time, self.unit_manager, self.game_map)
        self.refresh_vision(rebuild_units=False)
        self.tick += 1
        self.elapsed_time += delta_time
        self.print_record_timer += delta_time

    update = step

    def set_game_map(self, game_map: GameMap) -> None:
        if game_map is None:
            raise ValueError("Game map cannot be None")
        self.game_map = game_map
        self._map_snapshot = None
        self._clamp_camera_offset()
        for ai in self.unit_manager.enemy_ais:
            ai.game_map = game_map
        self.refresh_vision()

    def load_builtin_map(self, name: str) -> None:
        self.set_game_map(create_builtin_map(name))

    @classmethod
    def game_state_path(cls, file_name: str | Path) -> Path:
        """Resolve a state name in the project state directory.

        Plain names such as ``test_map_1`` are resolved below
        ``game/Map/game_states`` and receive the ``.json`` suffix. Explicit
        relative or absolute paths remain supported for tests and tools.
        """

        path = Path(file_name)
        if not path.suffix:
            path = path.with_suffix(".json")
        if not path.is_absolute() and path.parent == Path("."):
            path = cls.GAME_STATE_DIRECTORY / path
        return path

    def load_game_state(
        self,
        file_name: str | Path,
        *,
        using_ai_by_team: Mapping[Team, bool] | None = None,
        ai_intelligence_by_team: Mapping[Team, int] | None = None,
        position_jitter: float = 0.0,
        heading_jitter: float = 0.0,
    ) -> BattleWorld:
        """Replace this world with a versioned map/unit/bullet state.

        Unit id 0 is deliberately ignored. It is reserved for the optional
        keyboard-controlled unit and is never part of reusable environment
        initial states.
        """

        path = self.game_state_path(file_name)
        try:
            with path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Game state file does not exist: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid game state JSON in {path}: {exc}") from exc

        if not isinstance(payload, Mapping):
            raise ValueError(f"Game state root must be an object: {path}")
        if payload.get("format") != self.GAME_STATE_FORMAT:
            raise ValueError(
                f"Unsupported game state format in {path}: {payload.get('format')!r}"
            )
        if int(payload.get("version", -1)) != self.GAME_STATE_VERSION:
            raise ValueError(
                f"Unsupported game state version in {path}: {payload.get('version')!r}"
            )

        self.clear()
        self.set_game_map(self._map_from_game_state(payload.get("map"), path))
        self._load_rules(payload.get("rules", {}), path)

        world_state = payload.get("world", {})
        if not isinstance(world_state, Mapping):
            raise ValueError(f"world must be an object in {path}")
        self.tick = int(world_state.get("tick", 0))
        self.elapsed_time = float(world_state.get("elapsed_time", 0.0))
        self.print_record_timer = float(world_state.get("print_record_timer", 0.0))
        self.unit_manager.current_tick = self.tick
        self.unit_manager.unit_collision_count = int(
            world_state.get("unit_collision_count", 0)
        )
        camera_offset = world_state.get("camera_offset", (0.0, 0.0))
        self.camera_offset = [float(camera_offset[0]), float(camera_offset[1])]

        units = payload.get("units", [])
        if not isinstance(units, list):
            raise ValueError(f"units must be an array in {path}")
        seen_ids: set[int] = set()
        for unit_state in units:
            if not isinstance(unit_state, Mapping):
                raise ValueError(f"Every unit state must be an object in {path}")
            unit_id = int(unit_state.get("unit_id", -1))
            if unit_id == 0:
                continue
            if unit_id < 0 or unit_id in seen_ids:
                raise ValueError(f"Invalid or duplicate unit_id {unit_id} in {path}")
            seen_ids.add(unit_id)
            self._load_unit_state(
                unit_state,
                path,
                using_ai_by_team=using_ai_by_team,
                ai_intelligence_by_team=ai_intelligence_by_team,
                position_jitter=float(position_jitter),
                heading_jitter=float(heading_jitter),
            )

        bullets = payload.get("bullets", [])
        if not isinstance(bullets, list):
            raise ValueError(f"bullets must be an array in {path}")
        for bullet_state in bullets:
            if not isinstance(bullet_state, Mapping):
                raise ValueError(f"Every bullet state must be an object in {path}")
            # A projectile owned by the reserved keyboard unit is part of that
            # unit's transient state and is excluded for the same reason.
            if int(bullet_state.get("shooter_id", -1)) == 0:
                continue
            self._load_bullet_state(bullet_state, path)

        self._clamp_camera_offset()
        self.refresh_vision()
        return self

    def save_game_state(self, file_name: str | Path) -> Path:
        """Save the complete reusable world state, excluding unit id 0."""

        path = self.game_state_path(file_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": self.GAME_STATE_FORMAT,
            "version": self.GAME_STATE_VERSION,
            "map": {
                "tile_size": int(self.game_map.tile_size),
                "tiles": [
                    "".join(str(tile.letter) for tile in row)
                    for row in self.game_map.tiles
                ],
            },
            "rules": {
                "enable_unit_collision": self.unit_manager.enable_unit_collision,
                "use_tear_drop_vision": self.unit_manager.use_tear_drop_vision,
                "auto_communicate": self.unit_manager.auto_communicate_enabled,
            },
            "world": {
                "tick": int(self.tick),
                "elapsed_time": float(self.elapsed_time),
                "print_record_timer": float(self.print_record_timer),
                "unit_collision_count": int(
                    self.unit_manager.unit_collision_count
                ),
                "camera_offset": [
                    float(self.camera_offset[0]),
                    float(self.camera_offset[1]),
                ],
            },
            "units": [
                self._unit_game_state(unit)
                for unit in self.unit_manager.units
                if int(unit.id) != 0
            ],
            "bullets": [
                self._bullet_game_state(bullet)
                for bullet in self.bullet_manager.bullets
                if int(getattr(getattr(bullet, "shooter", None), "id", -1)) != 0
            ],
        }
        with path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        return path

    @staticmethod
    def _map_from_game_state(map_state: Any, path: Path) -> GameMap:
        if not isinstance(map_state, Mapping):
            raise ValueError(f"map must be an object in {path}")
        tile_size = int(map_state.get("tile_size", 64))
        if tile_size <= 0:
            raise ValueError(f"map.tile_size must be > 0 in {path}")
        tiles = map_state.get("tiles")
        if tiles is not None:
            if (
                not isinstance(tiles, list)
                or not tiles
                or not all(isinstance(row, str) and row for row in tiles)
                or len({len(row) for row in tiles}) != 1
            ):
                raise ValueError(f"map.tiles must be a non-empty rectangle in {path}")
            return GameMap(tiles, tile_size=tile_size)
        builtin = map_state.get("builtin")
        if builtin:
            game_map = create_builtin_map(str(builtin))
            if game_map.tile_size != tile_size:
                rows = ["".join(tile.letter for tile in row) for row in game_map.tiles]
                game_map = GameMap(rows, tile_size=tile_size)
            return game_map
        raise ValueError(f"map requires either tiles or builtin in {path}")

    def _load_rules(self, rules: Any, path: Path) -> None:
        if not isinstance(rules, Mapping):
            raise ValueError(f"rules must be an object in {path}")
        if "enable_unit_collision" in rules:
            self.unit_manager.enable_unit_collision = bool(
                rules["enable_unit_collision"]
            )
        if "use_tear_drop_vision" in rules:
            self.unit_manager.use_tear_drop_vision = bool(
                rules["use_tear_drop_vision"]
            )
        if "auto_communicate" in rules:
            self.unit_manager.auto_communicate_enabled = bool(
                rules["auto_communicate"]
            )

    @staticmethod
    def _team_from_game_state(value: Any, path: Path) -> Team:
        try:
            if isinstance(value, str):
                return Team[value.upper()]
            return Team(int(value))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid team {value!r} in {path}") from exc

    def _load_unit_state(
        self,
        state: Mapping[str, Any],
        path: Path,
        *,
        using_ai_by_team: Mapping[Team, bool] | None,
        ai_intelligence_by_team: Mapping[Team, int] | None,
        position_jitter: float,
        heading_jitter: float,
    ) -> None:
        unit_id = int(state["unit_id"])
        position = state.get("position", (0.0, 0.0))
        if not isinstance(position, Sequence) or len(position) != 2:
            raise ValueError(f"Invalid position for unit {unit_id} in {path}")
        if position_jitter < 0.0 or heading_jitter < 0.0:
            raise ValueError("Game state jitter values must be >= 0")
        position = (float(position[0]), float(position[1]))
        if position_jitter > 0.0:
            position = (
                position[0] + random.uniform(-position_jitter, position_jitter),
                position[1] + random.uniform(-position_jitter, position_jitter),
            )
        base_direction = float(state.get("direction_angle", 0.0))
        heading_offset = 0.0
        if heading_jitter > 0.0:
            heading_offset = random.uniform(-heading_jitter, heading_jitter)
        direction = base_direction + heading_offset
        team = self._team_from_game_state(state.get("team"), path)
        using_ai = bool(state.get("using_ai", False))
        if using_ai_by_team is not None and team in using_ai_by_team:
            using_ai = bool(using_ai_by_team[team])
        intelligence_level = int(
            state.get("ai_intelligence_level", DEFAULT_AI_INTELLIGENCE_LEVEL)
        )
        if (
            ai_intelligence_by_team is not None
            and team in ai_intelligence_by_team
        ):
            intelligence_level = int(ai_intelligence_by_team[team])
        unit = self.create_unit(
            str(state.get("unit_type", "tank")),
            team,
            position,
            unit_id=unit_id,
            using_ai=using_ai,
            ai_intelligence_level=intelligence_level,
            visible=bool(state.get("visible", True)),
            sight_range=(
                float(state["sight_range"])
                if "sight_range" in state
                else None
            ),
            communication_range=(
                float(state["communication_range"])
                if "communication_range" in state
                else None
            ),
            fire_cooldown=(
                float(state["fire_cooldown_override"])
                if state.get("fire_cooldown_override") is not None
                else None
            ),
            projectile_overrides=state.get("projectile_overrides", {}),
            initial_heading=direction,
            ai_fire_angle_tolerance=(
                float(state["ai_fire_angle_tolerance"])
                if "ai_fire_angle_tolerance" in state
                else None
            ),
            collision_scale=float(state.get("collision_scale", 1.0)),
        )

        for name in (
            "max_speed",
            "max_acceleration",
            "min_acceleration",
            "max_angular_speed",
            "turret_angular_speed",
            "max_health",
        ):
            if name in state:
                setattr(unit, name, float(state[name]))
        unit.health = float(state.get("health", unit.max_health))
        unit.is_alive = bool(state.get("is_alive", unit.health > 0.0))
        unit.direction_angle = unit.normalize_angle(direction)
        base_turret_direction = float(
            state.get("turret_direction_angle", base_direction)
        )
        base_turret_target = float(
            state.get("turret_target_angle", base_turret_direction)
        )
        unit.turret_direction_angle = unit.normalize_angle(
            base_turret_direction + heading_offset
        )
        unit.turret_target_angle = unit.normalize_angle(
            base_turret_target + heading_offset
        )
        unit.speed = float(state.get("speed", 0.0))
        unit.acceleration = float(state.get("acceleration", 0.0))
        unit.angular_speed = float(state.get("angular_speed", 0.0))
        unit.fire_cooldown = float(state.get("fire_cooldown", 0.0))
        ammunition = str(state.get("current_ammunition", unit.current_ammunition))
        if ammunition not in unit.ammunition_types:
            raise ValueError(
                f"Unit {unit_id} cannot use ammunition {ammunition!r} in {path}"
            )
        unit.current_ammunition = ammunition
        unit.is_switching_ammo = bool(state.get("is_switching_ammo", False))
        unit.reload_timer = float(state.get("reload_timer", 0.0))
        unit.target_ammunition = str(state.get("target_ammunition", ""))
        unit.destroy_enemy_count = int(state.get("destroy_enemy_count", 0))
        unit.damage_dealt = float(state.get("damage_dealt", 0.0))
        unit.assist_destroy_count = int(state.get("assist_destroy_count", 0))
        unit.assist_damage_dealt = float(state.get("assist_damage_dealt", 0.0))
        unit.damage_received = float(state.get("damage_received", 0.0))
        unit.potential_damage = float(state.get("potential_damage", 0.0))
        unit.killed_by = state.get("killed_by")
        unit.living_time = float(state.get("living_time", 0.0))
        unit.reward = float(state.get("reward", 0.0))
        unit.unit_collision_count = int(state.get("unit_collision_count", 0))
        unit.velocity = unit.cal_velocity()
        unit._update_bounding_box()
        unit._update_collision_box()

    def _load_bullet_state(self, state: Mapping[str, Any], path: Path) -> None:
        shooter_id = int(state.get("shooter_id", -1))
        shooter = self.get_unit(shooter_id)
        if shooter is None:
            raise ValueError(
                f"Bullet {state.get('projectile_id')!r} references missing "
                f"shooter {shooter_id} in {path}"
            )
        ammunition = str(state.get("ammunition", ""))
        bullet_class = get_class_from_str(ammunition)
        if bullet_class is None:
            raise ValueError(f"Invalid bullet ammunition {ammunition!r} in {path}")
        position = state.get("position", (0.0, 0.0))
        direction = state.get("velocity_direction", (1.0, 0.0))
        spec_state = state.get("spec", {})
        if not isinstance(spec_state, Mapping):
            raise ValueError(f"Bullet spec must be an object in {path}")
        default_spec = shooter.get_weapon_spec(ammunition)
        size_value = spec_state.get("size", default_spec.size)
        penetration_value = spec_state.get(
            "penetration",
            default_spec.penetration,
        )
        if not isinstance(size_value, Sequence) or len(size_value) != 2:
            raise ValueError(f"Bullet spec.size must contain two values in {path}")
        if (
            not isinstance(penetration_value, Sequence)
            or len(penetration_value) != 3
        ):
            raise ValueError(
                f"Bullet spec.penetration must contain three values in {path}"
            )
        spec = ProjectileSpec(
            name=ammunition,
            image_path=spec_state.get("image_path", default_spec.image_path),
            size=(float(size_value[0]), float(size_value[1])),
            lifetime=float(spec_state.get("lifetime", default_spec.lifetime)),
            speed_rate=float(spec_state.get("speed_rate", default_spec.speed_rate)),
            damage_rate=float(spec_state.get("damage_rate", default_spec.damage_rate)),
            cooldown=float(spec_state.get("cooldown", default_spec.cooldown)),
            penetration=(
                float(penetration_value[0]),
                float(penetration_value[1]),
                float(penetration_value[2]),
            ),
            is_explosive=bool(
                spec_state.get("is_explosive", default_spec.is_explosive)
            ),
            explosion_radius=float(
                spec_state.get("explosion_radius", default_spec.explosion_radius)
            ),
            explosion_damage_rate=float(
                spec_state.get(
                    "explosion_damage_rate",
                    default_spec.explosion_damage_rate,
                )
            ),
            explosion_image_path=spec_state.get(
                "explosion_image_path",
                default_spec.explosion_image_path,
            ),
        )
        bullet = bullet_class(
            projectile_id=str(state.get("projectile_id", "loaded_bullet")),
            shooter=shooter,
            shooter_team=shooter.team,
            position=(float(position[0]), float(position[1])),
            velocity_direction=(float(direction[0]), float(direction[1])),
            spec=spec,
        )
        bullet.lifetime = float(state.get("remaining_lifetime", bullet.lifetime))
        bullet.is_active = bool(state.get("is_active", True))
        bullet.has_collided = bool(state.get("has_collided", False))
        bullet.has_exploded = bool(state.get("has_exploded", False))
        bullet.explosion_timer = float(state.get("explosion_timer", 0.0))
        bullet.distance_traveled = float(state.get("distance_traveled", 0.0))
        bullet.potential_recorded_units = {
            int(unit_id)
            for unit_id in state.get("potential_recorded_unit_ids", [])
            if int(unit_id) != 0
        }
        self.add_bullet(bullet)

    @staticmethod
    def _unit_game_state(unit: BaseUnit) -> dict[str, Any]:
        return {
            "unit_id": int(unit.id),
            "unit_type": str(unit.unit_type),
            "team": unit.team.name,
            "position": [float(unit.position[0]), float(unit.position[1])],
            "using_ai": bool(unit.usingAI),
            "ai_intelligence_level": int(unit.ai_intelligence_level),
            "visible": bool(unit.visible),
            "sight_range": float(unit.sight_range),
            "communication_range": float(unit.communication_range),
            "collision_scale": float(unit.collision_scale),
            "max_speed": float(unit.max_speed),
            "max_acceleration": float(unit.max_acceleration),
            "min_acceleration": float(unit.min_acceleration),
            "max_angular_speed": float(unit.max_angular_speed),
            "turret_angular_speed": float(unit.turret_angular_speed),
            "max_health": float(unit.max_health),
            "health": float(unit.health),
            "is_alive": bool(unit.is_alive),
            "direction_angle": float(unit.direction_angle),
            "turret_direction_angle": float(unit.turret_direction_angle),
            "turret_target_angle": float(unit.turret_target_angle),
            "speed": float(unit.speed),
            "acceleration": float(unit.acceleration),
            "angular_speed": float(unit.angular_speed),
            "current_ammunition": str(unit.current_ammunition),
            "fire_cooldown": float(unit.fire_cooldown),
            "fire_cooldown_override": unit.fire_cooldown_override,
            "projectile_overrides": dict(unit.projectile_overrides),
            "ai_fire_angle_tolerance": float(unit.ai_fire_angle_tolerance),
            "is_switching_ammo": bool(unit.is_switching_ammo),
            "reload_timer": float(unit.reload_timer),
            "target_ammunition": str(unit.target_ammunition),
            "destroy_enemy_count": int(unit.destroy_enemy_count),
            "damage_dealt": float(unit.damage_dealt),
            "assist_destroy_count": int(unit.assist_destroy_count),
            "assist_damage_dealt": float(unit.assist_damage_dealt),
            "damage_received": float(unit.damage_received),
            "potential_damage": float(unit.potential_damage),
            "killed_by": unit.killed_by,
            "living_time": float(unit.living_time),
            "reward": float(unit.reward),
            "unit_collision_count": int(unit.unit_collision_count),
        }

    @staticmethod
    def _bullet_ammunition(bullet: BaseBullet) -> str:
        names = {
            "NormalShell": "normal_shell",
            "RocketShell": "rocket_shell",
            "HeavyShell": "heavy_shell",
        }
        try:
            return names[type(bullet).__name__]
        except KeyError as exc:
            raise ValueError(
                f"Cannot save unsupported bullet type {type(bullet).__name__!r}"
            ) from exc

    @classmethod
    def _bullet_game_state(cls, bullet: BaseBullet) -> dict[str, Any]:
        return {
            "projectile_id": str(bullet.id),
            "ammunition": cls._bullet_ammunition(bullet),
            "shooter_id": int(bullet.shooter.id),
            "position": [float(bullet.position[0]), float(bullet.position[1])],
            "velocity_direction": [
                float(bullet.velocity_direction[0]),
                float(bullet.velocity_direction[1]),
            ],
            "remaining_lifetime": float(bullet.lifetime),
            "is_active": bool(bullet.is_active),
            "has_collided": bool(bullet.has_collided),
            "has_exploded": bool(bullet.has_exploded),
            "explosion_timer": float(bullet.explosion_timer),
            "distance_traveled": float(bullet.distance_traveled),
            "potential_recorded_unit_ids": sorted(
                int(unit_id)
                for unit_id in bullet.potential_recorded_units
                if int(unit_id) != 0
            ),
            "spec": {
                "image_path": bullet.image_path,
                "size": [float(value) for value in bullet.size],
                "lifetime": float(bullet.max_lifetime),
                "speed_rate": float(bullet.speed_rate),
                "damage_rate": float(bullet.damage_rate),
                "cooldown": float(bullet.cooldown),
                "penetration": [float(value) for value in bullet.penetration],
                "is_explosive": bool(bullet.is_explosive),
                "explosion_radius": float(bullet.explosion_radius),
                "explosion_damage_rate": float(bullet.explosion_damage_rate),
                "explosion_image_path": bullet.explosion_image_path,
            },
        }

    def add_unit(self, unit: BaseUnit) -> None:
        placement_error = self._unit_placement_error(
            unit.position,
            unit.collision_size,
        )
        if placement_error is not None:
            raise ValueError(
                f"Unit {unit.id} cannot spawn at {unit.position}: {placement_error}"
            )
        self.unit_manager.add_unit(unit, self.bullet_manager, self.game_map)

    def _unit_placement_error(
        self,
        position: tuple[float, float],
        collision_size: tuple[float, float],
        *,
        exclude_unit: BaseUnit | None = None,
        check_unit_collision: bool = True,
    ) -> str | None:
        rect = centered_rect(position, collision_size)
        if not self.game_map.contains_rect(rect):
            return "outside map bounds"
        if self.game_map.check_collision(rect):
            return "overlaps impassable terrain"
        if check_unit_collision:
            blocker = self.unit_manager.find_unit_collision(
                rect,
                exclude_unit=exclude_unit,
            )
            if blocker is not None:
                return f"overlaps unit {blocker.id}"
        return None

    def can_place_unit(
        self,
        position: tuple[float, float],
        collision_size: tuple[float, float],
    ) -> bool:
        """查询新单位是否可在指定中心位置占用给定空间。"""

        return self._unit_placement_error(position, collision_size) is None

    def can_move_unit(
        self,
        unit_id: int,
        candidate_position: tuple[float, float],
    ) -> bool:
        """检查单位的下一个候选位置，忽略该单位自身占用区域。"""

        unit = self.get_unit(unit_id)
        if unit is None or not unit.is_alive:
            return False
        return self._unit_placement_error(
            candidate_position,
            unit.collision_size,
            exclude_unit=unit,
            check_unit_collision=self.unit_manager.enable_unit_collision,
        ) is None

    def create_unit(
        self,
        unit_type: str,
        team: Team,
        position: tuple[float, float] = (0.0, 0.0),
        *,
        unit_id: int | None = None,
        using_ai: bool = False,
        ai_intelligence_level: int = DEFAULT_AI_INTELLIGENCE_LEVEL,
        visible: bool = True,
        sight_range: float | None = None,
        communication_range: float | None = None,
        stat_scale_layers: Sequence[Mapping[str, Any]] = (),
        fire_cooldown: float | None = None,
        projectile_overrides: Mapping[str, Any] | None = None,
        initial_heading: float | None = None,
        ai_fire_angle_tolerance: float | None = None,
        collision_scale: float = 1.0,
    ) -> BaseUnit:
        """Create, configure, and register a unit through the world boundary.

        Configuration is applied before registration so an AI controller observes
        the final weapon and perception parameters during its construction.
        """

        resolved_id = len(self.unit_manager.units) if unit_id is None else unit_id
        unit = self.unit_manager.create_unit(
            unit_type,
            resolved_id,
            team,
            position,
            using_ai=using_ai,
            ai_intelligence_level=ai_intelligence_level,
            visible=visible,
        )
        if sight_range is not None:
            unit.sight_range = float(sight_range)
            unit.min_sight_range = UNIT_MIN_SIGHT_RATIO * unit.sight_range
        if communication_range is not None:
            unit.communication_range = float(communication_range)
        elif sight_range is not None:
            unit.communication_range = unit.sight_range

        for scales in stat_scale_layers:
            self._apply_unit_scales(unit, scales)

        unit.configure_collision(collision_scale)
        unit.configure_weapon(
            fire_cooldown=fire_cooldown,
            projectile_overrides=projectile_overrides,
        )
        if initial_heading is not None:
            self._set_unit_heading(unit, initial_heading)
        if ai_fire_angle_tolerance is not None:
            unit.ai_fire_angle_tolerance = float(ai_fire_angle_tolerance)

        self.add_unit(unit)
        return unit

    @staticmethod
    def _apply_unit_scales(unit: BaseUnit, scales: Mapping[str, Any]) -> None:
        """Apply one ordered layer of scenario stat multipliers."""

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

    @staticmethod
    def _set_unit_heading(unit: BaseUnit, heading: float) -> None:
        heading = unit.normalize_angle(float(heading))
        unit.direction_angle = heading
        unit.turret_direction_angle = heading
        unit.turret_target_angle = heading
        unit.velocity = unit.cal_velocity()
        unit._update_bounding_box()

    def add_bullet(self, bullet: BaseBullet | None) -> None:
        self.bullet_manager.add_bullet(bullet)

    def fire_weapon(self, shooter: BaseUnit) -> BaseBullet | None:
        return self.bullet_manager.fire(shooter)

    def has_line_of_sight(self, observer: BaseUnit, target: Any) -> bool:
        return self.unit_manager.has_line_of_sight(
            self.game_map,
            observer,
            target,
        )

    def is_visible(self, observer: BaseUnit, target: Any) -> bool:
        return self.unit_manager.is_visible(self.game_map, observer, target)

    def refresh_vision(self, *, rebuild_units: bool = True) -> None:
        self.unit_manager.refresh_vision(
            self.bullet_manager,
            self.game_map,
            rebuild_units=rebuild_units,
        )

    @staticmethod
    def _terrain_feature(tile) -> TerrainFeature:
        return (
            1.0,
            1.0 if getattr(tile, "blocks_unit", False) else 0.0,
            1.0 if getattr(tile, "blocks_bullet", False) else 0.0,
            1.0 if getattr(tile, "letter", "") == "w" else 0.0,
        )

    def snapshot(self) -> WorldSnapshot:
        """Capture the immutable state exposed beyond the game engine."""

        bullets = tuple(
            BulletSnapshot(
                index=index,
                projectile_id=str(bullet.id),
                shooter_team=bullet.shooter_team,
                position=(float(bullet.position[0]), float(bullet.position[1])),
                velocity=(float(bullet.velocity[0]), float(bullet.velocity[1])),
                active=bool(bullet.is_active),
            )
            for index, bullet in enumerate(self.bullet_manager.bullets)
        )

        units = []
        for unit in self.unit_manager.units:
            visible_unit_ids = frozenset(
                target.id
                for target in self.unit_manager.units
                if target is not unit and self.is_visible(unit, target)
            )
            visible_bullet_indices = frozenset(
                index
                for index, bullet in enumerate(self.bullet_manager.bullets)
                if self.is_visible(unit, bullet)
            )
            units.append(
                UnitSnapshot(
                    unit_id=int(unit.id),
                    team=unit.team,
                    unit_type=str(unit.unit_type),
                    position=(float(unit.position[0]), float(unit.position[1])),
                    size=(float(unit.size[0]), float(unit.size[1])),
                    health=float(unit.health),
                    max_health=float(unit.max_health),
                    alive=bool(unit.is_alive),
                    direction_angle=float(unit.direction_angle),
                    turret_direction_angle=float(unit.turret_direction_angle),
                    fire_cooldown_ratio=float(unit.fire_cooldown_ratio()),
                    speed=float(unit.speed),
                    max_speed=float(unit.max_speed),
                    angular_speed=float(unit.angular_speed),
                    max_angular_speed=float(unit.max_angular_speed),
                    weapon_range=float(unit.weapon_range()),
                    visible_unit_ids=visible_unit_ids,
                    visible_bullet_indices=visible_bullet_indices,
                    blocked_by_unit=bool(unit.blocked_by_unit),
                    unit_collision_count=int(unit.unit_collision_count),
                    collision_size=(
                        float(unit.collision_size[0]),
                        float(unit.collision_size[1]),
                    ),
                )
            )

        if self._map_snapshot is None:
            terrain = tuple(
                tuple(self._terrain_feature(tile) for tile in row)
                for row in self.game_map.tiles
            )
            self._map_snapshot = MapSnapshot(
                tile_size=int(self.game_map.tile_size),
                width=int(self.game_map.width),
                height=int(self.game_map.height),
                terrain=terrain,
            )
        return WorldSnapshot(
            tick=self.tick,
            elapsed_time=self.elapsed_time,
            units=tuple(units),
            bullets=bullets,
            game_map=self._map_snapshot,
            combat_events=tuple(self.unit_manager.combat_events),
        )

    def get_unit(self, unit_id: int) -> BaseUnit | None:
        return self.unit_manager.get_unit_by_id(unit_id)

    def can_unit_fire(self, unit_id: int) -> bool:
        unit = self.get_unit(unit_id)
        return bool(unit is not None and unit.can_fire())

    def set_unit_chassis(self, unit_id: int, action: Sequence[bool]) -> bool:
        """Apply forward/backward/left/right controls to one unit."""

        if len(action) != 4:
            raise ValueError("A chassis action must contain four boolean values")
        forward, backward, left, right = action
        moved = self.set_unit_movement(unit_id, bool(forward), bool(backward))
        turned = self.set_unit_turning(unit_id, bool(left), bool(right))
        return moved and turned

    def set_unit_movement(
        self,
        unit_id: int,
        forward: bool = False,
        backward: bool = False,
    ) -> bool:
        unit = self.get_unit(unit_id)
        return False if unit is None else unit.set_movement(forward, backward)

    def set_unit_turning(
        self,
        unit_id: int,
        left: bool = False,
        right: bool = False,
    ) -> bool:
        unit = self.get_unit(unit_id)
        return False if unit is None else unit.set_turning(left, right)

    def set_unit_turret_target_to_mouse(
        self,
        unit_id: int,
        mouse_pos,
        camera_offset=None,
    ) -> bool:
        unit = self.get_unit(unit_id)
        offset = self.camera_offset if camera_offset is None else camera_offset
        return False if unit is None else unit.set_turret_target_to_mouse(mouse_pos, offset)

    def set_unit_turret_target_angle(self, unit_id: int, angle: float) -> bool:
        unit = self.get_unit(unit_id)
        if unit is None:
            return False
        unit.turret_target_angle = float(angle)
        return True

    def set_unit_fire(self, unit_id: int) -> BaseBullet | None:
        unit = self.get_unit(unit_id)
        return None if unit is None else self.fire_weapon(unit)

    def set_unit_switch_ammo(self, unit_id: int) -> bool:
        unit = self.get_unit(unit_id)
        return False if unit is None else unit.switch_ammunition()

    def set_unit_action(self, unit_id: int, action: Action) -> None:
        self.set_unit_movement(unit_id, action.forward, action.backward)
        self.set_unit_turning(unit_id, action.left, action.right)
        self.set_unit_turret_target_to_mouse(unit_id, action.mouse_pos)
        if action.fire:
            self.set_unit_fire(unit_id)
        if action.switch_ammo:
            self.set_unit_switch_ammo(unit_id)

    def communicate(self, source_id: int, target_id: int):
        source = self.get_unit(source_id)
        target = self.get_unit(target_id)
        if source is None or target is None:
            return False, False, False, False
        return source.communicate_to(target)

    def broadcast(self, unit_id: int):
        unit = self.get_unit(unit_id)
        if unit is None:
            return False, False, False, False
        return unit.broadcast(self.unit_manager)

    def receive(self, unit_id: int, source_id: int):
        unit = self.get_unit(unit_id)
        source = self.get_unit(source_id)
        if unit is None or source is None:
            return False, False, False, False
        return unit.receive_from(source)

    def broadcast_receive(self, unit_id: int):
        unit = self.get_unit(unit_id)
        if unit is None:
            return False, False, False, False
        return unit.broadcast_receive(self.unit_manager)

    def move_camera(self, direction: Direction) -> None:
        offsets = {
            Direction.UP: (0.0, -self.camera_speed),
            Direction.DOWN: (0.0, self.camera_speed),
            Direction.LEFT: (-self.camera_speed, 0.0),
            Direction.RIGHT: (self.camera_speed, 0.0),
        }
        dx, dy = offsets[direction]
        self.camera_offset[0] += dx
        self.camera_offset[1] += dy
        self._clamp_camera_offset()

    def set_camera_viewport(self, viewport_size: tuple[float, float]) -> None:
        """Set the visible camera area without changing the world dimensions."""

        width, height = float(viewport_size[0]), float(viewport_size[1])
        if width <= 0.0 or height <= 0.0:
            raise ValueError("camera viewport dimensions must be > 0")
        self.camera_viewport_size = (width, height)
        self._clamp_camera_offset()

    def _clamp_camera_offset(self) -> None:
        world_width, world_height = self.game_map.get_map_size()
        viewport_width, viewport_height = self.camera_viewport_size
        max_x = max(0.0, float(world_width) - viewport_width)
        max_y = max(0.0, float(world_height) - viewport_height)
        self.camera_offset[0] = min(max(0.0, self.camera_offset[0]), max_x)
        self.camera_offset[1] = min(max(0.0, self.camera_offset[1]), max_y)

    def get_active_units_counts(self) -> int:
        return self.unit_manager.get_active_count()

    def get_active_bullets_counts(self) -> int:
        return self.bullet_manager.get_active_count()

    def iter_units(self):
        """Return a stable collection view for render and inspection adapters."""

        return tuple(self.unit_manager.units)

    def iter_bullets(self):
        return tuple(self.bullet_manager.bullets)

    def iter_map_tiles(self):
        return tuple(tuple(row) for row in self.game_map.tiles)

    def get_unit_obstacles(self):
        return tuple(self.game_map.unit_obstacles)

    def get_bullet_obstacles(self):
        return tuple(self.game_map.bullet_obstacles)

    def get_camera_offset(self) -> tuple[float, float]:
        self._clamp_camera_offset()
        return float(self.camera_offset[0]), float(self.camera_offset[1])

    def print_record(self) -> None:
        if self.print_record_timer <= 5.0:
            return
        self.print_record_timer = 0.0
        print(f"time: {int(self.elapsed_time)}")
        if GameMode.UNIT_RECORD_TEXT or GameMode.DEBUG_MODE:
            for unit in self.unit_manager.units:
                print(unit.get_record())
        if GameMode.PRINT_VISIBLE_UNIT or GameMode.DEBUG_MODE:
            unit = self.get_unit(0)
            if unit is not None:
                print(f"Unit 0 可见的单位ID: {unit.get_visible_unit_ids()}")

    def clear_bullets(self) -> None:
        self.bullet_manager.clear()

    def clear_units(self) -> None:
        self.unit_manager.clear()

    def clear(self) -> None:
        self.clear_units()
        self.clear_bullets()

    def save_map(self):
        return self.game_map.save()

    def save_unit(self):
        return self.unit_manager.save()

    def save_bullet(self):
        return self.bullet_manager.save()

    def save(self) -> dict[str, Any]:
        return {
            "map": self.save_map(),
            "units": self.save_unit(),
            "bullets": self.save_bullet(),
        }
