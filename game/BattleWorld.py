"""游戏地图、单位与子弹的统一战场入口。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from game.BattleState import (
    BulletSnapshot,
    MapSnapshot,
    TerrainFeature,
    UnitSnapshot,
    WorldSnapshot,
)
from game.Bullet.BulletManager import BulletManager
from game.GameMode import DEBUG_MODE, PRINT_VISIBLE_UNIT, UNIT_RECORD_TEXT
from game.Map.GameMap import GameMap, create_builtin_map, create_empty_map
from game.Parameter import ACC, Direction, Team
from game.Unit.UnitManager import UnitManager

if TYPE_CHECKING:
    from game.Bullet.BaseBullet import BaseBullet
    from game.Unit.BaseUnit import BaseUnit
    from game.utils import Action


class BattleWorld:
    """持有并推进一场战斗，也是交互游戏和强化学习环境的共同入口。"""

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

    def draw(self, screen, camera_offset=None) -> None:
        offset = self.camera_offset if camera_offset is None else camera_offset
        self.game_map.draw(screen, offset)
        self.unit_manager.draw(screen, offset)
        self.bullet_manager.draw(screen, offset)

    def set_game_map(self, game_map: GameMap) -> None:
        if game_map is None:
            raise ValueError("Game map cannot be None")
        self.game_map = game_map
        self._map_snapshot = None
        for ai in self.unit_manager.enemy_ais:
            ai.game_map = game_map
        self.refresh_vision()

    def load_builtin_map(self, name: str) -> None:
        self.set_game_map(create_builtin_map(name))

    def add_unit(self, unit: BaseUnit) -> None:
        self.unit_manager.add_unit(unit, self.bullet_manager, self.game_map)

    def create_unit(
        self,
        unit_type: str,
        team: Team,
        position: tuple[float, float] = (0.0, 0.0),
        *,
        unit_id: int | None = None,
        using_ai: bool = False,
        visible: bool = True,
    ) -> BaseUnit:
        resolved_id = len(self.unit_manager.units) if unit_id is None else unit_id
        unit = self.unit_manager.create_unit(
            unit_type,
            resolved_id,
            team,
            position,
            using_ai=using_ai,
            visible=visible,
        )
        self.add_unit(unit)
        return unit

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

    def draw_mouse_target(self, unit_id: int, surface, mouse_pos) -> None:
        unit = self.get_unit(unit_id)
        if unit is not None:
            unit._draw_mouse_target_line(surface, self.camera_offset, mouse_pos)

    def get_active_units_counts(self) -> int:
        return self.unit_manager.get_active_count()

    def get_active_bullets_counts(self) -> int:
        return self.bullet_manager.get_active_count()

    def print_record(self) -> None:
        if self.print_record_timer <= 5.0:
            return
        self.print_record_timer = 0.0
        print(f"time: {int(self.elapsed_time)}")
        if UNIT_RECORD_TEXT or DEBUG_MODE:
            for unit in self.unit_manager.units:
                print(unit.get_record())
        if PRINT_VISIBLE_UNIT or DEBUG_MODE:
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
