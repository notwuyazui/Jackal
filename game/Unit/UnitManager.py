"""单位创建、生命周期管理和视野更新。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pygame

from game.GameMode import AUTO_COMMUNICATE
from game.BattleState import CombatEvent
from game.Parameter import Team
from game.utils import SpatialIndex

if TYPE_CHECKING:
    from game.Bullet.BulletManager import BulletManager
    from game.Map.GameMap import GameMap
    from game.Unit.BaseUnit import BaseUnit


class UnitManager:
    def __init__(self) -> None:
        self.units: list[BaseUnit] = []
        self.enemy_ais: list[Any] = []
        self._unit_spatial_index = SpatialIndex[Any](64)
        self._unit_index_valid = False
        self._visibility_cache: dict[tuple[int, int], bool] = {}
        self._line_of_sight_cache: dict[tuple[int, int], bool] = {}
        self.combat_events: list[CombatEvent] = []
        self.current_tick = 0

    def begin_step(self, tick: int) -> None:
        """Start a new event batch for one authoritative world step."""

        self.current_tick = int(tick)
        self.combat_events.clear()

    def record_damage(self, source, target: BaseUnit, amount: float, destroyed: bool) -> None:
        """Record damage where it is applied so reward code need not infer it."""

        source_id = getattr(source, "id", None)
        source_team = getattr(source, "team", None)
        self.combat_events.append(
            CombatEvent(
                event_type="damage",
                tick=self.current_tick,
                source_id=source_id,
                source_team=source_team,
                target_id=target.id,
                target_team=target.team,
                amount=float(amount),
            )
        )
        if destroyed:
            self.combat_events.append(
                CombatEvent(
                    event_type="destroyed",
                    tick=self.current_tick,
                    source_id=source_id,
                    source_team=source_team,
                    target_id=target.id,
                    target_team=target.team,
                )
            )

    @staticmethod
    def create_unit(
        unit_type: str,
        unit_id: int,
        team: Team,
        position: tuple[float, float] = (0.0, 0.0),
        *,
        using_ai: bool = False,
        visible: bool = True,
    ) -> BaseUnit:
        """创建一个单位；调用方完成配置后再将其加入战场。"""

        # 延迟导入可避免 BaseUnit 初始化可见单位容器时形成循环依赖。
        from game.Unit.Archie.Archie import create_archie
        from game.Unit.Plane.Plane import create_plane
        from game.Unit.Tank.Tank import create_tank

        builders = {
            "archie": create_archie,
            "plane": create_plane,
            "tank": create_tank,
        }
        normalized_type = str(unit_type).lower()
        try:
            builder = builders[normalized_type]
        except KeyError as exc:
            available = ", ".join(sorted(builders))
            raise ValueError(
                f"Unsupported unit_type={unit_type!r}; available types: {available}"
            ) from exc

        unit = builder(
            int(unit_id),
            team,
            position=(float(position[0]), float(position[1])),
            usingAI=bool(using_ai),
            visible=bool(visible),
        )
        if unit is None:
            raise RuntimeError(f"Unit builder returned None for {normalized_type!r}")
        unit._update_bounding_box()
        return unit

    def add_unit(
        self,
        unit: BaseUnit,
        bullet_manager: BulletManager,
        game_map: GameMap,
    ) -> None:
        """添加单位，并为启用 AI 的单位创建控制器。"""

        from game.Unit.EnemyAI import EnemyAI

        if unit is None:
            return
        self.units.append(unit)
        self._unit_index_valid = False
        self.invalidate_perception_cache()
        if unit.usingAI:
            self.enemy_ais.append(EnemyAI(unit, self, bullet_manager, game_map))

    def update(
        self,
        delta_time: float,
        bullet_manager: BulletManager,
        game_map: GameMap,
    ) -> None:
        for ai in self.enemy_ais:
            ai.update()
        for unit in self.units:
            unit.update(delta_time, self, bullet_manager, game_map)

        if AUTO_COMMUNICATE:
            self.auto_communicate()

        dead_units = {unit for unit in self.units if not unit.is_alive}
        if dead_units:
            self.enemy_ais = [ai for ai in self.enemy_ais if ai.unit not in dead_units]
        self.rebuild_unit_spatial_index(game_map)

    @staticmethod
    def _entity_rect(entity) -> pygame.Rect:
        if entity.bounding_box is not None:
            return entity.bounding_box
        x, y = entity.position
        return pygame.Rect(int(x), int(y), 1, 1)

    def rebuild_unit_spatial_index(self, game_map: GameMap) -> None:
        if self._unit_spatial_index.cell_size != game_map.tile_size:
            self._unit_spatial_index = SpatialIndex[Any](game_map.tile_size)
        self._unit_spatial_index.rebuild(self.units, self._entity_rect)
        self._unit_index_valid = True

    def get_candidate_units(self, rect: pygame.Rect):
        if not self._unit_index_valid:
            return self.units
        return self._unit_spatial_index.query_rect(rect)

    def get_units_in_radius(
        self,
        position: tuple[float, float],
        radius: float,
    ):
        x, y = position
        rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        return self.get_candidate_units(rect)

    def invalidate_perception_cache(self) -> None:
        self._visibility_cache.clear()
        self._line_of_sight_cache.clear()

    @staticmethod
    def is_in_view(observer: BaseUnit, target: Any) -> bool:
        """执行不含地图遮挡的快速视野判断。"""

        return (
            getattr(observer, "is_alive", True)
            and getattr(target, "is_alive", True)
            and getattr(target, "is_active", True)
            and getattr(target, "visible", True)
            and not getattr(target, "conceal", False)
            and observer.is_in_sight(target)
        )

    def is_visible(
        self,
        game_map: GameMap,
        observer: BaseUnit,
        target: Any,
    ) -> bool:
        key = (id(observer), id(target))
        if key not in self._visibility_cache:
            self._visibility_cache[key] = self.is_in_view(
                observer,
                target,
            ) and self.has_line_of_sight(game_map, observer, target)
        return self._visibility_cache[key]

    def has_line_of_sight(
        self,
        game_map: GameMap,
        observer: BaseUnit,
        target: Any,
    ) -> bool:
        key = (id(observer), id(target))
        if key not in self._line_of_sight_cache:
            self._line_of_sight_cache[key] = game_map.has_line_of_sight(
                observer.position,
                target.position,
            )
        return self._line_of_sight_cache[key]

    def refresh_unit_vision(
        self,
        unit: BaseUnit,
        bullet_manager: BulletManager,
        game_map: GameMap,
    ) -> None:
        unit.visible_map = game_map
        unit.visible_units.clear()
        unit.visible_bullets.clear()
        sight_range = float(unit.sight_range)
        unit_candidates = self.get_units_in_radius(
            unit.position,
            sight_range,
        )
        bullet_candidates = bullet_manager.get_bullets_in_radius(
            unit.position,
            sight_range,
            game_map,
        )
        unit.visible_units.units.extend(
            target for target in unit_candidates if self.is_in_view(unit, target)
        )
        unit.visible_bullets.bullets.extend(
            bullet
            for bullet in bullet_candidates
            if self.is_in_view(unit, bullet)
        )

    def refresh_vision(
        self,
        bullet_manager: BulletManager,
        game_map: GameMap,
        *,
        rebuild_units: bool = True,
    ) -> None:
        if rebuild_units:
            self.rebuild_unit_spatial_index(game_map)
        self.invalidate_perception_cache()
        for unit in self.units:
            if unit.is_alive:
                self.refresh_unit_vision(unit, bullet_manager, game_map)

    def auto_communicate(self) -> None:
        """反复广播可见信息，直到没有友军获得新信息。"""

        changed = True
        while changed:
            changed = False
            for unit in (unit for unit in self.units if unit.is_alive):
                if unit.broadcast(self)[3]:
                    changed = True

    def draw(self, surface, camera_offset, mouse_pos=None) -> None:
        for unit in self.units:
            unit.draw(surface, camera_offset, mouse_pos)

    def get_unit_by_id(self, unit_id: int) -> BaseUnit | None:
        return next((unit for unit in self.units if unit.id == unit_id), None)

    def get_active_count(self) -> int:
        return sum(unit.is_alive for unit in self.units)

    def is_in(self, unit_id: int) -> bool:
        return any(unit.id == unit_id for unit in self.units)

    def clear(self) -> None:
        self.units.clear()
        self.enemy_ais.clear()
        self._unit_spatial_index.buckets.clear()
        self._unit_index_valid = False
        self.invalidate_perception_cache()
        self.combat_events.clear()

    def save(self) -> list[bool]:
        return [unit.save() for unit in self.units]
