"""单位创建、生命周期管理和视野更新。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from game.GameMode import AUTO_COMMUNICATE
from game.Parameter import Team

if TYPE_CHECKING:
    from game.Bullet.BulletManager import BulletManager
    from game.Map.GameMap import GameMap
    from game.Unit.BaseUnit import BaseUnit


class UnitManager:
    def __init__(self) -> None:
        self.units: list[BaseUnit] = []
        self.enemy_ais: list[Any] = []

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
        return self.is_in_view(observer, target) and game_map.has_line_of_sight(
            observer.position,
            target.position,
        )

    def refresh_unit_vision(
        self,
        unit: BaseUnit,
        bullet_manager: BulletManager,
        game_map: GameMap,
    ) -> None:
        unit.visible_map = game_map
        unit.visible_units.clear()
        unit.visible_bullets.clear()
        unit.visible_units.units.extend(
            target for target in self.units if self.is_in_view(unit, target)
        )
        unit.visible_bullets.bullets.extend(
            bullet
            for bullet in bullet_manager.bullets
            if self.is_in_view(unit, bullet)
        )

    def refresh_vision(
        self,
        bullet_manager: BulletManager,
        game_map: GameMap,
    ) -> None:
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

    def save(self) -> list[bool]:
        return [unit.save() for unit in self.units]
