"""子弹创建后的注册、更新和绘制。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pygame

from game.GameMode import BULLET_INFO_TEXT, DEBUG_MODE
from game.utils import SpatialIndex

if TYPE_CHECKING:
    from game.Bullet.BaseBullet import BaseBullet
    from game.Map.GameMap import GameMap
    from game.Unit.BaseUnit import BaseUnit
    from game.Unit.UnitManager import UnitManager


class BulletManager:
    def __init__(self) -> None:
        self.bullets: list[BaseBullet] = []
        self._spatial_index = SpatialIndex[Any](64)
        self._spatial_index_valid = False

    def add_bullet(self, bullet: BaseBullet | None) -> None:
        if bullet is None:
            return
        self.bullets.append(bullet)
        self._spatial_index_valid = False
        if BULLET_INFO_TEXT or DEBUG_MODE:
            print(f"子弹发射: ID={bullet.id}, 位置={bullet.position}")

    def fire(self, shooter: BaseUnit) -> BaseBullet | None:
        """让单位开火，并保证生成的子弹只注册一次。"""

        bullet = shooter.fire()
        self.add_bullet(bullet)
        return bullet

    def update(
        self,
        delta_time: float,
        unit_manager: UnitManager,
        game_map: GameMap,
    ) -> None:
        inactive = [
            bullet
            for bullet in self.bullets
            if not bullet.update(delta_time, unit_manager, game_map)
        ]
        if inactive:
            inactive_ids = {id(bullet) for bullet in inactive}
            self.bullets = [
                bullet for bullet in self.bullets if id(bullet) not in inactive_ids
            ]
            if BULLET_INFO_TEXT or DEBUG_MODE:
                for bullet in inactive:
                    print(f"子弹移除: ID={bullet.id}")
        self.rebuild_spatial_index(game_map)

    def rebuild_spatial_index(self, game_map: GameMap) -> None:
        if self._spatial_index.cell_size != game_map.tile_size:
            self._spatial_index = SpatialIndex[Any](game_map.tile_size)
        self._spatial_index.rebuild(
            self.bullets,
            lambda bullet: bullet.bounding_box,
        )
        self._spatial_index_valid = True

    def get_candidate_bullets(self, rect: pygame.Rect, game_map: GameMap):
        if not self._spatial_index_valid:
            self.rebuild_spatial_index(game_map)
        return self._spatial_index.query_rect(rect)

    def get_bullets_in_radius(
        self,
        position: tuple[float, float],
        radius: float,
        game_map: GameMap,
    ):
        x, y = position
        rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        return self.get_candidate_bullets(rect, game_map)

    def draw(self, surface, camera_offset) -> None:
        for bullet in self.bullets:
            bullet.draw(surface, camera_offset)

    def get_active_count(self) -> int:
        return sum(bullet.is_active for bullet in self.bullets)

    def clear(self) -> None:
        self.bullets.clear()
        self._spatial_index.buckets.clear()
        self._spatial_index_valid = False

    def save(self) -> list[bool]:
        return [bullet.save() for bullet in self.bullets]
