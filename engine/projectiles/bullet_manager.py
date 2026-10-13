"""子弹创建后的注册、更新和绘制。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from collections import deque

from engine.geometry import Rect

import engine.rules as GameMode
from engine.utils import SpatialIndex

if TYPE_CHECKING:
    from engine.projectiles.base_bullet import BaseBullet
    from engine.maps.game_map import GameMap
    from engine.units.base_unit import BaseUnit
    from engine.units.unit_manager import UnitManager


class BulletManager:
    def __init__(self) -> None:
        self.bullets: list[BaseBullet] = []
        self._spatial_index = SpatialIndex[Any](64)
        self._spatial_index_valid = False
        self.serials = {}
        self.sequence = 0
        self.event_sequence = 0
        self.events = []
        self.event_history = deque(maxlen=4096)
        self.pending_events = []
        self.current_tick = 0
        self.is_stepping = False

    def serial(self, bullet):
        if not hasattr(bullet, "render_id"):
            self.sequence += 1
            bullet.render_id = self.sequence
        return bullet.render_id

    def observe_event(self, event):
        self.event_sequence += 1
        event = {"seq": self.event_sequence, "tick": self.current_tick, **event}
        for key in ("position", "velocity", "target_ids"):
            if key in event:
                event[key] = tuple(event[key])
        self.events.append(event)
        self.event_history.append(event)
        if not self.is_stepping:
            self.pending_events.append(event)

    def begin_step(self, tick):
        self.current_tick = tick
        self.events = self.pending_events
        self.pending_events = []
        self.is_stepping = True

    def end_step(self, combat_events):
        for event in combat_events:
            self.observe_event({"kind": event.event_type,
                "source_id": event.source_id, "target_id": event.target_id,
                "amount": event.amount})
        self.is_stepping = False

    def events_since(self, cursor):
        if self.event_history and cursor < self.event_history[0]["seq"] - 1:
            raise ValueError("event cursor expired; replay or reset is required")
        from engine.contracts.render import BattleEvent
        return tuple(BattleEvent(**event) for event in self.event_history if event["seq"] > cursor)

    def add_bullet(self, bullet: BaseBullet | None) -> None:
        if bullet is None:
            return
        self.bullets.append(bullet)
        self.observe_event({"kind": "fire", "bullet_id": self.serial(bullet),
                            "source_id": bullet.shooter.id,
                            "position": bullet.position, "velocity": bullet.velocity})
        self._spatial_index_valid = False
        if GameMode.BULLET_INFO_TEXT or GameMode.DEBUG_MODE:
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
        self._observed_update_order = tuple(self.bullets)
        update_order = list(self.bullets)
        if getattr(unit_manager, "current_tick", 0) % 2 == 0:
            update_order.reverse()
        inactive = [
            bullet
            for bullet in update_order
            if not bullet.update(delta_time, unit_manager, game_map)
        ]
        if inactive:
            inactive_ids = {id(bullet) for bullet in inactive}
            self.bullets = [
                bullet for bullet in self.bullets if id(bullet) not in inactive_ids
            ]
            if GameMode.BULLET_INFO_TEXT or GameMode.DEBUG_MODE:
                for bullet in inactive:
                    print(f"子弹移除: ID={bullet.id}")
        for bullet in self._observed_update_order:
            if bullet.has_collided and bullet.collided_with in ("unit", "obstacle"):
                self.observe_event({"kind": "impact", "bullet_id": self.serial(bullet),
                    "source_id": bullet.shooter.id, "position": bullet.position,
                    "target_ids": tuple(u.id for u in bullet.collided_objects),
                    "surface": bullet.collided_with})
        self.rebuild_spatial_index(game_map)

    def rebuild_spatial_index(self, game_map: GameMap) -> None:
        if self._spatial_index.cell_size != game_map.tile_size:
            self._spatial_index = SpatialIndex[Any](game_map.tile_size)
        self._spatial_index.rebuild(
            self.bullets,
            lambda bullet: bullet.bounding_box,
        )
        self._spatial_index_valid = True

    def get_candidate_bullets(self, rect: Rect, game_map: GameMap):
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
        rect = Rect(x - radius, y - radius, radius * 2, radius * 2)
        return self.get_candidate_bullets(rect, game_map)

    def get_active_count(self) -> int:
        return sum(bullet.is_active for bullet in self.bullets)

    def clear(self) -> None:
        self.bullets.clear()
        self.sequence = self.event_sequence = 0
        self.events.clear()
        self.pending_events.clear()
        self.event_history.clear()
        self._spatial_index.buckets.clear()
        self._spatial_index_valid = False

    def save(self) -> list[bool]:
        return [bullet.save() for bullet in self.bullets]
