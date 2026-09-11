"""Simulation boundary around the existing map, unit, and bullet managers."""

from __future__ import annotations

from typing import Any

from game.Bullet.BulletManager import BulletManager
from game.Map.GameMap import GameMap
from game.Unit.UnitManager import UnitManager


class BattleWorld:
    """Own and advance one battle simulation.

    This class deliberately wraps the existing entities and managers. It gives
    the environment and interactive game one authoritative update order without
    requiring an immediate rewrite of unit, bullet, or map implementations.
    """

    def __init__(
        self,
        game_map: GameMap,
        unit_manager: UnitManager | None = None,
        bullet_manager: BulletManager | None = None,
    ) -> None:
        if game_map is None:
            raise ValueError("BattleWorld requires a game map")

        self.game_map = game_map
        self.unit_manager = unit_manager if unit_manager is not None else UnitManager()
        self.bullet_manager = (
            bullet_manager if bullet_manager is not None else BulletManager()
        )
        self.tick = 0
        self.elapsed_time = 0.0

    def add_unit(self, unit: Any) -> None:
        self.unit_manager.add_unit(unit, self.bullet_manager, self.game_map)

    def add_bullet(self, bullet: Any) -> None:
        self.bullet_manager.add_bullet(bullet)

    def refresh_vision(self) -> None:
        """Refresh current perception without advancing simulation time."""

        for unit in self.unit_manager.units:
            if unit is not None and unit.is_alive:
                unit._update_vision(
                    self.unit_manager,
                    self.bullet_manager,
                    self.game_map,
                )

    def step(self, delta_time: float) -> None:
        """Advance map, units/AI, bullets, and final perception in that order."""

        delta_time = float(delta_time)
        if delta_time <= 0.0:
            raise ValueError("delta_time must be > 0")

        self.game_map.update(delta_time)
        self.unit_manager.update(
            delta_time,
            self.unit_manager,
            self.bullet_manager,
            self.game_map,
        )
        self.bullet_manager.update(
            delta_time,
            self.unit_manager,
            self.game_map,
        )
        self.refresh_vision()

        self.tick += 1
        self.elapsed_time += delta_time

    def clear(self) -> None:
        """Remove dynamic entities while retaining the current map and clock."""

        self.unit_manager.clear()
        self.bullet_manager.clear()
