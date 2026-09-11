import math
import os
import sys
import unittest
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

from game.BattleWorld import BattleWorld
from game.Bullet.BulletManager import BulletManager
from game.Map.GameMap import GameMap
from game.Parameter import Team
from game.Unit.UnitManager import UnitManager


class _RecordingMap:
    bullet_obstacles: list[Any] = []

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def update(self, delta_time: float) -> None:
        self.calls.append(f"map:{delta_time}")


class _RecordingUnitManager:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.units = []
        self.enemy_ais = []

    def update(self, delta_time, bullet_manager, game_map) -> None:
        self.calls.append(f"units:{delta_time}")

    def refresh_vision(self, bullet_manager, game_map) -> None:
        self.calls.append("vision")


class _RecordingBulletManager:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.bullets = []

    def update(self, delta_time, unit_manager, game_map) -> None:
        self.calls.append(f"bullets:{delta_time}")


class _Target:
    def __init__(self, position=(100.0, 0.0)) -> None:
        self.position = position
        self.is_alive = True
        self.is_active = True
        self.visible = True
        self.conceal = False


class _Observer(_Target):
    def __init__(self, sight_range=200.0) -> None:
        super().__init__((0.0, 0.0))
        self.sight_range = sight_range

    def is_in_sight(self, target) -> bool:
        return math.dist(self.position, target.position) <= self.sight_range


class BattleWorldTests(unittest.TestCase):
    def test_step_has_one_authoritative_update_order(self) -> None:
        calls: list[str] = []
        world = BattleWorld(
            cast(Any, _RecordingMap(calls)),
            cast(Any, _RecordingUnitManager(calls)),
            cast(Any, _RecordingBulletManager(calls)),
        )

        world.step(0.25)

        self.assertEqual(
            calls,
            ["map:0.25", "units:0.25", "bullets:0.25", "vision"],
        )
        self.assertEqual(world.tick, 1)
        self.assertEqual(world.elapsed_time, 0.25)

    def test_non_positive_delta_time_is_rejected(self) -> None:
        world = BattleWorld(cast(Any, _RecordingMap([])))
        with self.assertRaises(ValueError):
            world.step(0.0)

    def test_visibility_combines_range_and_map_line_of_sight(self) -> None:
        observer = cast(Any, _Observer())
        target = cast(Any, _Target())
        manager = UnitManager()
        game_map = GameMap()

        self.assertTrue(manager.is_visible(game_map, observer, target))
        game_map.bullet_obstacles.append(pygame.Rect(45, -5, 10, 10))
        self.assertFalse(manager.is_visible(game_map, observer, target))

        target.conceal = True
        game_map.bullet_obstacles.clear()
        self.assertFalse(manager.is_visible(game_map, observer, target))

    def test_fire_registers_created_projectile_once(self) -> None:
        projectile = cast(Any, object())

        class Shooter:
            def fire(self):
                return projectile

        bullets = BulletManager()
        self.assertIs(bullets.fire(cast(Any, Shooter())), projectile)
        self.assertEqual(bullets.bullets, [projectile])

    def test_unit_manager_constructs_supported_units_with_valid_bounds(self) -> None:
        for unit_type in ("tank", "archie", "plane"):
            with self.subTest(unit_type=unit_type):
                unit = UnitManager.create_unit(
                    unit_type,
                    7,
                    Team.PLAYER,
                    (123.0, 234.0),
                )
                self.assertEqual(unit.unit_type, unit_type)
                self.assertEqual(unit.position, (123.0, 234.0))
                self.assertLessEqual(abs(unit.bounding_box.centerx - 123), 1)
                self.assertLessEqual(abs(unit.bounding_box.centery - 234), 1)

        with self.assertRaises(ValueError):
            UnitManager.create_unit("unknown", 1, Team.PLAYER)


if __name__ == "__main__":
    unittest.main()
