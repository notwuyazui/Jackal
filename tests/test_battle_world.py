import math
import os
import sys
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

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

    def refresh_vision(self, bullet_manager, game_map, **kwargs) -> None:
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
        self.bounding_box = pygame.Rect(position[0] - 5, position[1] - 5, 10, 10)
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

    def test_acc_scales_all_simulation_time(self) -> None:
        calls: list[str] = []
        world = BattleWorld(
            cast(Any, _RecordingMap(calls)),
            cast(Any, _RecordingUnitManager(calls)),
            cast(Any, _RecordingBulletManager(calls)),
        )

        with patch("game.BattleWorld.ACC", 2.0):
            world.step(0.25)

        self.assertEqual(
            calls,
            ["map:0.5", "units:0.5", "bullets:0.5", "vision"],
        )
        self.assertEqual(world.elapsed_time, 0.5)

    def test_visibility_combines_range_and_map_line_of_sight(self) -> None:
        observer = cast(Any, _Observer())
        target = cast(Any, _Target())
        manager = UnitManager()
        game_map = GameMap()

        self.assertTrue(manager.is_visible(game_map, observer, target))
        game_map.bullet_obstacles.append(pygame.Rect(45, -5, 10, 10))
        manager.invalidate_perception_cache()
        self.assertFalse(manager.is_visible(game_map, observer, target))

        target.conceal = True
        game_map.bullet_obstacles.clear()
        manager.invalidate_perception_cache()
        self.assertFalse(manager.is_visible(game_map, observer, target))

    def test_visibility_raycast_is_cached_for_one_perception_frame(self) -> None:
        class CountingMap(GameMap):
            def __init__(self) -> None:
                super().__init__()
                self.raycast_calls = 0

            def has_line_of_sight(self, start, end) -> bool:
                self.raycast_calls += 1
                return True

        observer = cast(Any, _Observer())
        target = cast(Any, _Target())
        manager = UnitManager()
        game_map = CountingMap()

        self.assertTrue(manager.is_visible(game_map, observer, target))
        self.assertTrue(manager.is_visible(game_map, observer, target))
        self.assertEqual(game_map.raycast_calls, 1)

        manager.invalidate_perception_cache()
        self.assertTrue(manager.is_visible(game_map, observer, target))
        self.assertEqual(game_map.raycast_calls, 2)

    def test_spatial_indices_reject_distant_candidates(self) -> None:
        game_map = GameMap()
        near = cast(Any, _Target((20.0, 20.0)))
        far = cast(Any, _Target((500.0, 500.0)))
        manager = UnitManager()
        manager.units = [near, far]
        manager.rebuild_unit_spatial_index(game_map)

        candidates = manager.get_units_in_radius((20.0, 20.0), 50.0)
        self.assertIn(near, candidates)
        self.assertNotIn(far, candidates)

        near_obstacle = pygame.Rect(64, 0, 64, 64)
        far_obstacle = pygame.Rect(64, 640, 64, 64)
        game_map.bullet_obstacles = [near_obstacle, far_obstacle]
        line_candidates = game_map.get_candidate_line_obstacles(
            (0.0, 32.0),
            (200.0, 32.0),
        )
        self.assertIn(near_obstacle, line_candidates)
        self.assertNotIn(far_obstacle, line_candidates)

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
