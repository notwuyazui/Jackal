import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

import game.GameMode as GameMode
from environment.action_controller import ActionController
from environment.rendering.pygame_renderer import PygameRenderer
from game.BattleWorld import BattleWorld
from game.BattleState import CombatEvent, WorldSnapshot
from game.Bullet.BulletManager import BulletManager
from game.Map.GameMap import GameMap
from game.Parameter import DEFAULT_AI_INTELLIGENCE_LEVEL, Team
from game.Unit.UnitManager import UnitManager


class _RecordingMap:
    bullet_obstacles: list[Any] = []

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def get_map_size(self) -> tuple[int, int]:
        return 960, 640

    def update(self, delta_time: float) -> None:
        self.calls.append(f"map:{delta_time}")


class _RecordingUnitManager:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.units = []
        self.enemy_ais = []
        self.combat_events = []

    def begin_step(self, tick: int) -> None:
        self.combat_events.clear()

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
        self.last_use_tear_drop_vision = None

    def is_in_sight(self, target, use_tear_drop_vision=False) -> bool:
        self.last_use_tear_drop_vision = use_tear_drop_vision
        return math.dist(self.position, target.position) <= self.sight_range


class BattleWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        collision_mode = patch.object(GameMode, "ENABLE_UNIT_COLLISION", False)
        collision_mode.start()
        self.addCleanup(collision_mode.stop)

    def test_mouse_target_line_is_only_drawn_for_keyboard_unit(self) -> None:
        renderer = PygameRenderer(100, 100)
        unit = SimpleNamespace(
            id=1,
            is_alive=True,
            visible=True,
            position=(20.0, 20.0),
            body_image_path=None,
            turret_image_path=None,
            direction_angle=0.0,
            turret_direction_angle=0.0,
        )
        try:
            with (
                patch.object(GameMode, "DEBUG_MODE", False),
                patch.object(GameMode, "DRAW_HEALTH_BAR", False),
                patch.object(GameMode, "DRAW_SIGHT_RANGE", False),
                patch.object(GameMode, "DRAW_MOUSE_TARGET_LINE", True),
                patch.object(pygame.draw, "line") as draw_line,
            ):
                renderer._draw_unit(unit, (0.0, 0.0), (80.0, 80.0), False)
                unit.id = 0
                renderer._draw_unit(unit, (0.0, 0.0), (80.0, 80.0), False)

            draw_line.assert_called_once()
        finally:
            renderer.close()

    def test_unit_manager_game_modes_use_defaults_and_allow_overrides(self) -> None:
        with patch.multiple(
            GameMode,
            ENABLE_UNIT_COLLISION=True,
            USE_TEAR_DROP_VISION=True,
            AUTO_COMMUNICATE=True,
        ):
            defaults = UnitManager()

        self.assertTrue(defaults.enable_unit_collision)
        self.assertTrue(defaults.use_tear_drop_vision)
        self.assertTrue(defaults.auto_communicate_enabled)

        overrides = UnitManager(
            enable_unit_collision=False,
            use_tear_drop_vision=False,
            auto_communicate=False,
        )
        self.assertFalse(overrides.enable_unit_collision)
        self.assertFalse(overrides.use_tear_drop_vision)
        self.assertFalse(overrides.auto_communicate_enabled)

        observer = cast(Any, _Observer())
        target = cast(Any, _Target())
        self.assertTrue(defaults.is_in_view(observer, target))
        self.assertTrue(observer.last_use_tear_drop_vision)
        self.assertTrue(overrides.is_in_view(observer, target))
        self.assertFalse(observer.last_use_tear_drop_vision)

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

    def test_placement_queries_share_bounds_terrain_and_unit_rules(self) -> None:
        game_map = GameMap(
            [
                "ooooo",
                "ooooo",
                "ooxoo",
                "ooooo",
                "ooooo",
            ],
            tile_size=64,
        )
        world = BattleWorld(
            game_map,
            unit_manager=UnitManager(enable_unit_collision=True),
        )
        collision_size = (16.0, 23.0)

        self.assertTrue(game_map.can_place_unit((96.0, 96.0), collision_size))
        self.assertTrue(world.can_place_unit((96.0, 96.0), collision_size))
        self.assertFalse(game_map.can_place_unit((4.0, 4.0), collision_size))
        self.assertFalse(world.can_place_unit((4.0, 4.0), collision_size))
        self.assertFalse(game_map.can_place_unit((160.0, 160.0), collision_size))
        self.assertFalse(world.can_place_unit((160.0, 160.0), collision_size))

        unit = world.create_unit(
            "tank",
            Team.PLAYER,
            (96.0, 96.0),
            unit_id=1,
        )
        self.assertTrue(world.can_move_unit(unit.id, unit.position))
        self.assertFalse(world.can_place_unit(unit.position, unit.collision_size))
        self.assertFalse(world.can_move_unit(unit.id, (160.0, 160.0)))
        self.assertFalse(world.can_move_unit(unit.id, (4.0, 4.0)))

        with self.assertRaisesRegex(ValueError, "outside map bounds"):
            world.create_unit("tank", Team.ENEMY, (4.0, 4.0), unit_id=2)
        with self.assertRaisesRegex(ValueError, "impassable terrain"):
            world.create_unit("tank", Team.ENEMY, (160.0, 160.0), unit_id=3)
        with self.assertRaisesRegex(ValueError, "overlaps unit 1"):
            world.create_unit("tank", Team.ENEMY, unit.position, unit_id=4)

    def test_action_precheck_uses_next_step_placement_query(self) -> None:
        game_map = GameMap(
            [
                "ooooo",
                "ooxoo",
                "ooooo",
            ],
            tile_size=64,
        )
        world = BattleWorld(
            game_map,
            unit_manager=UnitManager(enable_unit_collision=True),
        )
        unit = world.create_unit(
            "tank",
            Team.PLAYER,
            (110.0, 96.0),
            unit_id=1,
        )
        unit.velocity = (400.0, 0.0)
        controller = ActionController(
            auto_aim=True,
            fire_angle_tolerance=None,
            auto_fire_when_ready=False,
            delta_time=0.1,
        )
        snapshot = world.snapshot()

        self.assertFalse(world.can_move_unit(1, unit.candidate_position(0.1)))
        available = controller.available_actions(world, snapshot, 0)
        self.assertEqual(available[0], 1)
        self.assertEqual(available[1], 0)
        self.assertEqual(available[3], 1)

        controller.apply(world, snapshot, [1])
        self.assertEqual(unit.acceleration, 0.0)

        world.step(0.1)
        self.assertEqual(unit.position, (110.0, 96.0))

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
                self.assertFalse(hasattr(unit, "body_image"))
                self.assertFalse(hasattr(unit, "turret_image"))
                self.assertEqual(
                    unit.ai_intelligence_level,
                    DEFAULT_AI_INTELLIGENCE_LEVEL,
                )
                self.assertLessEqual(abs(unit.bounding_box.centerx - 123), 1)
                self.assertLessEqual(abs(unit.bounding_box.centery - 234), 1)

        with self.assertRaises(ValueError):
            UnitManager.create_unit("unknown", 1, Team.PLAYER)

    def test_world_configures_unit_before_registering_ai(self) -> None:
        world = BattleWorld(GameMap())
        unit = world.create_unit(
            "tank",
            Team.ENEMY,
            (123.0, 234.0),
            unit_id=9,
            using_ai=True,
            sight_range=350.0,
            communication_range=300.0,
            stat_scale_layers=(
                {"speed": 1.5, "health": 2.0},
                {"speed": 0.5, "health": 0.75},
            ),
            fire_cooldown=0.4,
            projectile_overrides={"normal_shell": {"speed_rate": 1.5}},
            initial_heading=725.0,
            ai_fire_angle_tolerance=3.0,
        )

        self.assertEqual(unit.sight_range, 350.0)
        self.assertEqual(unit.communication_range, 300.0)
        self.assertAlmostEqual(unit.max_speed, 37.5)
        self.assertAlmostEqual(unit.max_health, 150.0)
        self.assertAlmostEqual(unit.health, 150.0)
        self.assertEqual(unit.direction_angle, 5.0)
        self.assertEqual(unit.turret_direction_angle, 5.0)
        self.assertAlmostEqual(unit.fire_cooldown_duration(), 0.4)
        self.assertAlmostEqual(unit.get_weapon_spec().speed, 600.0)
        self.assertEqual(len(world.unit_manager.enemy_ais), 1)
        self.assertEqual(world.unit_manager.enemy_ais[0].fire_angle_tolerance, 3.0)

    def test_ai_intelligence_level_defaults_overrides_and_validates(self) -> None:
        world = BattleWorld(GameMap())
        default_unit = world.create_unit(
            "tank",
            Team.ENEMY,
            (100.0, 100.0),
            unit_id=1,
            using_ai=True,
        )
        custom_unit = world.create_unit(
            "archie",
            Team.ENEMY,
            (200.0, 100.0),
            unit_id=2,
            using_ai=True,
            ai_intelligence_level=8,
        )

        self.assertEqual(
            default_unit.ai_intelligence_level,
            DEFAULT_AI_INTELLIGENCE_LEVEL,
        )
        self.assertEqual(custom_unit.ai_intelligence_level, 8)
        self.assertEqual(
            [ai.intelligence_level for ai in world.unit_manager.enemy_ais],
            [DEFAULT_AI_INTELLIGENCE_LEVEL, 8],
        )
        with self.assertRaisesRegex(ValueError, "between 1 and 9"):
            UnitManager.create_unit(
                "plane",
                3,
                Team.ENEMY,
                ai_intelligence_level=10,
            )

    def test_world_command_interface_controls_and_fires_unit(self) -> None:
        world = BattleWorld(GameMap())
        unit = world.create_unit("tank", Team.PLAYER, unit_id=4)

        self.assertTrue(world.set_unit_chassis(4, (True, False, False, True)))
        self.assertEqual(unit.acceleration, unit.max_acceleration)
        self.assertEqual(unit.angular_speed, unit.max_angular_speed)
        self.assertTrue(world.set_unit_turret_target_angle(4, 123.0))
        self.assertEqual(unit.turret_target_angle, 123.0)
        self.assertTrue(world.can_unit_fire(4))
        self.assertIsNotNone(world.set_unit_fire(4))
        self.assertEqual(world.get_active_bullets_counts(), 1)

    def test_unit_collision_blocks_friendly_and_enemy_units(self) -> None:
        team_pairs = (
            (Team.PLAYER, Team.PLAYER),
            (Team.PLAYER, Team.ENEMY),
            (Team.ENEMY, Team.ENEMY),
        )
        for mover_team, blocker_team in team_pairs:
            with self.subTest(mover_team=mover_team, blocker_team=blocker_team):
                world = BattleWorld(GameMap())
                world.unit_manager.enable_unit_collision = True
                # Insert the higher id first; updates must still run by stable unit id.
                blocker = world.create_unit(
                    "tank", blocker_team, (122.0, 100.0), unit_id=2
                )
                mover = world.create_unit(
                    "tank", mover_team, (100.0, 100.0), unit_id=1
                )
                mover.direction_angle = 90.0
                mover.speed = 20.0
                mover.velocity = mover.cal_velocity()

                world.step(0.5)

                self.assertEqual(mover.position, (100.0, 100.0))
                self.assertTrue(mover.blocked_by_unit)
                self.assertEqual(mover.unit_collision_count, 1)
                self.assertEqual(world.unit_manager.unit_collision_count, 1)
                self.assertFalse(mover.collision_box.colliderect(blocker.collision_box))

                snapshot = world.snapshot()
                mover_snapshot = next(unit for unit in snapshot.units if unit.unit_id == 1)
                self.assertTrue(mover_snapshot.blocked_by_unit)
                self.assertEqual(mover_snapshot.unit_collision_count, 1)

    def test_disabled_unit_collision_preserves_previous_movement(self) -> None:
        world = BattleWorld(GameMap())
        world.unit_manager.enable_unit_collision = False
        mover = world.create_unit("tank", Team.PLAYER, (100.0, 100.0), unit_id=1)
        blocker = world.create_unit("tank", Team.ENEMY, (122.0, 100.0), unit_id=2)
        mover.direction_angle = 90.0
        mover.speed = 20.0
        mover.velocity = mover.cal_velocity()

        world.step(0.5)

        self.assertGreater(mover.position[0], 100.0)
        self.assertTrue(mover.collision_box.colliderect(blocker.collision_box))
        self.assertFalse(mover.blocked_by_unit)
        self.assertEqual(mover.unit_collision_count, 0)

    def test_collision_scale_does_not_change_render_or_hit_size(self) -> None:
        world = BattleWorld(GameMap())
        unit = world.create_unit(
            "tank",
            Team.PLAYER,
            (100.0, 100.0),
            unit_id=1,
            collision_scale=1.5,
        )

        self.assertEqual(unit.size, (16.0, 23.0))
        self.assertEqual(unit.base_collision_size, (16.0, 23.0))
        self.assertEqual(unit.collision_size, (24.0, 34.5))
        self.assertEqual(unit.bounding_box.size, (16, 23))
        self.assertEqual(unit.collision_box.size, (24, 34))

    def test_unit_collision_rejects_overlapping_spawn(self) -> None:
        world = BattleWorld(
            GameMap(),
            unit_manager=UnitManager(enable_unit_collision=False),
        )
        world.create_unit("tank", Team.PLAYER, (100.0, 100.0), unit_id=1)

        with self.assertRaisesRegex(ValueError, "cannot spawn.*overlaps unit 1"):
            world.create_unit("tank", Team.ENEMY, (100.0, 100.0), unit_id=2)

    def test_snapshot_is_read_only_and_contains_no_live_entities(self) -> None:
        world = BattleWorld(GameMap())
        unit = world.create_unit("tank", Team.PLAYER, (100.0, 100.0), unit_id=1)
        snapshot = world.snapshot()

        self.assertIsInstance(snapshot, WorldSnapshot)
        self.assertEqual(snapshot.allies[0].position, (100.0, 100.0))
        self.assertIsNot(snapshot.allies[0], unit)
        with self.assertRaises(AttributeError):
            snapshot.allies[0].health = 0.0  # type: ignore[misc]

    def test_damage_emits_structured_combat_events(self) -> None:
        world = BattleWorld(GameMap())
        attacker = world.create_unit(
            "tank", Team.PLAYER, (100.0, 100.0), unit_id=1
        )
        target = world.create_unit(
            "tank", Team.ENEMY, (200.0, 100.0), unit_id=2
        )
        world.unit_manager.begin_step(1)

        target.take_damage(world.unit_manager, attacker, target.health)
        events = world.snapshot().combat_events

        self.assertEqual([event.event_type for event in events], ["damage", "destroyed"])
        self.assertTrue(all(isinstance(event, CombatEvent) for event in events))
        self.assertEqual(events[0].amount, target.max_health)
        self.assertEqual(events[0].source_id, attacker.id)
        self.assertEqual(events[0].target_id, target.id)


if __name__ == "__main__":
    unittest.main()
