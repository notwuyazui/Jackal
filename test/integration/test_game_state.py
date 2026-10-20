from __future__ import annotations

import json
import os
from pathlib import Path
from random import Random
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from environment.jackal_env import JackalEnv
from engine.battle_world import BattleWorld
from engine.parameters import ArmorType, Team
from environment.marl_adapter import JackalMultiAgentEnv
from training.run_game_keyboard import init_game


class GameStateTests(unittest.TestCase):
    def test_unarmed_units_can_be_saved_loaded_and_rendered(self):
        world = BattleWorld()
        unit = world.create_unit('tank', Team.PLAYER, (256., 256.), unit_id=1, ammunition_types=[])
        self.assertFalse(unit.can_fire())
        self.assertEqual(world.render_manifest(.03).units[0].shell_range, 0.)
        with tempfile.TemporaryDirectory() as directory:
            path = world.save_game_state(Path(directory) / 'unarmed.json')
            restored = BattleWorld().load_game_state(path)
        self.assertEqual(restored.get_unit(1).to_game_state(), unit.to_game_state())
        self.assertEqual(restored.render_manifest(.03), world.render_manifest(.03))

    def test_duplicate_ids_and_invalid_dimensions_do_not_register_units(self):
        world = BattleWorld()
        world.create_unit('tank', Team.PLAYER, (256., 256.), unit_id=1)
        before = world.render_snapshot()
        with self.assertRaisesRegex(ValueError, 'Duplicate unit id'):
            world.create_unit('tank', Team.ENEMY, (512., 256.), unit_id=1)
        for dimensions in ((0., 20.), (float('nan'), 20.), (20., float('inf'))):
            with self.assertRaises(ValueError):
                world.create_unit('tank', Team.ENEMY, (512., 256.), size=dimensions)
            self.assertEqual(world.render_snapshot(), before)
        # len(units) == 1 is already taken; automatic ids must avoid duplicates.
        self.assertEqual(world.create_unit('tank', Team.ENEMY, (512., 256.)).id, 2)

    def test_custom_footprints_armor_and_ammunition_survive_save_and_load(self):
        world = BattleWorld()
        unit = world.create_unit('tank', Team.PLAYER, (256., 256.), unit_id=1,
            size=(20., 30.), collision_size=(12., 18.), collision_scale=.75,
            armor_type=ArmorType.HEAVY, ammunition_types=['normal_shell'], ammo_switch_time=.8)
        unit.min_sight_range = 125.
        world.fire_weapon(unit)
        world.refresh_vision()
        with tempfile.TemporaryDirectory() as directory:
            path = world.save_game_state(Path(directory) / 'custom.json')
            restored_world = BattleWorld().load_game_state(path)
        restored = restored_world.get_unit(1)
        self.assertEqual(restored.to_game_state(), unit.to_game_state())
        self.assertEqual(restored.bounding_box, unit.bounding_box)
        self.assertEqual(restored.collision_box, unit.collision_box)
        self.assertEqual(restored.collision_size, (9., 13.5))
        self.assertFalse(restored.switch_ammunition('rocket_shell'))
        self.assertEqual(restored_world.render_snapshot().units, world.render_snapshot().units)

    def test_restoring_a_dead_unit_does_not_block_an_overlapping_live_unit(self):
        world = BattleWorld()
        # The live unit has moved onto a position vacated by a destroyed tank.
        world.create_unit('tank', Team.PLAYER, (256., 256.), unit_id=1)
        world.create_unit('tank', Team.ENEMY, (256., 256.), unit_id=100, alive=False, health=0)
        world.refresh_vision()
        with tempfile.TemporaryDirectory() as directory:
            path = world.save_game_state(Path(directory) / 'dead.json')
            restored = BattleWorld().load_game_state(path)
        self.assertEqual([u.is_alive for u in restored.iter_units()], [True, False])
        self.assertEqual(restored.render_snapshot().units, world.render_snapshot().units)

    def test_nonfinite_physics_parameters_fail_without_advancing_the_world(self):
        world = BattleWorld().load_game_state('border_map_3t_vs_3t')
        original = world.render_snapshot()
        for value in (float('nan'), float('inf'), -1., 0.):
            with self.assertRaises(ValueError):
                world.step(value)
            self.assertEqual(world.render_snapshot(), original)
        unit = world.get_unit(1)
        for value in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                unit.configure_collision(value)
            with self.assertRaises(ValueError):
                unit.configure_weapon(fire_cooldown=value)
        self.assertEqual(world.render_snapshot(), original)

    def test_game_state_jitter_uses_supplied_rng(self) -> None:
        first = BattleWorld().load_game_state(
            "test_map_3t2a1p_vs_2t3a1p",
            position_jitter=1.0,
            heading_jitter=2.0,
            rng=Random(123),
        )
        second = BattleWorld().load_game_state(
            "test_map_3t2a1p_vs_2t3a1p",
            position_jitter=1.0,
            heading_jitter=2.0,
            rng=Random(123),
        )

        first_state = [
            (unit.id, unit.position, unit.direction_angle)
            for unit in first.unit_manager.units
        ]
        second_state = [
            (unit.id, unit.position, unit.direction_angle)
            for unit in second.unit_manager.units
        ]
        self.assertEqual(first_state, second_state)

    def test_valley_training_config_loads_5v5_state(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        config_path = (
            project_root
            / "training"
            / "configs"
            / "marl2"
            / "jackal_autoaim_5v5_etdqmix_hetero_valley_mapfeat7_10m_v2.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["env"]["game_state_file"],
            "valley_map_3t2a_vs_3t2a",
        )

        env = JackalMultiAgentEnv(config["env"])
        try:
            observations, state = env.reset()
            info = env.get_env_info()
            self.assertEqual(info["n_agents"], 5)
            self.assertEqual(info["n_enemies"], 5)
            self.assertEqual(info["obs_shape"], 363)
            self.assertEqual(info["state_shape"], 586)
            self.assertEqual(len(observations), 5)
            self.assertEqual(state.shape, (586,))
            self.assertEqual(env.env.world.game_map.get_map_size(), (960, 640))
            self.assertIsNone(env.env.world.get_unit(0))

            reward, done, _, next_observations, next_state = env.step([0] * 5)
            self.assertIsInstance(reward, float)
            self.assertFalse(done)
            self.assertEqual(len(next_observations), 5)
            self.assertEqual(next_state.shape, (586,))
        finally:
            env.close()

    def test_builtin_initial_states_exclude_keyboard_unit(self) -> None:
        cases = (
            ("test_map_3t2a1p_vs_2t3a1p", 12, (960, 640)),
            ("big_map_test_4t4a1p_vs_4t4a1p", 18, (1600, 1024)),
        )
        for state_name, unit_count, map_size in cases:
            with self.subTest(state_name=state_name):
                world = BattleWorld().load_game_state(state_name)
                self.assertEqual(len(world.unit_manager.units), unit_count)
                self.assertEqual(world.game_map.get_map_size(), map_size)
                self.assertIsNone(world.get_unit(0))
                self.assertEqual(len(world.bullet_manager.bullets), 0)

    def test_training_test_adds_keyboard_unit_after_state_load(self) -> None:
        cases = (
            ("test_map", 13, (100.0, 500.0)),
            ("big_map_test", 19, (100.0, 100.0)),
        )
        for map_name, unit_count, keyboard_spawn in cases:
            with self.subTest(map_name=map_name):
                world = init_game(map_name)
                self.assertEqual(len(world.unit_manager.units), unit_count)
                keyboard_unit = world.get_unit(0)
                assert keyboard_unit is not None
                self.assertEqual(keyboard_unit.position, keyboard_spawn)
                self.assertFalse(keyboard_unit.usingAI)

    def test_round_trip_excludes_unit_zero_and_its_bullet(self) -> None:
        world = BattleWorld()
        world.load_builtin_map("test_map")
        keyboard = world.create_unit(
            "tank",
            Team.PLAYER,
            (100.0, 500.0),
            unit_id=0,
        )
        ally = world.create_unit(
            "tank",
            Team.PLAYER,
            (256.0, 448.0),
            unit_id=1,
        )
        world.create_unit(
            "tank",
            Team.ENEMY,
            (768.0, 128.0),
            unit_id=100,
        )
        self.assertIsNotNone(world.fire_weapon(keyboard))
        self.assertIsNotNone(world.fire_weapon(ally))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_trip.json"
            world.save_game_state(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn(0, [unit["unit_id"] for unit in payload["units"]])
            self.assertNotIn(0, [bullet["shooter_id"] for bullet in payload["bullets"]])

            restored = BattleWorld().load_game_state(path)

        self.assertIsNone(restored.get_unit(0))
        self.assertEqual({unit.id for unit in restored.unit_manager.units}, {1, 100})
        self.assertEqual(len(restored.bullet_manager.bullets), 1)
        self.assertEqual(restored.bullet_manager.bullets[0].shooter.id, 1)

    def test_environment_uses_state_but_keeps_agents_externally_controlled(self) -> None:
        env = JackalEnv(
            game_state_file="test_map_3t2a1p_vs_2t3a1p",
            headless=True,
            max_steps=2,
            enemy_use_ai=True,
            enemy_ai_intelligence_level=7,
        )
        try:
            observations, _ = env.reset()
            self.assertEqual(env.n_agents, 6)
            self.assertEqual(env.n_enemies, 6)
            self.assertEqual(len(observations), 6)
            self.assertIsNone(env.world.get_unit(0))
            self.assertTrue(all(not unit.usingAI for unit in env.agents))
            self.assertTrue(all(unit.usingAI for unit in env.enemies))
            self.assertTrue(
                all(unit.ai_intelligence_level == 7 for unit in env.enemies)
            )
        finally:
            env.close()

    def test_state_restores_bullet_cooldown_seconds_and_unit_rate_override(self) -> None:
        world = BattleWorld()
        ally = world.create_unit(
            "tank", Team.PLAYER, (256.0, 256.0), unit_id=1,
            projectile_overrides={"normal_shell": {"cooldown_rate": 0.5}},
        )
        bullet = world.fire_weapon(ally)
        self.assertIsNotNone(bullet)
        assert bullet is not None
        original_cooldown = bullet.cooldown
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cooldown_state.json"
            world.save_game_state(path)
            # The saved projectile's actual duration is historical state, not
            # a fresh calculation using a possibly changed global setting.
            with patch("engine.units.base_unit.BULLET_COOLDOWN", 3.0):
                restored = BattleWorld().load_game_state(path)
                unit = restored.get_unit(1)
                assert unit is not None
                self.assertEqual(unit.get_weapon_spec().cooldown_rate, 0.5)
                self.assertEqual(unit.fire_cooldown_duration(), 1.5)
                self.assertEqual(unit.fire_cooldown, original_cooldown)
                self.assertEqual(restored.bullet_manager.bullets[0].cooldown, original_cooldown)


if __name__ == "__main__":
    unittest.main()
