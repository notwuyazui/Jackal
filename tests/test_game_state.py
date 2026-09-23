from __future__ import annotations

import json
import os
from pathlib import Path
from random import Random
import sys
import tempfile
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.jackal_env import JackalEnv
from game.BattleWorld import BattleWorld
from game.Parameter import Team
from training.marl2.envs.jackal_env import JackalMultiAgentEnv
from training.test import init_game


class GameStateTests(unittest.TestCase):
    def test_game_state_jitter_uses_supplied_rng(self) -> None:
        first = BattleWorld().load_game_state(
            "test_map_7v7_1",
            position_jitter=1.0,
            heading_jitter=2.0,
            rng=Random(123),
        )
        second = BattleWorld().load_game_state(
            "test_map_7v7_1",
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
        project_root = Path(__file__).resolve().parents[1]
        config_path = (
            project_root
            / "training"
            / "configs"
            / "marl2"
            / "jackal_autoaim_5v5_etdqmix_hetero_valley_mapfeat7_10m_v2.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["env"]["game_state_file"], "valley_map_5v5_1")

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
            ("test_map_7v7_1", 12, (960, 640)),
            ("big_map_test_9v9_1", 18, (1600, 1024)),
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
            game_state_file="test_map_7v7_1",
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


if __name__ == "__main__":
    unittest.main()
