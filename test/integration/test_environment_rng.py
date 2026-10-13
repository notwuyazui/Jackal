import random
from random import Random
import unittest

import numpy as np

from environment.jackal_env import JackalEnv
from environment.scenario import create_scenario_map


def _make_env(seed=None) -> JackalEnv:
    return JackalEnv(
        headless=True,
        seed=seed,
        n_agents=1,
        n_enemies=1,
        ally_positions=[(220.0, 220.0)],
        enemy_positions=[(740.0, 420.0)],
        ally_initial_headings=[15.0],
        enemy_initial_headings=[195.0],
        position_jitter=20.0,
        heading_jitter=10.0,
        max_steps=1,
        enemy_use_ai=False,
    )


def _initial_units(env: JackalEnv):
    return tuple(
        (unit.unit_id, unit.position, unit.direction_angle)
        for unit in env.snapshot.units
    )


class EnvironmentRandomnessTests(unittest.TestCase):
    def test_random_map_uses_the_supplied_rng(self) -> None:
        kwargs = {
            "map_name": "random",
            "map_file": None,
            "map_data": None,
            "map_tile_size": 64,
        }
        first = create_scenario_map(**kwargs, rng=Random(123))
        second = create_scenario_map(**kwargs, rng=Random(123))
        different = create_scenario_map(**kwargs, rng=Random(124))

        self.assertEqual(first.to_strings(), second.to_strings())
        self.assertNotEqual(first.to_strings(), different.to_strings())

    def test_environment_seed_reproduces_random_builtin_map(self) -> None:
        kwargs = {
            "headless": True,
            "map_name": "random",
            "n_agents": 1,
            "n_enemies": 1,
            "ally_positions": [(224.0, 224.0)],
            "enemy_positions": [(736.0, 416.0)],
            "enemy_use_ai": False,
            "max_steps": 1,
        }
        first = JackalEnv(**kwargs)
        second = JackalEnv(**kwargs)
        try:
            first.reset(seed=1)
            second.reset(seed=1)
            self.assertEqual(
                first.world.game_map.to_strings(),
                second.world.game_map.to_strings(),
            )
        finally:
            first.close()
            second.close()

    def test_explicit_seed_reproduces_initial_scene_and_observations(self) -> None:
        first = _make_env()
        second = _make_env()
        try:
            first_obs, first_state = first.reset(seed=982000)
            second_obs, second_state = second.reset(seed=982000)

            self.assertEqual(_initial_units(first), _initial_units(second))
            for left, right in zip(first_obs, second_obs):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(first_state, second_state)
        finally:
            first.close()
            second.close()

    def test_private_rng_is_independent_of_module_global_random(self) -> None:
        first = _make_env(seed=77)
        second = _make_env(seed=77)
        try:
            random.seed(1)
            for _ in range(100):
                random.random()
            first.reset()

            random.seed(999)
            for _ in range(7):
                random.random()
            second.reset()

            self.assertEqual(first.get_runtime_info()["episode_seed"], second.get_runtime_info()["episode_seed"])
            self.assertEqual(_initial_units(first), _initial_units(second))
        finally:
            first.close()
            second.close()

    def test_different_seeds_change_jitter_and_step_info_records_seed(self) -> None:
        env = _make_env()
        try:
            env.reset(seed=10)
            first_units = _initial_units(env)
            env.reset(seed=11)
            second_units = _initial_units(env)
            self.assertNotEqual(first_units, second_units)

            _, _, _, done, info = env.step([0])
            self.assertTrue(done)
            self.assertEqual(info["episode_seed"], 11)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
