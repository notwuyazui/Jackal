import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environment.jackal_env import JackalEnv
from game.Bullet.weapon_specs import get_projectile_spec


class ProjectileSpecTests(unittest.TestCase):
    def test_canonical_ranges_match_projectile_motion(self) -> None:
        self.assertEqual(get_projectile_spec("normal_shell").max_range, 480.0)
        self.assertEqual(get_projectile_spec("rocket_shell").max_range, 720.0)
        self.assertEqual(get_projectile_spec("heavy_shell").max_range, 1600.0)

    def test_absolute_speed_override_updates_derived_range(self) -> None:
        spec = get_projectile_spec("normal_shell", {"speed": 600.0})

        self.assertEqual(spec.speed, 600.0)
        self.assertEqual(spec.max_range, 720.0)

    def test_conflicting_speed_overrides_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_projectile_spec(
                "normal_shell",
                {"speed": 600.0, "speed_rate": 1.5},
            )


class EnvironmentWeaponIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = JackalEnv(
            headless=True,
            auto_aim=False,
            enemy_use_ai=True,
            agent_fire_cooldown_by_type={"tank": 0.35},
            enemy_ai_fire_cooldown_by_type={"tank": 0.45},
            bullet_overrides_by_unit_type={
                "tank": {
                    "normal_shell": {
                        "speed_rate": 1.5,
                        "damage_rate": 3.0,
                    }
                }
            },
        )
        self.env.reset()

    def tearDown(self) -> None:
        self.env.close()

    def test_unit_is_single_source_for_cooldown_range_and_overrides(self) -> None:
        agent = self.env.agents[0]
        enemy = self.env.enemies[0]

        self.assertIsNone(self.env.renderer)
        self.assertEqual(agent.fire_cooldown_duration(), 0.35)
        self.assertEqual(enemy.fire_cooldown_duration(), 0.45)
        self.assertEqual(agent.weapon_range(), 720.0)
        self.assertEqual(agent.get_weapon_spec("rocket_shell").speed, 480.0)
        self.assertTrue(agent.can_fire())

        bullet = agent.fire()

        self.assertIsNotNone(bullet)
        assert bullet is not None
        self.assertEqual(bullet.cooldown, 0.35)
        self.assertEqual(bullet.speed, 600.0)
        self.assertEqual(bullet.base_damage, 30.0)
        self.assertEqual(agent.fire_cooldown, bullet.cooldown)
        self.assertFalse(agent.can_fire())
        self.assertFalse(hasattr(self.env, "agent_fire_cooldowns"))
        self.assertTrue(
            all(not hasattr(ai, "fire_cooldown") for ai in self.env.unit_manager.enemy_ais)
        )

    def test_environment_step_advances_the_unit_cooldown_once(self) -> None:
        agent = self.env.agents[0]
        assert self.env.world is not None

        self.env.step([27])

        player_bullets = [
            bullet
            for bullet in self.env.bullet_manager.bullets
            if bullet.shooter is agent
        ]
        self.assertEqual(len(player_bullets), 1)
        self.assertEqual(player_bullets[0].cooldown, 0.35)
        self.assertAlmostEqual(agent.fire_cooldown, 0.32)
        self.assertEqual(self.env.world.tick, 1)
        self.assertAlmostEqual(self.env.world.elapsed_time, 0.03)
        self.assertIs(self.env.game_map, self.env.world.game_map)
        self.assertIs(self.env.unit_manager, self.env.world.unit_manager)
        self.assertIs(self.env.bullet_manager, self.env.world.bullet_manager)

        self.env.step([0])
        self.assertAlmostEqual(agent.fire_cooldown, 0.29)

    def test_environment_metadata_matches_observation_outputs(self) -> None:
        obs, state = self.env.reset()
        info = self.env.get_env_info()

        self.assertEqual(info["n_agents"], len(obs))
        self.assertEqual(info["obs_shape"], obs[0].shape[0])
        self.assertEqual(info["state_shape"], state.shape[0])
        self.assertEqual(info["n_actions"], self.env.n_actions)

        runtime_info = self.env.get_runtime_info()
        self.assertEqual(runtime_info["steps"], 0)
        self.assertEqual(runtime_info["world_tick"], 0)
        self.assertEqual(runtime_info["active_allies"], 1)
        self.assertEqual(runtime_info["active_enemies"], 1)
        self.assertEqual(runtime_info["active_bullets"], 0)

    def test_collision_configuration_preserves_environment_shapes(self) -> None:
        collision_env = JackalEnv(
            headless=True,
            auto_aim=False,
            enable_unit_collision=True,
            use_tear_drop_vision=True,
            auto_communicate=True,
            collision_scale=1.5,
        )
        try:
            obs, state = collision_env.reset()
            info = collision_env.get_env_info()
            baseline_info = self.env.get_env_info()

            self.assertTrue(info["enable_unit_collision"])
            self.assertTrue(info["use_tear_drop_vision"])
            self.assertTrue(info["auto_communicate"])
            self.assertTrue(collision_env.unit_manager.enable_unit_collision)
            self.assertTrue(collision_env.unit_manager.use_tear_drop_vision)
            self.assertTrue(collision_env.unit_manager.auto_communicate_enabled)
            self.assertEqual(info["n_actions"], baseline_info["n_actions"])
            self.assertEqual(info["obs_shape"], baseline_info["obs_shape"])
            self.assertEqual(info["state_shape"], baseline_info["state_shape"])
            self.assertEqual(obs[0].shape[0], baseline_info["obs_shape"])
            self.assertEqual(state.shape[0], baseline_info["state_shape"])
            self.assertEqual(collision_env.agents[0].size, (16.0, 23.0))
            self.assertEqual(collision_env.agents[0].collision_size, (24.0, 34.5))

            runtime_info = collision_env.get_runtime_info()
            self.assertEqual(runtime_info["blocked_by_unit"], (False,))
            self.assertEqual(runtime_info["unit_collision_count"], (0,))
        finally:
            collision_env.close()

    def test_reset_rejects_spawn_on_impassable_terrain(self) -> None:
        invalid_env = JackalEnv(
            headless=True,
            map_data=("xxx", "xox", "xxx"),
            map_tile_size=64,
            ally_positions=((32.0, 32.0),),
            enemy_positions=((96.0, 96.0),),
        )
        try:
            with self.assertRaisesRegex(ValueError, "impassable terrain"):
                invalid_env.reset()
        finally:
            invalid_env.close()

    def test_default_five_vs_five_spawns_are_valid(self) -> None:
        env = JackalEnv(
            headless=True,
            n_agents=5,
            n_enemies=5,
            enemy_use_ai=False,
        )
        try:
            obs, _ = env.reset()
            self.assertEqual(len(obs), 5)
            self.assertEqual(len(env.world.unit_manager.units), 10)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
