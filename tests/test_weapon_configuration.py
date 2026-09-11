import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

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

        self.env.step([27])

        player_bullets = [
            bullet
            for bullet in self.env.bullet_manager.bullets
            if bullet.shooter is agent
        ]
        self.assertEqual(len(player_bullets), 1)
        self.assertEqual(player_bullets[0].cooldown, 0.35)
        self.assertAlmostEqual(agent.fire_cooldown, 0.32)

        self.env.step([0])
        self.assertAlmostEqual(agent.fire_cooldown, 0.29)


if __name__ == "__main__":
    unittest.main()
