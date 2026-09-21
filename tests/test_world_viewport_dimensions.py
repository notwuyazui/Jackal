import math
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from environment.jackal_env import JackalEnv
from environment.rendering.pygame_renderer import PygameRenderer
from game.Parameter import Direction


def _large_map() -> list[str]:
    rows = ["o" * 20 for _ in range(12)]
    rows[9] = "o" * 15 + "x" + "o" * 4
    return rows


def _large_env(**overrides) -> JackalEnv:
    kwargs = {
        "headless": True,
        "auto_aim": False,
        "enemy_use_ai": False,
        "map_data": _large_map(),
        "map_tile_size": 64,
        "ally_positions": ((1000.0, 700.0),),
        "enemy_positions": ((1200.0, 700.0),),
        "include_state_map_features": True,
        "state_map_grid_size": 2,
    }
    kwargs.update(overrides)
    return JackalEnv(**kwargs)


class WorldViewportDimensionTests(unittest.TestCase):
    def test_current_map_keeps_legacy_normalization(self) -> None:
        env = JackalEnv(headless=True, auto_aim=False, enemy_use_ai=False)
        try:
            observations, state = env.reset()
            agent = env.snapshot.allies[0]

            self.assertEqual((env.world_width, env.world_height), (960, 640))
            self.assertEqual((env.viewport_width, env.viewport_height), (960, 640))
            np.testing.assert_allclose(
                observations[0][:2],
                (agent.position[0] / 960.0, agent.position[1] / 640.0),
            )
            np.testing.assert_allclose(
                state[1:3],
                (agent.position[0] / 960.0, agent.position[1] / 640.0),
            )
        finally:
            env.close()

    def test_large_map_drives_normalization_and_global_map_sampling(self) -> None:
        env = _large_env()
        try:
            observations, state = env.reset()
            info = env.get_env_info()

            self.assertEqual((env.world_width, env.world_height), (1280, 768))
            self.assertEqual((info["world_width"], info["world_height"]), (1280, 768))
            self.assertEqual(
                (info["viewport_width"], info["viewport_height"]),
                (960, 640),
            )
            np.testing.assert_allclose(
                observations[0][:2],
                (1000.0 / 1280.0, 700.0 / 768.0),
            )
            np.testing.assert_allclose(
                state[1:3],
                (1000.0 / 1280.0, 700.0 / 768.0),
            )
            self.assertLessEqual(float(observations[0][0]), 1.0)
            self.assertLessEqual(float(observations[0][1]), 1.0)
            self.assertEqual(env.scenario_config.arena_size, (1280, 768))
            self.assertEqual(env.reward_manager.arena_size, (1280.0, 768.0))

            # The last 2x2 global-map sample lands at world position (960, 576),
            # where the large test map contains its only blocking tile.
            np.testing.assert_array_equal(
                state[-5:-1],
                np.asarray((1.0, 1.0, 1.0, 0.0), dtype=np.float32),
            )
        finally:
            env.close()

    def test_renderer_keeps_fixed_viewport_and_clamps_camera_to_world(self) -> None:
        env = _large_env()
        renderer = PygameRenderer(960, 640)
        try:
            env.reset()
            for _ in range(100):
                env.world.move_camera(Direction.RIGHT)
                env.world.move_camera(Direction.DOWN)

            surface = renderer.draw(env.world)

            self.assertEqual(surface.get_size(), (960, 640))
            self.assertEqual(renderer.rgb_array().shape, (640, 960, 3))
            self.assertEqual(env.world.get_camera_offset(), (320.0, 128.0))

            env.world.set_unit_turret_target_to_mouse(1, (100.0, 100.0))
            expected_angle = (
                math.degrees(math.atan2(228.0 - 700.0, 420.0 - 1000.0)) + 90.0
            ) % 360.0
            self.assertAlmostEqual(env.agents[0].turret_target_angle, expected_angle)
        finally:
            renderer.close()
            env.close()

    def test_video_writer_uses_viewport_dimensions_on_large_map(self) -> None:
        writer = Mock()
        with patch(
            "environment.jackal_env.create_video_writer",
            return_value=writer,
        ) as create_writer:
            env = _large_env(use_video=True)
            try:
                env.reset()

                self.assertEqual(create_writer.call_args.args[2], (960, 640))
                frame = writer.write.call_args.args[0]
                self.assertEqual(frame.shape, (640, 960, 3))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
