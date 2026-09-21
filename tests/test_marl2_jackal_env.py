import unittest
from unittest.mock import patch

import numpy as np

from training.marl2.envs.jackal_env import JackalMultiAgentEnv


class _FakeJackalEnv:
    def __init__(self, **kwargs):
        self.reset_calls = 0

    def get_env_info(self):
        return {
            "n_agents": 2,
            "n_enemies": 2,
            "n_actions": 4,
            "obs_shape": 3,
            "state_shape": 5,
            "episode_limit": 10,
        }

    def reset(self):
        self.reset_calls += 1
        obs = [np.zeros(3, dtype=np.float32) for _ in range(2)]
        state = np.zeros(5, dtype=np.float32)
        return obs, state


class _MismatchedJackalEnv(_FakeJackalEnv):
    def reset(self):
        self.reset_calls += 1
        obs = [np.zeros(2, dtype=np.float32) for _ in range(2)]
        state = np.zeros(5, dtype=np.float32)
        return obs, state


class JackalMultiAgentEnvResetTests(unittest.TestCase):
    def test_constructor_does_not_start_an_episode(self) -> None:
        with patch("training.marl2.envs.jackal_env.JackalEnv", _FakeJackalEnv):
            env = JackalMultiAgentEnv({"name": "jackal", "use_video": True})

        self.assertEqual(env.env.reset_calls, 0)

        obs, state = env.reset()

        self.assertEqual(env.env.reset_calls, 1)
        self.assertEqual(obs[0].shape, (3,))
        self.assertEqual(state.shape, (5,))

    def test_first_real_reset_still_validates_metadata_shapes(self) -> None:
        with patch("training.marl2.envs.jackal_env.JackalEnv", _MismatchedJackalEnv):
            env = JackalMultiAgentEnv({"name": "jackal"})

        with self.assertRaisesRegex(RuntimeError, "metadata does not match"):
            env.reset()


if __name__ == "__main__":
    unittest.main()
