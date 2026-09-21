import unittest

import numpy as np

from training.marl2.runners.parallel_episode_runner import ParallelEpisodeRunner


class _FakeConnection:
    def __init__(self) -> None:
        self.pending = []
        self.step = 0
        self.reset_seeds = []

    def send(self, message) -> None:
        command, data = message
        if command == "get_avail_actions":
            raise AssertionError("runner performed a redundant availability round trip")
        if command == "reset":
            self.step = 0
            self.reset_seeds.append(data)
            self.pending.append(
                (
                    np.zeros((1, 1), dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                    np.ones((1, 2), dtype=np.float32),
                )
            )
            return
        if command == "step":
            self.step += 1
            terminated = self.step >= 2
            next_avail_actions = (
                None if terminated else np.ones((1, 2), dtype=np.float32)
            )
            self.pending.append(
                (
                    1.0,
                    terminated,
                    {"battle_won": terminated},
                    np.full((1, 1), self.step, dtype=np.float32),
                    np.full(1, self.step, dtype=np.float32),
                    next_avail_actions,
                )
            )
            return
        raise AssertionError(f"unexpected command: {command}")

    def recv(self):
        return self.pending.pop(0)


class _FakeMAC:
    def init_hidden(self, batch_size) -> None:
        self.batch_size = batch_size

    def select_actions_batch(
        self,
        obs_batch,
        avail_actions_batch,
        epsilon,
        test_mode,
        active_envs,
    ):
        return np.zeros((len(obs_batch), 1), dtype=np.int64)


def _make_runner(n_envs=2):
    runner = object.__new__(ParallelEpisodeRunner)
    runner.n_envs = n_envs
    runner.n_agents = 1
    runner.n_actions = 2
    runner.obs_shape = 1
    runner.state_shape = 1
    runner.episode_limit = 2
    runner.mac = _FakeMAC()
    runner.parent_conns = [_FakeConnection() for _ in range(n_envs)]
    runner.closed = False
    return runner


class ParallelEpisodeRunnerTests(unittest.TestCase):
    def test_training_run_reuses_availability_returned_by_reset_and_step(self) -> None:
        runner = _make_runner()

        batches, stats = runner.run(test_mode=False, epsilon=0.1)

        self.assertEqual(len(batches), 2)
        self.assertEqual([stat["episode_length"] for stat in stats], [2, 2])
        self.assertTrue(all(stat["battle_won"] for stat in stats))

    def test_evaluation_can_use_part_of_pool_without_allocating_batches(self) -> None:
        runner = _make_runner()

        batches, stats = runner.run(
            test_mode=True,
            epsilon=0.0,
            n_episodes=1,
            episode_seeds=[123],
            collect_batches=False,
        )

        self.assertIsNone(batches)
        self.assertEqual(len(stats), 1)
        self.assertEqual(runner.parent_conns[0].reset_seeds, [123])
        self.assertEqual(runner.parent_conns[1].reset_seeds, [])


if __name__ == "__main__":
    unittest.main()
