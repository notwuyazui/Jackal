import unittest

from training.marl2.runners.parallel_episode_runner import ParallelEpisodeRunner
from training.train_qmix_marl2 import (
    _performance_metrics,
    _should_run_extra_evaluation,
    evaluate,
)


class _FakeParallelEvalRunner(ParallelEpisodeRunner):
    def __init__(self) -> None:
        self.n_envs = 2
        self.calls = []

    def run(
        self,
        test_mode=False,
        epsilon=0.0,
        *,
        n_episodes=None,
        episode_seeds=None,
        collect_batches=True,
    ):
        self.calls.append((n_episodes, episode_seeds, collect_batches))
        stats = [
            {
                "episode_return": float(seed),
                "episode_length": 10,
                "battle_won": bool(seed % 2),
                "episode_limit": False,
                "no_kill_timeout": False,
            }
            for seed in episode_seeds
        ]
        return None, stats


class ExtraEvaluationConfigurationTests(unittest.TestCase):
    def test_non_positive_episode_count_disables_extra_evaluation(self) -> None:
        self.assertFalse(_should_run_extra_evaluation(0, 64))
        self.assertFalse(_should_run_extra_evaluation(-1, 64))

    def test_matching_episode_count_reuses_standard_evaluation(self) -> None:
        self.assertFalse(_should_run_extra_evaluation(64, 64))

    def test_distinct_positive_episode_count_runs_extra_evaluation(self) -> None:
        self.assertTrue(_should_run_extra_evaluation(128, 64))


class ParallelEvaluationTests(unittest.TestCase):
    def test_parallel_evaluation_uses_batches_and_exact_episode_seeds(self) -> None:
        runner = _FakeParallelEvalRunner()

        stats = evaluate(runner, n_episodes=5, seed_base=10)

        self.assertEqual(
            runner.calls,
            [
                (2, [10, 11], False),
                (2, [12, 13], False),
                (1, [14], False),
            ],
        )
        self.assertAlmostEqual(stats["return_mean"], 12.0)
        self.assertAlmostEqual(stats["battle_won_mean"], 0.4)


class PerformanceMetricsTests(unittest.TestCase):
    def test_reports_phase_ratios_and_wall_clock_throughput(self) -> None:
        metrics = _performance_metrics(
            {"rollout": 4.0, "learner": 3.0, "evaluation": 1.0, "env_steps": 200},
            wall_seconds=10.0,
        )

        self.assertAlmostEqual(metrics["rollout_ratio"], 0.4)
        self.assertAlmostEqual(metrics["learner_ratio"], 0.3)
        self.assertAlmostEqual(metrics["evaluation_ratio"], 0.1)
        self.assertAlmostEqual(metrics["env_steps_per_second"], 20.0)


if __name__ == "__main__":
    unittest.main()
