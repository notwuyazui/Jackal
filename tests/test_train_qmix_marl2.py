import json
from pathlib import Path
import unittest

from training.marl2.runners.parallel_episode_runner import ParallelEpisodeRunner
from training.train_qmix_marl2 import (
    _performance_metrics,
    _select_rollback_action,
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


class _FakeSerialEvalRunner:
    def __init__(self) -> None:
        self.seeds = []

    def run(self, test_mode=False, epsilon=0.0, *, episode_seed=None):
        self.seeds.append(episode_seed)
        return None, {
            "episode_return": float(episode_seed),
            "episode_length": 10,
            "battle_won": False,
            "episode_limit": False,
            "no_kill_timeout": False,
            "episode_seed": episode_seed,
        }


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

    def test_serial_evaluation_passes_exact_episode_seeds_to_environment(self) -> None:
        runner = _FakeSerialEvalRunner()

        stats = evaluate(runner, n_episodes=3, seed_base=50)

        self.assertEqual(runner.seeds, [50, 51, 52])
        self.assertAlmostEqual(stats["return_mean"], 51.0)


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


class RollbackProtectionTests(unittest.TestCase):
    def test_allows_rollbacks_until_limit_then_stops(self) -> None:
        common = {
            "collapse_detected": True,
            "cooldown_ready": True,
            "rollback_max_times": 3,
            "stop_on_exhaustion": True,
        }
        self.assertEqual(
            _select_rollback_action(rollback_count=2, **common),
            "rollback",
        )
        self.assertEqual(
            _select_rollback_action(rollback_count=3, **common),
            "stop",
        )

    def test_cooldown_prevents_rollback_and_stop(self) -> None:
        self.assertIsNone(
            _select_rollback_action(
                collapse_detected=True,
                cooldown_ready=False,
                rollback_count=3,
                rollback_max_times=3,
                stop_on_exhaustion=True,
            )
        )

    def test_valley_config_enables_only_requested_stability_changes(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "training"
            / "configs"
            / "marl2"
            / "jackal_autoaim_5v5_etdqmix_hetero_valley_mapfeat7_10m_v2.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        env_config = config["env"]
        train_config = config["train"]

        self.assertEqual(env_config["position_jitter"], 3.0)
        self.assertEqual(env_config["heading_jitter"], 3.0)
        self.assertEqual(train_config["best_model_window"], 8)
        self.assertEqual(train_config["best_model_max_recent_zero"], 3)
        self.assertEqual(
            [
                train_config["stabilize_window_win_rate"],
                train_config["stabilize_stage2_window_win_rate"],
                train_config["stabilize_stage3_window_win_rate"],
            ],
            [0.15, 0.25, 0.35],
        )
        self.assertEqual(train_config["rollback_cooldown_evals"], 5)
        self.assertEqual(train_config["rollback_max_times"], 3)
        self.assertEqual(train_config["rollback_buffer_warmup_episodes"], 64)
        self.assertTrue(train_config["rollback_verify_after_load"])
        self.assertTrue(train_config["rollback_stop_on_exhaustion"])

        # Requested items 4 and 5 remain disabled/unchanged.
        self.assertIsNone(train_config["early_stop_win_rate"])
        self.assertEqual(config["algo"]["name"], "qmix")


if __name__ == "__main__":
    unittest.main()
