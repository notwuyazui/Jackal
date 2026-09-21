import unittest

from training.train_qmix_marl2 import _should_run_extra_evaluation


class ExtraEvaluationConfigurationTests(unittest.TestCase):
    def test_non_positive_episode_count_disables_extra_evaluation(self) -> None:
        self.assertFalse(_should_run_extra_evaluation(0, 64))
        self.assertFalse(_should_run_extra_evaluation(-1, 64))

    def test_matching_episode_count_reuses_standard_evaluation(self) -> None:
        self.assertFalse(_should_run_extra_evaluation(64, 64))

    def test_distinct_positive_episode_count_runs_extra_evaluation(self) -> None:
        self.assertTrue(_should_run_extra_evaluation(128, 64))


if __name__ == "__main__":
    unittest.main()
