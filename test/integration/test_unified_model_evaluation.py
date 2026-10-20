"""Cross-model checkpoint, evaluation metrics and resource lifecycle regressions."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import torch

from environment.evaluation import EvaluationStopped, ModelEvaluationSession
from environment.jackal_env import JackalEnv
from environment.marl_adapter import JackalMultiAgentEnv
from training.DQN_Test.network import QNetwork
from training.DRQN_Test.network import DRQNNetwork
from training.marl2.controllers.basic_mac import BasicMAC
from training.marl2.runners.episode_runner import EpisodeRunner
from training.model_evaluation import DQNPolicy, DRQNPolicy, create_policy, find_run_config, load_checkpoint
from training.test_model import evaluate, parse_args, summarize


class UnifiedModelEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def checkpoint(self, kind, *, periodic=False):
        directory = self.root / kind
        directory.mkdir()
        env_config = {"auto_aim": kind != "drqn", "max_steps": 3,
                      "position_jitter": 4.0, "heading_jitter": 5.0}
        if kind in {"qmix", "etd"}:
            env_config.update(n_agents=2, n_enemies=2)
        env = JackalEnv(**env_config)
        self.addCleanup(env.close)
        info = env.get_env_info()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            if kind == "dqn":
                network = QNetwork(info["state_shape"], info["n_actions"])
                config = {"algo": "dqn", "state_dim": info["state_shape"],
                          "action_dim": info["n_actions"], **env_config}
                weights = network.state_dict()
                name = "dqn_run_config.json"
            elif kind == "drqn":
                network = DRQNNetwork(info["state_shape"], info["n_actions"], hidden_dim=24)
                config = {"algo": "drqn", "state_dim": info["state_shape"],
                          "action_dim": info["n_actions"], "hidden_dim": 24,
                          "chassis_dim": 9, "turret_dim": 3, "fire_dim": 2,
                          "fire_action_id": 27, "factorized_action": True, **env_config}
                weights = network.state_dict()
                name = "run_config.json"
            else:
                agent = {"name": "etd_rnn" if kind == "etd" else "rnn", "rnn_hidden_dim": 16,
                         "etd_entity_embed_dim": 16, "etd_attn_heads": 4, "n_enemies": 2}
                config = {"seed": 17, "env": env_config, "agent": agent, "algo": {"name": "qmix"}}
                mac = BasicMAC(info["n_agents"], info["n_actions"], info["obs_shape"], agent, torch.device("cpu"))
                weights = {"agent": mac.agent.state_dict(), "mixer": {}, "optimizer": {}}
                name = "config.json"
        config_path = directory / name
        config_path.write_text(json.dumps(config))
        if periodic:
            directory = directory / "checkpoints"
            directory.mkdir()
        checkpoint = directory / ("ep100.pth" if periodic else "model.pth")
        torch.save(weights, checkpoint)
        return checkpoint, config_path, config, info

    def evaluate(self, checkpoint, *options):
        args = parse_args(["--checkpoint", str(checkpoint), "--episodes", "2", "--device", "cpu", *options])
        with redirect_stdout(io.StringIO()):
            return evaluate(args)

    def test_all_network_types_return_the_same_schema_and_reproducible_metrics(self):
        fields = None
        for kind in ("dqn", "drqn", "qmix", "etd"):
            with self.subTest(kind=kind):
                checkpoint, config, _, _ = self.checkpoint(kind)
                original = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (checkpoint, config)}
                output = self.root / f"{kind}.json"
                first = self.evaluate(checkpoint, "--seed-base", "101", "--output", str(output))
                second = self.evaluate(checkpoint, "--seed-base", "101")
                self.assertEqual(first, second)
                self.assertEqual(first, json.loads(output.read_text()))
                self.assertEqual(first["model_type"], "qmix" if kind == "etd" else kind)
                self.assertEqual(first["episodes_completed"], 2)
                self.assertEqual(first["ep_len_mean"], 3)
                self.assertEqual([s["episode_seed"] for s in first["episode_stats"]], [101, 102])
                self.assertEqual(first["return_mean"], np.mean([s["episode_return"] for s in first["episode_stats"]]))
                if fields is None:
                    fields = set(first)
                self.assertEqual(set(first), fields)
                for path, digest in original.items():
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_qmix_results_match_the_existing_training_episode_runner(self):
        checkpoint, _, config, _ = self.checkpoint("qmix")
        report = self.evaluate(checkpoint, "--seed-base", "101")
        env = JackalMultiAgentEnv(config["env"])
        self.addCleanup(env.close)
        info = env.get_env_info()
        mac = BasicMAC(info["n_agents"], info["n_actions"], info["obs_shape"], config["agent"], torch.device("cpu"))
        mac.agent.load_state_dict(torch.load(checkpoint, weights_only=True)["agent"])
        mac.agent.eval()
        runner = EpisodeRunner(env, mac)
        for expected in report["episode_stats"]:
            seed = expected["episode_seed"]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            _, actual = runner.run(test_mode=True, epsilon=0.0, episode_seed=seed)
            for name, value in actual.items():
                self.assertEqual(expected[name], value, name)

    def test_periodic_drqn_checkpoint_uses_parent_run_config(self):
        checkpoint, config, _, _ = self.checkpoint("drqn", periodic=True)
        self.assertEqual(find_run_config(checkpoint, "drqn"), config)
        report = self.evaluate(checkpoint)
        self.assertEqual(report["config"], str(config.resolve()))
        self.assertEqual(report["ep_len_mean"], 3)

    def test_eval_env_and_cli_overrides_do_not_modify_configuration(self):
        checkpoint, path, config, _ = self.checkpoint("qmix")
        config["train"] = {"eval_env": {"max_steps": 1}}
        path.write_text(json.dumps(config))
        original = path.read_bytes()
        self.assertEqual(self.evaluate(checkpoint)["ep_len_mean"], 1)
        self.assertEqual(self.evaluate(checkpoint, "--max-steps", "2")["ep_len_mean"], 2)
        self.assertEqual(path.read_bytes(), original)

    def test_dqn_and_drqn_mask_actions_and_disable_gradients(self):
        for kind in ("dqn", "drqn"):
            with self.subTest(kind=kind):
                checkpoint, _, config, info = self.checkpoint(kind)
                _, weights = load_checkpoint(checkpoint)
                policy = create_policy(kind, weights, config, info, torch.device("cpu"))
                assert isinstance(policy, (DQNPolicy, DRQNPolicy))
                self.assertFalse(policy.network.training)
                self.assertTrue(all(not p.requires_grad for p in policy.network.parameters()))
                with torch.no_grad():
                    if isinstance(policy, DQNPolicy):
                        policy.network.fc3.weight.zero_()
                        policy.network.fc3.bias.zero_()
                        policy.network.fc3.bias[3] = 10
                        policy.network.fc3.bias[9] = 100
                    else:
                        for head in (policy.network.chassis_head, policy.network.turret_head, policy.network.fire_head):
                            head.weight.zero_(); head.bias.zero_()
                        policy.network.chassis_head.bias[3] = 10
                        policy.network.turret_head.bias[2] = 10
                        policy.network.fire_head.bias[1] = 100
                policy.reset()
                available = [1] * info["n_actions"]
                available[-1] = 0
                state = np.zeros(info["state_shape"], dtype=np.float32)
                self.assertEqual(policy.select_actions([], state, [available]), [3 if kind == "dqn" else 21])
                if isinstance(policy, DRQNPolicy):
                    assert policy.hidden_state is not None
                    self.assertGreater(float(policy.hidden_state.abs().sum()), 0)
                    self.assertFalse(policy.hidden_state.requires_grad)
                    policy.reset()
                    self.assertEqual(float(policy.hidden_state.abs().sum()), 0)

    def test_only_requested_episodes_create_videos_and_all_writers_close(self):
        checkpoint, _, _, _ = self.checkpoint("dqn")
        writers = []
        def create_writer(*args, **kwargs):
            writer = Mock()
            writers.append(writer)
            return writer
        with patch("environment.jackal_env.create_video_writer", side_effect=create_writer):
            self.evaluate(checkpoint)
            self.assertEqual(writers, [])
            self.evaluate(checkpoint, "--video-dir", str(self.root / "videos"))
        self.assertEqual(len(writers), 2)
        for writer in writers:
            self.assertEqual(writer.write.call_count, 4)
            writer.release.assert_called_once()

    def test_invalid_model_configuration_closes_environment_without_creating_video(self):
        checkpoint, path, config, _ = self.checkpoint("dqn")
        config["state_dim"] += 1
        path.write_text(json.dumps(config))
        created = []
        def create_env(**kwargs):
            env = JackalEnv(**kwargs)
            env.close = Mock(wraps=env.close)
            created.append(env)
            return env
        with patch("environment.marl_adapter.JackalEnv", side_effect=create_env), \
                patch("environment.jackal_env.create_video_writer") as writer:
            with self.assertRaisesRegex(ValueError, "state_dim"):
                self.evaluate(checkpoint, "--video-dir", str(self.root / "videos"))
            writer.assert_not_called()
        created[0].close.assert_called_once()
        self.assertTrue(created[0]._closed)

    def test_closing_native_window_stops_before_another_physics_step(self):
        with ModelEvaluationSession({"render_mode": "2D", "headless": False, "max_steps": 3}) as session:
            session.reset(seed=1)
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            with redirect_stdout(io.StringIO()), self.assertRaises(EvaluationStopped):
                session.step([0])
            self.assertEqual(session.env.get_runtime_info()["steps"], 0)
        self.assertTrue(session.env._closed)

    def test_cancelled_episode_is_excluded_from_the_report(self):
        checkpoint, _, _, _ = self.checkpoint("dqn")
        finished = {"episode_return": 12.0, "episode_length": 2, "battle_won": True,
                    "winner": "PLAYER", "reason": "elimination", "episode_seed": 101}
        with patch("training.test_model.run_episode", side_effect=[finished, EvaluationStopped()]):
            report = self.evaluate(checkpoint)
        self.assertTrue(report["interrupted"])
        self.assertEqual(report["episodes_completed"], 1)
        self.assertEqual(report["win_rate"], 1)
        self.assertEqual(report["return_mean"], 12)
        with patch("training.test_model.run_episode", side_effect=EvaluationStopped()):
            report = self.evaluate(checkpoint)
        self.assertEqual(report["episodes_completed"], 0)
        self.assertIsNone(report["win_rate"])

    def test_report_cannot_overwrite_weights_or_training_config(self):
        checkpoint, config, _, _ = self.checkpoint("dqn")
        for path in (checkpoint, config):
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "覆盖"):
                self.evaluate(checkpoint, "--output", str(path))
            self.assertEqual(path.read_bytes(), original)

    def test_metrics_distinguish_loss_from_draw(self):
        report = summarize([
            {"battle_won": True, "winner": "PLAYER", "episode_return": 10, "episode_length": 3},
            {"battle_won": False, "winner": "ENEMY", "episode_return": -4, "episode_length": 7},
            {"battle_won": False, "winner": "DRAW", "episode_return": 0, "episode_length": 5},
        ])
        self.assertEqual(report["win_rate"], 1 / 3)
        self.assertEqual(report["draw_rate"], 1 / 3)
        self.assertEqual(report["loss_rate"], 1 / 3)
        self.assertEqual(report["return_mean"], 2)
        self.assertEqual(report["ep_len_mean"], 5)

    def test_legacy_entries_share_arguments_and_reject_empty_episode_counts(self):
        from training import DQN_test_render, DRQN_test_render, test_qmix_marl2
        expected = vars(parse_args(["--checkpoint", "model.pth"]))
        for module in (DQN_test_render, DRQN_test_render, test_qmix_marl2):
            self.assertEqual(vars(module.parse_args(["--checkpoint", "model.pth"])), expected)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--checkpoint", "model.pth", "--episodes", "0"])


if __name__ == "__main__":
    unittest.main()
