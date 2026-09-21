import copy
import unittest

import numpy as np
import torch

from training.marl2.controllers.basic_mac import BasicMAC
from training.marl2.learners.qmix_learner import (
    QMixLearner,
    _should_checkpoint_online_activations,
)
from training.marl2.modules.mixers.qmix import QMixer


class QMixLearnerGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.n_agents = 2
        self.n_actions = 3
        self.obs_shape = 5
        self.state_shape = 7
        self.mac = BasicMAC(
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            obs_shape=self.obs_shape,
            agent_cfg={
                "name": "rnn",
                "rnn_hidden_dim": 16,
                "use_last_action": True,
                "use_agent_id": True,
            },
            device=torch.device("cpu"),
        )
        self.mixer = QMixer(
            n_agents=self.n_agents,
            state_dim=self.state_shape,
            embed_dim=8,
        )
        self.learner = QMixLearner(
            self.mac,
            self.mixer,
            {
                "gamma": 0.99,
                "lr": 1e-3,
                "target_update_mode": "soft",
                "target_update_tau": 0.01,
            },
            torch.device("cpu"),
        )

    def test_target_networks_never_build_gradients(self) -> None:
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.learner.target_agent.parameters())
        )
        self.assertTrue(
            all(not parameter.requires_grad for parameter in self.learner.target_mixer.parameters())
        )

        batch_size = 4
        observations = torch.randn(batch_size, self.n_agents, self.obs_shape)
        previous_actions = torch.zeros(
            batch_size,
            self.n_agents,
            self.n_actions,
        )
        hidden = torch.zeros(batch_size, self.n_agents, self.mac.hidden_dim)
        q_values, next_hidden = self.learner._target_forward(
            observations,
            previous_actions,
            hidden,
        )

        self.assertFalse(q_values.requires_grad)
        self.assertFalse(next_hidden.requires_grad)

    def test_training_update_still_backpropagates_online_network(self) -> None:
        batch_size = 2
        sequence_length = 4
        batch = {
            "obs": np.random.randn(
                batch_size,
                sequence_length + 1,
                self.n_agents,
                self.obs_shape,
            ).astype(np.float32),
            "state": np.random.randn(
                batch_size,
                sequence_length + 1,
                self.state_shape,
            ).astype(np.float32),
            "avail_actions": np.ones(
                (
                    batch_size,
                    sequence_length + 1,
                    self.n_agents,
                    self.n_actions,
                ),
                dtype=np.float32,
            ),
            "actions": np.random.randint(
                0,
                self.n_actions,
                size=(batch_size, sequence_length, self.n_agents, 1),
                dtype=np.int64,
            ),
            "reward": np.random.randn(
                batch_size,
                sequence_length,
                1,
            ).astype(np.float32),
            "terminated": np.zeros(
                (batch_size, sequence_length, 1),
                dtype=np.float32,
            ),
            "filled": np.ones(
                (batch_size, sequence_length, 1),
                dtype=np.float32,
            ),
        }

        # Exercise the accelerator memory-saving path on CPU as well;
        # checkpointing is backend independent, while CUDA/MPS may be absent.
        self.learner._checkpoint_online_activations = True
        stats = self.learner.train(batch)

        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertTrue(any(parameter.grad is not None for parameter in self.mac.agent.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.learner.target_agent.parameters()))

    def test_checkpointed_recurrence_matches_outputs_and_gradients(self) -> None:
        direct_mac = copy.deepcopy(self.mac)
        checkpoint_mac = copy.deepcopy(self.mac)
        direct_learner = QMixLearner(
            direct_mac,
            copy.deepcopy(self.mixer),
            self.learner.algo_cfg,
            torch.device("cpu"),
        )
        checkpoint_learner = QMixLearner(
            checkpoint_mac,
            copy.deepcopy(self.mixer),
            self.learner.algo_cfg,
            torch.device("cpu"),
        )
        checkpoint_learner._checkpoint_online_activations = True

        batch_size = 2
        sequence_length = 5
        observations = torch.randn(
            batch_size,
            sequence_length,
            self.n_agents,
            self.obs_shape,
        )
        previous_actions = torch.randn(
            batch_size,
            sequence_length,
            self.n_agents,
            self.n_actions,
        )

        def run_sequence(learner):
            hidden = torch.zeros(batch_size, self.n_agents, self.mac.hidden_dim)
            outputs = []
            for timestep in range(sequence_length):
                q_values, hidden = learner._online_forward(
                    observations[:, timestep],
                    previous_actions[:, timestep],
                    hidden,
                )
                outputs.append(q_values)
            stacked_outputs = torch.stack(outputs, dim=1)
            loss = stacked_outputs.square().mean() + hidden.square().mean()
            loss.backward()
            return stacked_outputs.detach(), hidden.detach()

        direct_outputs, direct_hidden = run_sequence(direct_learner)
        checkpoint_outputs, checkpoint_hidden = run_sequence(checkpoint_learner)

        torch.testing.assert_close(checkpoint_outputs, direct_outputs)
        torch.testing.assert_close(checkpoint_hidden, direct_hidden)
        direct_parameters = dict(direct_mac.agent.named_parameters())
        checkpoint_parameters = dict(checkpoint_mac.agent.named_parameters())
        self.assertEqual(direct_parameters.keys(), checkpoint_parameters.keys())
        for name, direct_parameter in direct_parameters.items():
            checkpoint_parameter = checkpoint_parameters[name]
            self.assertIsNotNone(direct_parameter.grad, msg=name)
            self.assertIsNotNone(checkpoint_parameter.grad, msg=name)
            torch.testing.assert_close(
                checkpoint_parameter.grad,
                direct_parameter.grad,
                msg=f"gradient mismatch for {name}",
            )

    def test_long_bptt_checkpointing_is_enabled_on_accelerators(self) -> None:
        self.assertFalse(_should_checkpoint_online_activations(torch.device("cpu")))
        self.assertTrue(_should_checkpoint_online_activations(torch.device("cuda")))
        self.assertTrue(_should_checkpoint_online_activations(torch.device("mps")))


if __name__ == "__main__":
    unittest.main()
