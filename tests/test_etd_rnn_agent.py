import copy
import unittest

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from training.marl2.modules.agents.etd_rnn_agent import (
    ETDRNNAgent,
    _MPSCompatibleAdaptiveAvgPool2d,
)


class MPSCompatiblePoolingTests(unittest.TestCase):
    @staticmethod
    def _build_etd_agent():
        n_agents = 5
        n_enemies = 5
        n_actions = 10
        unit_type_dim = 2
        obs_shape = (
            ETDRNNAgent.SELF_DIM
            + unit_type_dim
            + (n_agents - 1) * (ETDRNNAgent.ALLY_DIM + unit_type_dim)
            + n_enemies * (ETDRNNAgent.ENEMY_DIM + unit_type_dim)
            + 3 * ETDRNNAgent.BULLET_DIM
            + 7 * 7 * ETDRNNAgent.MAP_CELL_DIM
            + ETDRNNAgent.TIME_DIM
        )
        input_shape = obs_shape + n_actions + n_agents
        agent = ETDRNNAgent(
            input_shape=input_shape,
            n_actions=n_actions,
            n_agents=n_agents,
            obs_shape=obs_shape,
            hidden_dim=64,
            entity_embed_dim=48,
            attn_heads=6,
            attn_layers=2,
            n_enemies=n_enemies,
            unit_type_dim=unit_type_dim,
        )
        return agent, input_shape, n_actions

    def test_seven_by_seven_pool_matches_adaptive_average_pool(self) -> None:
        inputs = torch.randn(4, 8, 7, 7)
        expected = F.adaptive_avg_pool2d(inputs, (3, 3))
        actual = _MPSCompatibleAdaptiveAvgPool2d((3, 3))(inputs)

        self.assertEqual(actual.shape, (4, 8, 3, 3))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_region_fallback_matches_nonuniform_adaptive_pool(self) -> None:
        inputs = torch.randn(2, 4, 5, 8)
        expected = F.adaptive_avg_pool2d(inputs, (3, 3))
        actual = _MPSCompatibleAdaptiveAvgPool2d._pool_by_regions(
            inputs,
            (3, 3),
        )

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_etd_agent_accepts_seven_by_seven_map_features(self) -> None:
        agent, input_shape, n_actions = self._build_etd_agent()

        q_values, next_hidden = agent(
            torch.randn(6, input_shape),
            agent.init_hidden(6, torch.device("cpu")),
        )

        self.assertEqual(q_values.shape, (6, n_actions))
        self.assertEqual(next_hidden.shape, (6, 64))

    def test_checkpointed_etd_recurrence_matches_gradients(self) -> None:
        direct_agent, input_shape, _ = self._build_etd_agent()
        checkpoint_agent = copy.deepcopy(direct_agent)
        sequence_inputs = torch.randn(3, 4, input_shape)

        def run_sequence(agent, use_checkpoint):
            hidden = agent.init_hidden(4, torch.device("cpu"))
            outputs = []
            for inputs in sequence_inputs:
                if use_checkpoint:
                    q_values, hidden = checkpoint(
                        agent,
                        inputs,
                        hidden,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    q_values, hidden = agent(inputs, hidden)
                outputs.append(q_values)
            stacked_outputs = torch.stack(outputs, dim=1)
            loss = stacked_outputs.square().mean() + hidden.square().mean()
            loss.backward()
            return stacked_outputs.detach(), hidden.detach()

        direct_outputs, direct_hidden = run_sequence(direct_agent, False)
        checkpoint_outputs, checkpoint_hidden = run_sequence(checkpoint_agent, True)

        torch.testing.assert_close(checkpoint_outputs, direct_outputs)
        torch.testing.assert_close(checkpoint_hidden, direct_hidden)
        for (direct_name, direct_parameter), (checkpoint_name, checkpoint_parameter) in zip(
            direct_agent.named_parameters(),
            checkpoint_agent.named_parameters(),
        ):
            self.assertEqual(direct_name, checkpoint_name)
            self.assertIsNotNone(direct_parameter.grad, msg=direct_name)
            self.assertIsNotNone(checkpoint_parameter.grad, msg=checkpoint_name)
            torch.testing.assert_close(
                checkpoint_parameter.grad,
                direct_parameter.grad,
                msg=f"gradient mismatch for {direct_name}",
            )


if __name__ == "__main__":
    unittest.main()
