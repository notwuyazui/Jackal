import numpy as np
import torch
from typing import Optional

from training.marl2.modules.agents.etd_rnn_agent import ETDRNNAgent
from training.marl2.modules.agents.rnn_agent import RNNAgent


class BasicMAC:
    """Shared-agent multi-agent controller (PyMARL-style)."""

    def __init__(self, n_agents, n_actions, obs_shape, agent_cfg, device):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.device = device

        self.hidden_dim = int(agent_cfg.get("rnn_hidden_dim", 128))
        self.use_last_action = bool(agent_cfg.get("use_last_action", True))
        self.use_agent_id = bool(agent_cfg.get("use_agent_id", True))

        input_shape = obs_shape
        if self.use_last_action:
            input_shape += n_actions
        if self.use_agent_id:
            input_shape += n_agents

        agent_name = str(agent_cfg.get("name", "rnn")).lower()
        if agent_name == "rnn":
            self.agent = RNNAgent(
                input_shape,
                n_actions,
                hidden_dim=self.hidden_dim,
                input_hidden_dim=agent_cfg.get("rnn_input_hidden_dim", None),
                input_layers=int(agent_cfg.get("rnn_input_layers", 1)),
            ).to(self.device)
        elif agent_name == "etd_rnn":
            self.agent = ETDRNNAgent(
                input_shape=input_shape,
                n_actions=n_actions,
                n_agents=n_agents,
                obs_shape=obs_shape,
                hidden_dim=self.hidden_dim,
                use_last_action=self.use_last_action,
                use_agent_id=self.use_agent_id,
                entity_embed_dim=int(agent_cfg.get("etd_entity_embed_dim", self.hidden_dim)),
                attn_heads=int(agent_cfg.get("etd_attn_heads", 4)),
                attn_layers=int(agent_cfg.get("etd_attn_layers", 1)),
                attn_dropout=float(agent_cfg.get("etd_attn_dropout", 0.0)),
                max_obs_bullets=int(agent_cfg.get("max_obs_bullets", 3)),
                n_enemies=agent_cfg.get("n_enemies", None),
                unit_type_dim=int(agent_cfg.get("unit_type_dim", 0)),
                log_attention=bool(agent_cfg.get("etd_log_attention", False)),
                map_fusion=str(agent_cfg.get("etd_map_fusion", "tokens")),
                map_gate_init=float(agent_cfg.get("etd_map_gate_init", -2.0)),
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported agent name={agent_name!r}. Expected 'rnn' or 'etd_rnn'.")

        self.hidden_states: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    def init_hidden(self, batch_size):
        # Hidden state is stored per (batch, agent).
        self.hidden_states = self.agent.init_hidden(batch_size * self.n_agents, self.device)
        self.prev_actions = torch.zeros(
            batch_size,
            self.n_agents,
            self.n_actions,
            device=self.device,
            dtype=torch.float32,
        )

    def _build_inputs(self, obs, prev_actions):
        # obs: [bs, n_agents, obs_shape]
        # prev_actions: [bs, n_agents, n_actions]
        pieces = [obs]
        if self.use_last_action:
            pieces.append(prev_actions)
        if self.use_agent_id:
            bs = obs.shape[0]
            eye = torch.eye(self.n_agents, device=self.device, dtype=torch.float32)
            ids = eye.unsqueeze(0).expand(bs, -1, -1)
            pieces.append(ids)
        return torch.cat(pieces, dim=-1)

    def forward_train(self, obs, prev_actions, hidden_states):
        # obs: [bs, n_agents, obs_shape]
        # prev_actions: [bs, n_agents, n_actions]
        bs = obs.shape[0]
        inputs = self._build_inputs(obs, prev_actions)
        flat_inputs = inputs.reshape(bs * self.n_agents, -1)
        flat_hidden = hidden_states.reshape(bs * self.n_agents, self.hidden_dim)

        flat_q, flat_next_hidden = self.agent(flat_inputs, flat_hidden)
        q = flat_q.view(bs, self.n_agents, self.n_actions)
        next_hidden = flat_next_hidden.view(bs, self.n_agents, self.hidden_dim)
        return q, next_hidden

    def select_actions(self, obs, avail_actions, epsilon, test_mode=False):
        actions = self.select_actions_batch(
            obs_batch=np.array(obs, dtype=np.float32)[None, ...],
            avail_actions_batch=np.array(avail_actions, dtype=np.float32)[None, ...],
            epsilon=epsilon,
            test_mode=test_mode,
        )
        return actions[0].tolist()

    def select_actions_batch(self, obs_batch, avail_actions_batch, epsilon, test_mode=False, active_envs=None):
        # obs_batch: [bs, n_agents, obs_shape]
        # avail_actions_batch: [bs, n_agents, n_actions]
        obs_tensor = torch.tensor(np.array(obs_batch), dtype=torch.float32, device=self.device)
        avail_tensor = torch.tensor(np.array(avail_actions_batch), dtype=torch.float32, device=self.device)

        batch_size = obs_tensor.shape[0]
        if self.prev_actions is None or self.prev_actions.shape[0] != batch_size:
            self.init_hidden(batch_size=batch_size)
        assert self.hidden_states is not None
        assert self.prev_actions is not None

        if active_envs is None:
            active_idx: np.ndarray = np.arange(batch_size, dtype=np.int64)
        else:
            active_idx = np.array(list(active_envs), dtype=np.int64)

        chosen_actions = np.zeros((batch_size, self.n_agents), dtype=np.int64)

        if active_idx.size > 0:
            hidden = self.hidden_states.view(batch_size, self.n_agents, self.hidden_dim)
            active_idx_t = torch.tensor(active_idx, dtype=torch.long, device=self.device)

            obs_active = obs_tensor.index_select(0, active_idx_t)
            avail_active = avail_tensor.index_select(0, active_idx_t)
            prev_active = self.prev_actions.index_select(0, active_idx_t)
            hidden_active = hidden.index_select(0, active_idx_t)

            with torch.no_grad():
                q_values, next_hidden = self.forward_train(obs_active, prev_active, hidden_active)
            hidden[active_idx_t] = next_hidden
            self.hidden_states = hidden.reshape(batch_size * self.n_agents, self.hidden_dim)

            q_values[avail_active == 0] = -1e10
            greedy_actions = q_values.argmax(dim=-1).detach().cpu().numpy()  # [n_active, n_agents]
            avail_active_np = np.array(avail_actions_batch)[active_idx]

            chosen_active = np.zeros((active_idx.size, self.n_agents), dtype=np.int64)
            if test_mode:
                chosen_active = greedy_actions
            else:
                for local_b in range(active_idx.size):
                    for agent_id in range(self.n_agents):
                        valid = np.where(avail_active_np[local_b, agent_id] > 0)[0]
                        if len(valid) == 0:
                            chosen_active[local_b, agent_id] = 0
                            continue
                        if np.random.rand() < epsilon:
                            chosen_active[local_b, agent_id] = int(np.random.choice(valid))
                        else:
                            chosen_active[local_b, agent_id] = int(greedy_actions[local_b, agent_id])

            chosen_actions[active_idx] = chosen_active

            # 仅更新未终止环境的 last_action，保持与 PyMARL 的 envs_not_terminated 语义一致。
            for local_b, b in enumerate(active_idx.tolist()):
                self.prev_actions[b].zero_()
                for agent_id in range(self.n_agents):
                    action = int(chosen_active[local_b, agent_id])
                    self.prev_actions[b, agent_id, action] = 1.0

        return chosen_actions
