import copy

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _should_checkpoint_online_activations(device):
    """Use recomputation on accelerator backends that retain long BPTT graphs."""
    return torch.device(device).type in {"cuda", "mps"}


class QMixLearner:
    def __init__(self, mac, mixer, algo_cfg, device):
        self.mac = mac
        self.mixer = mixer
        self.device = device
        self.algo_cfg = algo_cfg
        # Long recurrent episodes retain one autograd graph per timestep. ETD
        # attention/map activations can exhaust a laptop CUDA GPU or the MPS
        # allocator before the first backward pass. Trade recomputation for
        # bounded activation storage on accelerator backends. This does not
        # truncate BPTT or change the model, loss, batch size, or optimizer.
        self._checkpoint_online_activations = _should_checkpoint_online_activations(device)

        self.gamma = float(algo_cfg.get("gamma", 0.99))
        self.lr = float(algo_cfg.get("lr", 5e-4))
        self.grad_norm_clip = float(algo_cfg.get("grad_norm_clip", 10.0))
        self.double_q = bool(algo_cfg.get("double_q", True))
        self.target_update_interval = int(algo_cfg.get("target_update_interval", 200))
        self.target_update_mode = str(algo_cfg.get("target_update_mode", "hard")).lower()
        self.target_update_tau = float(algo_cfg.get("target_update_tau", 0.005))
        if self.target_update_mode not in {"hard", "soft"}:
            raise ValueError(
                f"Unsupported target_update_mode={self.target_update_mode!r}. "
                "Expected 'hard' or 'soft'."
            )
        if not 0.0 < self.target_update_tau <= 1.0:
            raise ValueError(f"target_update_tau must be in (0, 1], got {self.target_update_tau}")

        self.target_agent = copy.deepcopy(self.mac.agent).to(self.device)
        self.target_mixer = copy.deepcopy(self.mixer).to(self.device)
        self.target_agent.requires_grad_(False)
        self.target_mixer.requires_grad_(False)

        params = list(self.mac.agent.parameters()) + list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(params, lr=self.lr, alpha=0.99, eps=1e-5)

        self.train_steps = 0

    def _online_forward(self, obs, prev_actions, hidden_states):
        if not self._checkpoint_online_activations or not torch.is_grad_enabled():
            return self.mac.forward_train(obs, prev_actions, hidden_states)
        return checkpoint(
            self.mac.forward_train,
            obs,
            prev_actions,
            hidden_states,
            use_reentrant=False,
            preserve_rng_state=True,
        )

    @torch.no_grad()
    def _target_forward(self, obs, prev_actions, hidden_states):
        bs = obs.shape[0]
        inputs = self.mac._build_inputs(obs, prev_actions)
        flat_inputs = inputs.reshape(bs * self.mac.n_agents, -1)
        flat_hidden = hidden_states.reshape(bs * self.mac.n_agents, self.mac.hidden_dim)
        flat_q, flat_next_hidden = self.target_agent(flat_inputs, flat_hidden)
        q = flat_q.view(bs, self.mac.n_agents, self.mac.n_actions)
        next_hidden = flat_next_hidden.view(bs, self.mac.n_agents, self.mac.hidden_dim)
        return q, next_hidden

    def train(self, batch):
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        states = torch.tensor(batch["state"], dtype=torch.float32, device=self.device)
        avail_actions = torch.tensor(batch["avail_actions"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.long, device=self.device)
        rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        terminated = torch.tensor(batch["terminated"], dtype=torch.float32, device=self.device)
        filled = torch.tensor(batch["filled"], dtype=torch.float32, device=self.device)

        filled_steps = filled.squeeze(-1).sum(dim=1)
        if filled_steps.numel() == 0:
            return {
                "loss": 0.0,
                "loss_td": 0.0,
                "grad_norm": 0.0,
                "q_tot_mean": 0.0,
                "q_taken_mean": 0.0,
                "target_mean": 0.0,
                "td_error_abs": 0.0,
            }

        max_filled_t = int(filled_steps.max().item())
        if max_filled_t <= 0:
            return {
                "loss": 0.0,
                "loss_td": 0.0,
                "grad_norm": 0.0,
                "q_tot_mean": 0.0,
                "q_taken_mean": 0.0,
                "target_mean": 0.0,
                "td_error_abs": 0.0,
            }

        obs = obs[:, : max_filled_t + 1]
        states = states[:, : max_filled_t + 1]
        avail_actions = avail_actions[:, : max_filled_t + 1]
        actions = actions[:, :max_filled_t]
        rewards = rewards[:, :max_filled_t]
        terminated = terminated[:, :max_filled_t]
        filled = filled[:, :max_filled_t]

        bs = obs.shape[0]
        max_t = actions.shape[1]
        n_agents = self.mac.n_agents
        n_actions = self.mac.n_actions

        actions_onehot = F.one_hot(actions.squeeze(-1), num_classes=n_actions).float()
        last_actions = torch.zeros(
            bs,
            max_t + 1,
            n_agents,
            n_actions,
            dtype=torch.float32,
            device=self.device,
        )
        last_actions[:, 1:] = actions_onehot

        hidden_eval = torch.zeros(bs, n_agents, self.mac.hidden_dim, device=self.device)
        hidden_target = torch.zeros(bs, n_agents, self.mac.hidden_dim, device=self.device)
        mixer_requires_hidden = bool(
            getattr(self.mixer, "requires_hidden_states", False)
        )

        mac_out = []
        target_out = []
        eval_hidden_history = []
        target_hidden_history = []
        if mixer_requires_hidden:
            eval_hidden_history.append(hidden_eval)
            target_hidden_history.append(hidden_target)
        for t in range(max_t + 1):
            q_eval, hidden_eval = self._online_forward(
                obs[:, t],
                last_actions[:, t],
                hidden_eval,
            )
            q_target, hidden_target = self._target_forward(obs[:, t], last_actions[:, t], hidden_target)
            mac_out.append(q_eval)
            target_out.append(q_target)
            if mixer_requires_hidden:
                eval_hidden_history.append(hidden_eval)
                target_hidden_history.append(hidden_target)

        mac_out = torch.stack(mac_out, dim=1)  # [bs, t+1, n_agents, n_actions]
        target_out = torch.stack(target_out, dim=1)
        if mixer_requires_hidden:
            eval_hidden_out = torch.stack(eval_hidden_history, dim=1)
            target_hidden_out = torch.stack(target_hidden_history, dim=1)

        chosen_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        avail_next = avail_actions[:, 1:]
        no_avail_next = avail_next.sum(dim=3, keepdim=True) <= 0
        if no_avail_next.any():
            avail_next = avail_next.clone()
            avail_next[..., 0:1] = torch.where(
                no_avail_next,
                torch.ones_like(avail_next[..., 0:1]),
                avail_next[..., 0:1],
            )
        target_next_qvals = target_out[:, 1:].clone()
        target_next_qvals[avail_next == 0] = -1e10

        if self.double_q:
            mac_next_qvals = mac_out[:, 1:].clone().detach()
            mac_next_qvals[avail_next == 0] = -1e10
            greedy_next_actions = mac_next_qvals.argmax(dim=3, keepdim=True)
            target_max_qvals = torch.gather(target_next_qvals, 3, greedy_next_actions).squeeze(3)
        else:
            target_max_qvals = target_next_qvals.max(dim=3)[0]

        if mixer_requires_hidden:
            eval_mixer_hidden = eval_hidden_out[:, 1:-1]
            target_mixer_hidden = target_hidden_out[:, 2:]
            if bool(self.algo_cfg.get("detach_mixer_hidden", False)):
                eval_mixer_hidden = eval_mixer_hidden.detach()
                target_mixer_hidden = target_mixer_hidden.detach()
            q_tot = self.mixer(chosen_qvals, states[:, :-1], eval_mixer_hidden).squeeze(-1)
            with torch.no_grad():
                target_q_tot = self.target_mixer(
                    target_max_qvals,
                    states[:, 1:],
                    target_mixer_hidden,
                ).squeeze(-1)
        else:
            q_tot = self.mixer(chosen_qvals, states[:, :-1]).squeeze(-1)
            with torch.no_grad():
                target_q_tot = self.target_mixer(
                    target_max_qvals,
                    states[:, 1:],
                ).squeeze(-1)

        targets = rewards.squeeze(-1) + self.gamma * (1.0 - terminated.squeeze(-1)) * target_q_tot

        mask = filled.squeeze(-1)
        # Defensive masking: ensure timesteps after true termination do not contribute to TD loss.
        mask[:, 1:] = mask[:, 1:] * (1.0 - terminated.squeeze(-1)[:, :-1])
        td_error = q_tot - targets.detach()
        masked_td_error = td_error * mask
        mask_sum = mask.sum().clamp(min=1.0)

        chosen_mask = mask.unsqueeze(-1)
        chosen_mask_sum = chosen_mask.sum().clamp(min=1.0)

        loss = (masked_td_error ** 2).sum() / mask_sum

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.mac.agent.parameters()) + list(self.mixer.parameters()),
            self.grad_norm_clip,
        )
        self.optimizer.step()

        self.train_steps += 1
        if self.target_update_mode == "soft":
            self.soft_update_targets(self.target_update_tau)
        elif self.train_steps % self.target_update_interval == 0:
            self.update_targets()

        stats = {
            "loss": float(loss.item()),
            "loss_td": float(loss.item()),
            "grad_norm": float(grad_norm),
            "q_tot_mean": float((q_tot * mask).sum().item() / mask_sum.item()),
            "q_taken_mean": float((chosen_qvals * chosen_mask).sum().item() / chosen_mask_sum.item()),
            "target_mean": float((targets * mask).sum().item() / mask_sum.item()),
            "td_error_abs": float(masked_td_error.abs().sum().item() / mask_sum.item()),
        }
        return stats

    def update_targets(self):
        self.target_agent.load_state_dict(self.mac.agent.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def soft_update_targets(self, tau):
        with torch.no_grad():
            for target_param, param in zip(self.target_agent.parameters(), self.mac.agent.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def save_models(self, save_path, meta=None):
        payload = {
            "agent": self.mac.agent.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_agent": self.target_agent.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "meta": meta or {},
        }
        torch.save(payload, save_path)

    def load_models(self, load_path):
        payload = torch.load(load_path, map_location=self.device)
        self.mac.agent.load_state_dict(payload["agent"])
        self.mixer.load_state_dict(payload["mixer"])
        self.target_agent.load_state_dict(payload.get("target_agent", payload["agent"]))
        self.target_mixer.load_state_dict(payload.get("target_mixer", payload["mixer"]))

        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.train_steps = int(payload.get("train_steps", 0))
        return payload
