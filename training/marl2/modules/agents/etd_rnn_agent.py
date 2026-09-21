import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MPSCompatibleAdaptiveAvgPool2d(nn.Module):
    """Adaptive average pooling without MPS non-divisible-size failures."""

    def __init__(self, output_size):
        super().__init__()
        self.output_size = (int(output_size[0]), int(output_size[1]))

    @staticmethod
    def _fixed_pool_params(input_size, output_size):
        quotient, remainder = divmod(input_size, output_size)
        if remainder == 0:
            return quotient, quotient
        if remainder == 1 and quotient > 0:
            return quotient + 1, quotient
        return None

    @staticmethod
    def _pool_by_regions(inputs, output_size):
        output_height, output_width = output_size
        input_height, input_width = inputs.shape[-2:]
        rows = []
        for row in range(output_height):
            row_start = row * input_height // output_height
            row_end = math.ceil((row + 1) * input_height / output_height)
            cells = []
            for column in range(output_width):
                column_start = column * input_width // output_width
                column_end = math.ceil(
                    (column + 1) * input_width / output_width
                )
                cells.append(
                    inputs[
                        ...,
                        row_start:row_end,
                        column_start:column_end,
                    ].mean(dim=(-2, -1))
                )
            rows.append(torch.stack(cells, dim=-1))
        return torch.stack(rows, dim=-2)

    def forward(self, inputs):
        output_height, output_width = self.output_size
        input_height, input_width = inputs.shape[-2:]
        height_params = self._fixed_pool_params(input_height, output_height)
        width_params = self._fixed_pool_params(input_width, output_width)
        if height_params is not None and width_params is not None:
            return F.avg_pool2d(
                inputs,
                kernel_size=(height_params[0], width_params[0]),
                stride=(height_params[1], width_params[1]),
            )
        if inputs.device.type != "mps":
            return F.adaptive_avg_pool2d(inputs, self.output_size)
        return self._pool_by_regions(inputs, self.output_size)


class ETDRNNAgent(nn.Module):
    """JACKAL entity-decoupled temporal agent with the standard QMIX Q interface."""

    SELF_DIM = 10
    ALLY_DIM = 11
    ENEMY_DIM = 13
    BULLET_DIM = 9
    MAP_CELL_DIM = 4
    MAP_POOL_SIZE = 3
    TIME_DIM = 1

    def __init__(
        self,
        input_shape,
        n_actions,
        n_agents,
        obs_shape,
        hidden_dim=128,
        use_last_action=True,
        use_agent_id=True,
        entity_embed_dim=64,
        attn_heads=4,
        attn_layers=1,
        attn_dropout=0.0,
        max_obs_bullets=3,
        n_enemies=None,
        unit_type_dim=0,
        log_attention=False,
        map_fusion="tokens",
        map_gate_init=-2.0,
    ):
        super().__init__()
        self.input_shape = int(input_shape)
        self.n_actions = int(n_actions)
        self.n_agents = int(n_agents)
        self.obs_shape = int(obs_shape)
        self.hidden_dim = int(hidden_dim)
        self.use_last_action = bool(use_last_action)
        self.use_agent_id = bool(use_agent_id)
        self.entity_embed_dim = int(entity_embed_dim)
        self.attn_heads = int(attn_heads)
        self.attn_layers = int(attn_layers)
        self.max_obs_bullets = int(max_obs_bullets)
        self.unit_type_dim = int(unit_type_dim)
        self.log_attention = bool(log_attention)
        self.map_fusion = str(map_fusion).lower()
        if self.map_fusion not in {"tokens", "gated", "none"}:
            raise ValueError(
                f"Unsupported map_fusion={self.map_fusion!r}. "
                "Expected 'tokens', 'gated', or 'none'."
            )

        self.self_dim = self.SELF_DIM + self.unit_type_dim
        self.ally_dim = self.ALLY_DIM + self.unit_type_dim
        self.enemy_dim = self.ENEMY_DIM + self.unit_type_dim

        if self.entity_embed_dim % self.attn_heads != 0:
            raise ValueError("entity_embed_dim must be divisible by attn_heads")

        self.n_allies = self.n_agents - 1
        self.n_enemies = self._resolve_n_enemies(n_enemies)
        self.map_dim = self._resolve_map_dim()
        self.has_map_features = self.map_dim > 0
        self.map_cell_dim = self.MAP_CELL_DIM if self.has_map_features and self.map_dim % self.MAP_CELL_DIM == 0 else self.map_dim
        self.map_grid_size = self._resolve_map_grid_size()
        self.map_pool_size = min(self.MAP_POOL_SIZE, self.map_grid_size) if self.has_map_features else 0
        self.n_map_tokens = self.map_pool_size * self.map_pool_size if self.has_map_features else 0
        self.map_tokens_in_attention = self.has_map_features and self.map_fusion == "tokens"
        self.use_gated_map_context = self.has_map_features and self.map_fusion == "gated"
        self.n_attn_map_tokens = self.n_map_tokens if self.map_tokens_in_attention else 0
        self.n_entities = 1 + self.n_allies + self.n_enemies + self.max_obs_bullets + self.n_attn_map_tokens

        self.last_action_dim = self.n_actions if self.use_last_action else 0
        self.agent_id_dim = self.n_agents if self.use_agent_id else 0
        self.self_input_dim = (
            self.self_dim
            + self.TIME_DIM
            + self.last_action_dim
            + self.agent_id_dim
        )

        self.self_encoder = nn.Sequential(
            nn.Linear(self.self_input_dim, self.entity_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.entity_embed_dim, self.entity_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.ally_encoder = nn.Sequential(
            nn.Linear(self.ally_dim, self.entity_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.entity_embed_dim, self.entity_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(self.enemy_dim, self.entity_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.entity_embed_dim, self.entity_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.bullet_encoder = nn.Sequential(
            nn.Linear(self.BULLET_DIM, self.entity_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.entity_embed_dim, self.entity_embed_dim),
            nn.ReLU(inplace=True),
        )
        if self.has_map_features:
            conv_hidden = max(32, self.entity_embed_dim // 2)
            self.map_encoder = nn.Sequential(
                nn.Conv2d(self.map_cell_dim, conv_hidden, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(conv_hidden, self.entity_embed_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                _MPSCompatibleAdaptiveAvgPool2d(
                    (self.map_pool_size, self.map_pool_size)
                ),
            )
            self.map_pos_embed = nn.Parameter(
                torch.zeros(1, self.n_map_tokens, self.entity_embed_dim)
            )
            nn.init.normal_(self.map_pos_embed, mean=0.0, std=0.02)
            if self.use_gated_map_context:
                self.map_context_proj = nn.Sequential(
                    nn.Linear(self.entity_embed_dim, self.entity_embed_dim),
                    nn.ReLU(inplace=True),
                    nn.LayerNorm(self.entity_embed_dim),
                )
                self.map_gate = nn.Linear(
                    self.self_input_dim + self.entity_embed_dim,
                    self.entity_embed_dim,
                )
                nn.init.constant_(self.map_gate.bias, float(map_gate_init))
            else:
                self.map_context_proj = None
                self.map_gate = None
        else:
            self.map_encoder = None
            self.map_pos_embed = None
            self.map_context_proj = None
            self.map_gate = None
        self.entity_norm = nn.LayerNorm(self.entity_embed_dim)
        self.spatio_attn = nn.MultiheadAttention(
            embed_dim=self.entity_embed_dim,
            num_heads=self.attn_heads,
            dropout=float(attn_dropout),
            batch_first=True,
        )
        extra_layers = max(0, self.attn_layers - 1)
        self.extra_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.entity_embed_dim,
                    num_heads=self.attn_heads,
                    dropout=float(attn_dropout),
                    batch_first=True,
                )
                for _ in range(extra_layers)
            ]
        )
        self.extra_attn_norms = nn.ModuleList(
            [nn.LayerNorm(self.entity_embed_dim) for _ in range(extra_layers)]
        )
        self.extra_ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.entity_embed_dim, self.entity_embed_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.entity_embed_dim * 2, self.entity_embed_dim),
                )
                for _ in range(extra_layers)
            ]
        )
        self.extra_ffn_norms = nn.ModuleList(
            [nn.LayerNorm(self.entity_embed_dim) for _ in range(extra_layers)]
        )

        self.rnn_input_dim = self.self_input_dim + self.entity_embed_dim
        self.rnn_in = nn.Linear(self.rnn_input_dim, self.hidden_dim)
        self.rnn = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim + self.entity_embed_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.q_out = nn.Linear(self.hidden_dim, self.n_actions)

        self.latest_attn_weights = None
        self.latest_attn_scores = None
        self.latest_entity_order = {
            "self_tokens": 1,
            "ally_tokens": self.n_allies,
            "enemy_tokens": self.n_enemies,
            "bullet_tokens": self.max_obs_bullets,
            "map_tokens": self.n_attn_map_tokens,
            "map_fusion": self.map_fusion,
        }

    def _resolve_n_enemies(self, n_enemies):
        if n_enemies is not None:
            return int(n_enemies)

        known = (
            self.self_dim
            + self.n_allies * self.ally_dim
            + self.max_obs_bullets * self.BULLET_DIM
            + self.TIME_DIM
        )
        remaining = self.obs_shape - known
        if remaining < 0 or remaining % self.enemy_dim != 0:
            raise ValueError(
                "Cannot infer n_enemies from JACKAL obs layout: "
                f"obs_shape={self.obs_shape}, n_agents={self.n_agents}, "
                f"max_obs_bullets={self.max_obs_bullets}, unit_type_dim={self.unit_type_dim}"
            )
        return remaining // self.enemy_dim

    def _resolve_map_dim(self):
        known = (
            self.self_dim
            + self.n_allies * self.ally_dim
            + self.n_enemies * self.enemy_dim
            + self.max_obs_bullets * self.BULLET_DIM
            + self.TIME_DIM
        )
        remaining = self.obs_shape - known
        if remaining < 0:
            raise ValueError(
                "Cannot resolve JACKAL map feature tail: "
                f"obs_shape={self.obs_shape}, known={known}"
            )
        return remaining

    def _resolve_map_grid_size(self):
        if self.map_dim <= 0:
            return 0
        if self.map_dim % self.map_cell_dim != 0:
            raise ValueError(
                f"Cannot split JACKAL map features: map_dim={self.map_dim}, "
                f"map_cell_dim={self.map_cell_dim}"
            )
        n_cells = self.map_dim // self.map_cell_dim
        grid = int(math.isqrt(n_cells))
        if grid * grid != n_cells:
            raise ValueError(
                "JACKAL map features must form a square local grid for CNN encoding: "
                f"map_dim={self.map_dim}, map_cell_dim={self.map_cell_dim}, n_cells={n_cells}"
            )
        return grid

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _split_inputs(self, inputs):
        batch_size = inputs.shape[0]
        obs = inputs[:, : self.obs_shape]
        tail = inputs[:, self.obs_shape :]

        tail_idx = 0
        last_action = None
        agent_id = None
        if self.use_last_action:
            last_action = tail[:, tail_idx : tail_idx + self.n_actions]
            tail_idx += self.n_actions
        if self.use_agent_id:
            agent_id = tail[:, tail_idx : tail_idx + self.n_agents]

        obs_idx = 0
        self_feats = obs[:, obs_idx : obs_idx + self.self_dim]
        obs_idx += self.self_dim

        ally_total = self.n_allies * self.ally_dim
        ally_feats = obs[:, obs_idx : obs_idx + ally_total].view(
            batch_size,
            self.n_allies,
            self.ally_dim,
        )
        obs_idx += ally_total

        enemy_total = self.n_enemies * self.enemy_dim
        enemy_feats = obs[:, obs_idx : obs_idx + enemy_total].view(
            batch_size,
            self.n_enemies,
            self.enemy_dim,
        )
        obs_idx += enemy_total

        bullet_total = self.max_obs_bullets * self.BULLET_DIM
        bullet_feats = obs[:, obs_idx : obs_idx + bullet_total].view(
            batch_size,
            self.max_obs_bullets,
            self.BULLET_DIM,
        )
        obs_idx += bullet_total

        remaining_obs = obs[:, obs_idx:]
        if remaining_obs.shape[-1] < self.TIME_DIM:
            raise ValueError("JACKAL obs is missing the final time feature")
        if self.map_dim > 0:
            map_feats = remaining_obs[:, : self.map_dim].view(
                batch_size,
                self.map_grid_size,
                self.map_grid_size,
                self.map_cell_dim,
            )
        else:
            map_feats = None
        time_feats = remaining_obs[:, self.map_dim : self.map_dim + self.TIME_DIM]

        self_parts = [self_feats, time_feats]
        if last_action is not None:
            self_parts.append(last_action)
        if agent_id is not None:
            self_parts.append(agent_id)
        self_input = torch.cat(self_parts, dim=-1)

        return self_input, ally_feats, enemy_feats, bullet_feats, map_feats

    def _entity_padding_mask(self, ally_feats, enemy_feats, bullet_feats, map_feats):
        batch_size = ally_feats.shape[0]
        self_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=ally_feats.device)

        masks = [self_mask]
        if self.n_allies > 0:
            masks.append(ally_feats[..., 1] <= 0.0)
        if self.n_enemies > 0:
            masks.append(enemy_feats[..., 1] <= 0.0)
        if self.max_obs_bullets > 0:
            bullet_visible = bullet_feats[..., 0] > 0.0
            bullet_active = bullet_feats[..., 1] > 0.0
            masks.append(~(bullet_visible & bullet_active))
        if map_feats is not None and self.map_tokens_in_attention:
            masks.append(torch.zeros(batch_size, self.n_map_tokens, dtype=torch.bool, device=ally_feats.device))
        return torch.cat(masks, dim=1)

    def _encode_map_tokens(self, map_feats):
        # map_feats: [bs, grid, grid, channels]
        map_image = map_feats.permute(0, 3, 1, 2).contiguous()
        encoded = self.map_encoder(map_image)
        return encoded.flatten(2).transpose(1, 2).contiguous()

    def _gated_map_context(self, self_input, map_tokens):
        map_ctx = map_tokens.mean(dim=1)
        map_ctx = self.map_context_proj(map_ctx)
        gate_input = torch.cat([self_input, map_ctx], dim=-1)
        map_gate = torch.sigmoid(self.map_gate(gate_input))
        return map_gate * map_ctx

    def forward(self, inputs, hidden_state):
        self_input, ally_feats, enemy_feats, bullet_feats, map_feats = self._split_inputs(inputs)

        self_token = self.self_encoder(self_input).unsqueeze(1)
        tokens = [self_token]
        if self.n_allies > 0:
            tokens.append(self.ally_encoder(ally_feats))
        if self.n_enemies > 0:
            tokens.append(self.enemy_encoder(enemy_feats))
        if self.max_obs_bullets > 0:
            tokens.append(self.bullet_encoder(bullet_feats))
        map_tokens = None
        if map_feats is not None and self.map_fusion != "none":
            map_tokens = self._encode_map_tokens(map_feats) + self.map_pos_embed
            if self.map_tokens_in_attention:
                tokens.append(map_tokens)

        entity_tokens = self.entity_norm(torch.cat(tokens, dim=1))
        padding_mask = self._entity_padding_mask(ally_feats, enemy_feats, bullet_feats, map_feats)
        attn_out, attn_weights = self.spatio_attn(
            query=entity_tokens,
            key=entity_tokens,
            value=entity_tokens,
            key_padding_mask=padding_mask,
            need_weights=self.log_attention,
            average_attn_weights=False,
        )
        for attn, attn_norm, ffn, ffn_norm in zip(
            self.extra_attn_layers,
            self.extra_attn_norms,
            self.extra_ffns,
            self.extra_ffn_norms,
        ):
            residual = attn_out
            layer_out, _ = attn(
                query=attn_out,
                key=attn_out,
                value=attn_out,
                key_padding_mask=padding_mask,
                need_weights=False,
            )
            attn_out = attn_norm(residual + layer_out)
            attn_out = ffn_norm(attn_out + ffn(attn_out))
        attn_self_ctx = attn_out[:, 0]

        if self.log_attention:
            self.latest_attn_weights = attn_weights.detach()
            self.latest_attn_scores = attn_weights.detach().mean(dim=1)
        else:
            self.latest_attn_weights = None
            self.latest_attn_scores = None

        if map_tokens is not None and self.use_gated_map_context:
            attn_self_ctx = attn_self_ctx + self._gated_map_context(self_input, map_tokens)

        rnn_features = torch.cat([self_input, attn_self_ctx], dim=-1)
        rnn_input = F.relu(self.rnn_in(rnn_features), inplace=True)
        next_hidden = self.rnn(rnn_input, hidden_state)
        fused = self.fusion(torch.cat([next_hidden, attn_self_ctx], dim=-1))
        q = self.q_out(fused)
        return q, next_hidden
