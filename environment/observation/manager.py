"""Facade and immutable configuration for observation encoders."""

from dataclasses import dataclass
from typing import List

import numpy as np

from environment.observation.global_state import GlobalStateEncoder
from environment.observation.local_obs import LocalObservationEncoder
from environment.observation.map_encoder import MapFeatureEncoder
from game.BattleState import WorldSnapshot


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    screen_width: float
    screen_height: float
    n_agents: int
    n_enemies: int
    unit_type_names: tuple[str, ...]
    include_unit_type_onehot: bool
    max_obs_bullets: int
    max_state_bullets: int
    sight_range: float
    bullet_norm_speed: float
    max_steps: int
    include_obs_map_features: bool
    include_state_map_features: bool
    obs_map_grid_size: int
    obs_map_cell_size: float
    state_map_grid_size: int
    map_feature_dim: int = 4

    @property
    def unit_type_dim(self) -> int:
        return len(self.unit_type_names) if self.include_unit_type_onehot else 0

    def unit_type_onehot(self, unit_type: str) -> list[float]:
        if not self.include_unit_type_onehot:
            return []
        onehot = [0.0] * self.unit_type_dim
        if unit_type in self.unit_type_names:
            onehot[self.unit_type_names.index(unit_type)] = 1.0
        return onehot


class ObservationManager:
    """Encode snapshots without retaining the mutable environment or world."""

    def __init__(self, config: ObservationConfig) -> None:
        self.config = config
        self.map_encoder = MapFeatureEncoder(config)
        self.local = LocalObservationEncoder(config, self.map_encoder)
        self.global_state = GlobalStateEncoder(config, self.map_encoder)

    def get_observations(self, snapshot: WorldSnapshot) -> List[np.ndarray]:
        return self.local.encode(snapshot)

    def get_state(self, snapshot: WorldSnapshot) -> np.ndarray:
        return self.global_state.encode(snapshot)

    def observation_dim(self) -> int:
        return self.local.dimension()

    def state_dim(self) -> int:
        return self.global_state.dimension()

    def observation_map_dim(self) -> int:
        return self.map_encoder.observation_dim()

    def state_map_dim(self) -> int:
        return self.map_encoder.state_dim()
