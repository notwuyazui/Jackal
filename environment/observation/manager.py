"""Facade that owns all observation encoders."""

from typing import TYPE_CHECKING, List

import numpy as np

from environment.observation.global_state import GlobalStateEncoder
from environment.observation.local_obs import LocalObservationEncoder
from environment.observation.map_encoder import MapFeatureEncoder

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv


class ObservationManager:
    """Stable entry point for local observations and centralized state."""

    def __init__(self, env: "JackalEnv") -> None:
        self.map_encoder = MapFeatureEncoder(env)
        self.local = LocalObservationEncoder(env, self.map_encoder)
        self.global_state = GlobalStateEncoder(env, self.map_encoder)

    def get_observations(self) -> List[np.ndarray]:
        return self.local.encode()

    def get_state(self) -> np.ndarray:
        return self.global_state.encode()

    def observation_dim(self) -> int:
        return self.local.dimension()

    def state_dim(self) -> int:
        return self.global_state.dimension()

    def observation_map_dim(self) -> int:
        return self.map_encoder.observation_dim()

    def state_map_dim(self) -> int:
        return self.map_encoder.state_dim()
