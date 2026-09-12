"""Local and global terrain feature encoders."""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from game.BattleState import MapSnapshot, UnitSnapshot

if TYPE_CHECKING:
    from environment.observation.manager import ObservationConfig


class MapFeatureEncoder:
    """Encode immutable terrain snapshots for local and global policies."""

    def __init__(self, config: ObservationConfig) -> None:
        self.config = config

    def observation_dim(self) -> int:
        if not self.config.include_obs_map_features:
            return 0
        grid_size = self.config.obs_map_grid_size
        return grid_size * grid_size * self.config.map_feature_dim

    def state_dim(self) -> int:
        if not self.config.include_state_map_features:
            return 0
        grid_size = self.config.state_map_grid_size
        return grid_size * grid_size * self.config.map_feature_dim

    def local_features(
        self,
        game_map: MapSnapshot,
        agent: UnitSnapshot,
    ) -> List[float]:
        if not self.config.include_obs_map_features:
            return []

        grid_size = self.config.obs_map_grid_size
        half_grid = grid_size // 2
        cell_size = self.config.obs_map_cell_size
        features: List[float] = []
        base_x, base_y = agent.position
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                sample_x = base_x + (grid_x - half_grid) * cell_size
                sample_y = base_y + (grid_y - half_grid) * cell_size
                features.extend(game_map.terrain_at(sample_x, sample_y))
        return features

    def global_features(self, game_map: MapSnapshot) -> List[float]:
        if not self.config.include_state_map_features:
            return []

        grid_size = self.config.state_map_grid_size
        features: List[float] = []
        width = max(1.0, float(self.config.screen_width))
        height = max(1.0, float(self.config.screen_height))
        for grid_y in range(grid_size):
            sample_y = (grid_y + 0.5) * height / grid_size
            for grid_x in range(grid_size):
                sample_x = (grid_x + 0.5) * width / grid_size
                features.extend(game_map.terrain_at(sample_x, sample_y))
        return features
