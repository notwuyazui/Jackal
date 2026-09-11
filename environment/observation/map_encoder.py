"""Local and global terrain feature encoders."""

from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv


class MapFeatureEncoder:
    """Encode map tiles without coupling terrain logic to the Gym interface."""

    def __init__(self, env: "JackalEnv") -> None:
        self.env = env

    def observation_dim(self) -> int:
        if not self.env.include_obs_map_features:
            return 0
        grid_size = self.env.obs_map_grid_size
        return grid_size * grid_size * self.env.map_feature_dim

    def state_dim(self) -> int:
        if not self.env.include_state_map_features:
            return 0
        grid_size = self.env.state_map_grid_size
        return grid_size * grid_size * self.env.map_feature_dim

    def terrain_at(self, x: float, y: float) -> List[float]:
        game_map = getattr(self.env, "game_map", None)
        if game_map is None:
            return [0.0] * self.env.map_feature_dim

        col = int(float(x) // game_map.tile_size)
        row = int(float(y) // game_map.tile_size)
        if row < 0 or row >= game_map.height or col < 0 or col >= game_map.width:
            return [0.0, 0.0, 0.0, 0.0]

        tile = game_map.tiles[row][col]
        blocks_unit = 1.0 if getattr(tile, "blocks_unit", False) else 0.0
        blocks_bullet = 1.0 if getattr(tile, "blocks_bullet", False) else 0.0
        water = 1.0 if getattr(tile, "letter", "") == "w" else 0.0
        return [1.0, blocks_unit, blocks_bullet, water]

    def local_features(self, agent: Any) -> List[float]:
        if not self.env.include_obs_map_features:
            return []

        grid_size = self.env.obs_map_grid_size
        half_grid = grid_size // 2
        cell_size = self.env.obs_map_cell_size
        features: List[float] = []
        base_x, base_y = agent.position
        for grid_y in range(grid_size):
            for grid_x in range(grid_size):
                sample_x = base_x + (grid_x - half_grid) * cell_size
                sample_y = base_y + (grid_y - half_grid) * cell_size
                features.extend(self.terrain_at(sample_x, sample_y))
        return features

    def global_features(self) -> List[float]:
        if not self.env.include_state_map_features:
            return []

        grid_size = self.env.state_map_grid_size
        features: List[float] = []
        width = max(1.0, float(self.env.screen_width))
        height = max(1.0, float(self.env.screen_height))
        for grid_y in range(grid_size):
            sample_y = (grid_y + 0.5) * height / grid_size
            for grid_x in range(grid_size):
                sample_x = (grid_x + 0.5) * width / grid_size
                features.extend(self.terrain_at(sample_x, sample_y))
        return features
