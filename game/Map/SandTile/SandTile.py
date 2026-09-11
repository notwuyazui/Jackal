"""沙地地块：不阻挡单位或子弹，并降低单位的移动速度。"""

from game.Map.BaseTile import BaseTile


class SandTile(BaseTile):
    def __init__(self, x: float = 0.0, y: float = 0.0, tile_size: int = 64, id=None):
        super().__init__(
            id=id,
            x=x,
            y=y,
            tile_size=tile_size,
            name="sand",
            letter="s",
            image_path="Map/SandTile/sand.png",
            blocks_bullet=False,
            blocks_unit=False,
            destructible=False,
            damage_per_step=0.1,
            slow_multiplier=0.75,
        )
