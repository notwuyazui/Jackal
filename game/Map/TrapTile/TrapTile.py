"""陷阱地块：不阻挡单位或子弹，并在单位停留时造成伤害。"""

from game.Map.BaseTile import BaseTile
from game.Parameter import UNIT_HEALTH


class TrapTile(BaseTile):
    def __init__(self, x: float = 0.0, y: float = 0.0, tile_size: int = 64, id=None):
        super().__init__(
            id=id,
            x=x,
            y=y,
            tile_size=tile_size,
            name="trap",
            letter="t",
            image_path="Map/TrapTile/trap.png",
            blocks_bullet=False,
            blocks_unit=False,
            destructible=False,
            damage_per_step=0.001 * UNIT_HEALTH,
            slow_multiplier=1.0,
        )
