"""陷阱地块：不阻挡单位或子弹，并在单位停留时造成伤害。"""

from engine.maps.base_tile import BaseTile
from engine.parameters import UNIT_HEALTH


class TrapTile(BaseTile):
    def __init__(self, x: float = 0.0, y: float = 0.0, tile_size: int = 64, id=None):
        super().__init__(
            id=id,
            x=x,
            y=y,
            tile_size=tile_size,
            name="trap",
            letter="t",

            blocks_bullet=False,
            blocks_unit=False,
            destructible=False,
            damage_per_step=0.001 * UNIT_HEALTH,
            slow_multiplier=1.0,
        )
