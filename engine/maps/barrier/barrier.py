"""
    障碍地块：阻挡单位和子弹
"""

from engine.maps.base_tile import BaseTile

class BarrierTile(BaseTile):
    """障碍：阻挡单位和子弹"""
    def __init__(self, x: float = 0.0, y: float = 0.0, tile_size: int = 64, id = None):
        self.id = id
        self.x = x
        self.y = y
        self.tile_size = tile_size

        self.name = "barrier"
        self.letter = 'x'

        self.blocks_bullet = True
        self.blocks_unit = True
        self.destructible = False

        super().__init__(id, self.x, self.y, self.tile_size,
                    self.name, self.letter,
                    blocks_bullet=self.blocks_bullet, 
                    blocks_unit=self.blocks_unit, 
                    destructible=self.destructible)