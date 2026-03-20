"""
    水域：阻挡单位，不阻挡子弹
"""

from Map.BaseTile import BaseTile

class WaterTile(BaseTile):
    def __init__(self, x: float = 0.0, y: float = 0.0, tile_size: int = 64, id = None):
        self.id = id
        self.x = x
        self.y = y
        self.tile_size = tile_size

        self.name = "water"
        self.letter = 'w'
        self.image_path = "Map/WaterTile/water.png"
        self.blocks_bullet = False
        self.blocks_unit = True
        self.destructible = False

        super().__init__(id, self.x, self.y, self.tile_size,
                    self.name, self.letter, self.image_path,
                    blocks_bullet=self.blocks_bullet, 
                    blocks_unit=self.blocks_unit, 
                    destructible=self.destructible)