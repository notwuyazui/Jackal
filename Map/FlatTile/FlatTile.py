"""
    平地地块：不阻挡任何东西，最默认的地块
"""

from Map.BaseTile import BaseTile

class FlatTile(BaseTile):
    def __init__(self, x: float = 0.0, y: float = 0.0, tile_size: int = 64, id = None):
        self.id = id
        self.x = x
        self.y = y
        self.tile_size = tile_size

        self.name = "flat"
        self.letter = 'o'
        self.image_path = "Map/FlatTile/flat.png"
        self.blocks_bullet = False
        self.blocks_unit = False
        self.destructible = False                    # 是否可破坏

        super().__init__(id, self.x, self.y, self.tile_size,
                         self.name, self.letter, self.image_path,
                         blocks_bullet=self.blocks_bullet, 
                         blocks_unit=self.blocks_unit, 
                         destructible=self.destructible)