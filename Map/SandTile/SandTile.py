"""
    沙地地块：不阻挡任何东西，会降低单位25%的移动速度。
"""

from Map.BaseTile import BaseTile

class SandTile(BaseTile):
    def __init__(self, x: float = 0.0, y: float = 0.0, tile_size: int = 64, id = None):
        self.id = id
        self.x = x
        self.y = y
        self.tile_size = tile_size

        self.name = "sand"
        self.letter = 's'
        self.image_path = "Map/SandTile/sand.png"
        self.blocks_bullet = False
        self.blocks_unit = False
        self.destructible = False                    # 是否可破坏
        self.damage_per_step = 0.1                       # 单位踏上时造成的伤害
        self.slow_multiplier = 0.75                   # 移动速度减缓倍数

        super().__init__(id, self.x, self.y, self.tile_size,
                         self.name, self.letter, self.image_path,
                         blocks_bullet=self.blocks_bullet, 
                         blocks_unit=self.blocks_unit, 
                         destructible=self.destructible,
                         damage_per_step=self.damage_per_step,
                         slow_multiplier=self.slow_multiplier)