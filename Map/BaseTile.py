'''
    地块的基础类
'''

import pygame
import os
import random
from typing import List, Tuple, Optional
from Parameter import *
from utils import load_image, get_next_filename

class BaseTile:
    def __init__(self, id=None, x=0.0, y=0.0, tile_size=64, name="base", letter='?', image_path=None,
                 blocks_bullet=False, 
                 blocks_unit=False, 
                 destructible=False, 
                 max_health=0, 
                 destroyed_tile_type=None,
                 explodes_on_destroy=False,
                 explosion_range=0,
                 explosion_damage=0,
                 crushable=False,
                 provides_invisibility=False,
                 damage_per_step=0.0,
                 slow_multiplier=1.0):
        self.x = x                      # 世界坐标左上角 x
        self.y = y
        self.tile_size = tile_size
        self.rect = pygame.Rect(x, y, tile_size, tile_size)

        self.name = name
        self.letter = letter
        self.image_path = image_path
        self.image = load_image(image_path) if image_path else None

        # 独特属性
        self.blocks_bullet = blocks_bullet                                                  # 是否阻挡子弹
        self.blocks_unit = blocks_unit                                                      # 是否阻挡单位
        self.destructible = destructible                                                    # 是否可破坏
        self.max_health = max_health if destructible else 0                                 # 初始生命值（仅当可破坏时有效）
        self.destroyed_tile_type = destroyed_tile_type if destructible else None            # 被破坏后变成的地块类型（如 'flat'）
        self.explodes_on_destroy = explodes_on_destroy if destructible else False           # 被破坏时是否爆炸
        self.explosion_range = explosion_range if explodes_on_destroy else None
        self.explosion_damage = explosion_damage if explodes_on_destroy else None
        self.crushable = crushable if not blocks_unit else False                            # 是否可被碾压
        self.provides_invisibility = provides_invisibility if not blocks_unit else False    # 是否提供隐身
        self.damage_per_step = damage_per_step if not blocks_unit else 0                # 单位踏上时造成的伤害
        self.slow_multiplier = slow_multiplier if not blocks_unit else 1.0           # 单位踏上时减速倍数

        # 实时属性
        self.id = id
        self.current_health = self.max_health               # 当前生命值

    def draw(self, surface: pygame.Surface, camera_offset: Tuple[float, float] = (0, 0)) -> None:
        """绘制地块到表面"""
        if self.image:
            screen_x = self.x - camera_offset[0]
            screen_y = self.y - camera_offset[1]
            surface.blit(self.image, (screen_x, screen_y))

    def update(self, delta_time: float) -> None:
        """更新地块状态（可被子类重写）"""
        pass

    def is_walkable(self) -> bool:
        return not self.blocks_unit

    def is_bullet_blocked(self) -> bool:
        return self.blocks_bullet

    def is_destructible(self) -> bool:
        return self.destructible

    def is_bombable(self) -> bool:
        return self.explodes_on_destroy

    def is_crushable(self) -> bool:
        return self.crushable

    def is_invisible(self) -> bool:
        return self.provides_invisibility

    def is_harmful(self) -> bool:
        return self.damage_per_step > 0

    def get_current_health(self) -> Optional[float]:
        return self.current_health if self.destructible else None

    def take_damage(self, amount: float) -> Tuple[bool, float]:
        """承受伤害"""
        if not self.destructible:
            return False, 0
        self.current_health -= amount
        if self.current_health <= 0:
            self.current_health = 0
            return True, 0
        return False, self.current_health

    def on_crush(self):
        """当被碾压时调用，返回应变成的地块类型"""
        if self.crushable:
            return self.destroyed_tile_type
        return None

    def apply_buff(self, unit):
        """单位踏上时的效果"""
        if self.damage_per_step > 0:
            unit.take_damage_from_tile(self.damage_per_step)
        if self.slow_multiplier < 1.0:
            unit.speed_slow_multiplier *= self.slow_multiplier
        if self.provides_invisibility:
            unit.conceal = True
        return unit
