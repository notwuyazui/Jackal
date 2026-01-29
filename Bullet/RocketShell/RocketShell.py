'''
    火箭弹/爆炸弹
'''

import pygame
from Bullet.BaseBullet import BaseBullet
from typing import List, Tuple, Optional, Dict, Any
from Parameter import *

class RocketShell(BaseBullet):
    def __init__(self, 
                 projectile_id: str, 
                 shooter_id: int, 
                 shooter_team: Team,
                 position: Tuple[float, float] = (0.0, 0.0), 
                 velocity_direction: Tuple[float, float] = (1.0, 0.0)):
        
        # 火箭弹属性
        bullet_image_path = None  # 可以使用默认图像或指定路径
        size = (12, 12)  # 稍大尺寸
        lifetime = 1.5  # 1.5秒生命期
        speed_rate = 0.8  # 速度稍慢
        damage_rate = 1.5  # 基础伤害更高
        penetration = [0.8, 1.0, 1.2]  # 对重甲效果更好
        
        # 爆炸属性
        is_explosive = True
        explosion_radius = 60.0  # 爆炸半径
        explosion_damage_rate = 0.8  # 爆炸伤害系数
        explosion_image_path = None  # 可以指定爆炸效果图像路径
        
        super().__init__(
            projectile_id=projectile_id,
            shooter_id=shooter_id,
            shooter_team=shooter_team,
            position=position,
            velocity_direction=velocity_direction,
            bullet_image_path=bullet_image_path,
            size=size,
            lifetime=lifetime,
            speed_rate=speed_rate,
            damage_rate=damage_rate,
            penetration=penetration,
            is_explosive=is_explosive,
            explosion_radius=explosion_radius,
            explosion_damage_rate=explosion_damage_rate,
            explosion_image_path=explosion_image_path
        )