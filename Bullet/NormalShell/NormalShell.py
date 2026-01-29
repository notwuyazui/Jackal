'''
    普通炮弹
'''

import pygame
from Bullet.BaseBullet import BaseBullet
from typing import List, Tuple, Optional, Dict, Any
from Parameter import *

class NormalShell(BaseBullet):
    def __init__(self, 
                 projectile_id: str, 
                 shooter_id: int, 
                 shooter_team: Team,
                 position: Tuple[float, float] = (0.0, 0.0), 
                 velocity_direction: Tuple[float, float] = (1.0, 0.0)):
        
        # 普通炮弹属性
        bullet_image_path = "Bullet/NormalShell/normalshell.png"
        size = (8, 8)  # 小尺寸
        lifetime = 2.0  # 2秒生命期
        speed_rate = 1.0  # 基础速度
        damage_rate = 1.0  # 基础伤害
        penetration = [1.2, 1.0, 0.8]  # 对轻甲120%，中甲100%，重甲80%
        
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
            is_explosive=False  # 普通炮弹不会爆炸
        )