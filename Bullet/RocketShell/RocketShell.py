'''
    火箭弹
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
        
        bullet_image_path = "Bullet/RocketShell/RocketShell.png"
        size = (32, 32)
        lifetime = 2.0 
        speed_rate = 1.0 
        damage_rate = 1.0           # 被命中本体受伤=直接伤害+爆炸伤害
        penetration = [0.8, 1.0, 1.2]
        
        # 爆炸属性
        is_explosive = True
        explosion_radius = 50.0 
        explosion_damage_rate = 1.0     
        explosion_image_path = "Bullet/RocketShell/RocketShellExplosion.png"
        
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