'''
    普通炮弹
'''
from Bullet.BaseBullet import BaseBullet
from typing import Tuple
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
        size = (8, 8)
        lifetime = 2.0 
        speed_rate = 1.0 
        damage_rate = 1.0 
        penetration = [1.2, 1.0, 0.8] 
        
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