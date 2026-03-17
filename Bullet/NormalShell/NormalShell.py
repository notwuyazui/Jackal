'''
    普通炮弹，拥有标准的伤害，对重甲单位伤害较低，速度一般，射程较短，冷却时间较短，不会爆炸
'''
from Bullet.BaseBullet import BaseBullet
from typing import Tuple
from Parameter import *

class NormalShell(BaseBullet):
    def __init__(self, 
                 projectile_id: str, 
                 shooter, 
                 shooter_team: Team,
                 position: Tuple[float, float] = (0.0, 0.0), 
                 velocity_direction: Tuple[float, float] = (1.0, 0.0)):
        
        # 普通炮弹属性
        self.bullet_image_path = "Bullet/NormalShell/normalshell.png"
        self.size = (6, 6)
        self.lifetime = 1.2 
        self.speed_rate = 1.0 
        self.damage_rate = 1.0 
        self.cooldown = 0.4
        self.penetration = [1.0, 0.8, 0.6] 
        
        self.is_explosive = False  # 普通炮弹不会爆炸
        
        super().__init__(
            projectile_id=projectile_id,
            shooter=shooter,
            shooter_team=shooter_team,
            position=position,
            velocity_direction=velocity_direction,
            bullet_image_path=self.bullet_image_path,
            size=self.size,
            lifetime=self.lifetime,
            speed_rate=self.speed_rate,
            damage_rate=self.damage_rate,
            cooldown=self.cooldown,
            penetration=self.penetration,
            is_explosive=self.is_explosive  # 普通炮弹不会爆炸
        )