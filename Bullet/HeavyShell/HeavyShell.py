'''
    普通炮弹
'''
from Bullet.BaseBullet import BaseBullet
from typing import Tuple
from Parameter import *

class HeavyShell(BaseBullet):
    def __init__(self, 
                 projectile_id: str, 
                 shooter, 
                 shooter_team: Team,
                 position: Tuple[float, float] = (0.0, 0.0), 
                 velocity_direction: Tuple[float, float] = (1.0, 0.0)):
        
        # 普通炮弹属性
        self.bullet_image_path = "Bullet/HeavyShell/heavyshell.png"
        self.size = (10, 10)
        self.lifetime = 8.0 
        self.speed_rate = 0.5 
        self.damage_rate = 2.0
        self.cooldown = 1.2
        self.penetration = [0.8, 1.0, 1.2] 
        
        # 爆炸属性
        self.is_explosive = True
        self.explosion_radius = 100.0 
        self.explosion_damage_rate = 2.0  
        self.explosion_image_path = "Bullet/HeavyShell/HeavyShellExplosion.png"
        
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
            is_explosive=self.is_explosive,
            explosion_radius=self.explosion_radius,
            explosion_damage_rate=self.explosion_damage_rate,
            explosion_image_path=self.explosion_image_path
        )