'''
    火箭弹，拥有1.5倍的伤害，对重甲单位伤害较高，速度较快，射程一般，冷却时间一般，小范围爆炸，爆炸伤害为一半
'''

from game.Bullet.BaseBullet import BaseBullet
from game.Bullet.weapon_specs import ProjectileSpec, get_projectile_spec
from typing import Tuple
from game.Parameter import Team

class RocketShell(BaseBullet):
    def __init__(self, 
                 projectile_id: str, 
                 shooter, 
                 shooter_team: Team,
                 position: Tuple[float, float] = (0.0, 0.0), 
                 velocity_direction: Tuple[float, float] = (1.0, 0.0),
                 spec: ProjectileSpec | None = None):
        spec = spec or get_projectile_spec("rocket_shell")
        super().__init__(
            projectile_id=projectile_id,
            shooter=shooter,
            shooter_team=shooter_team,
            position=position,
            velocity_direction=velocity_direction,
            bullet_image_path=spec.image_path,
            size=spec.size,
            lifetime=spec.lifetime,
            speed_rate=spec.speed_rate,
            damage_rate=spec.damage_rate,
            cooldown=spec.cooldown,
            penetration=list(spec.penetration),
            is_explosive=spec.is_explosive,
            explosion_radius=spec.explosion_radius,
            explosion_damage_rate=spec.explosion_damage_rate,
            explosion_image_path=spec.explosion_image_path,
        )
