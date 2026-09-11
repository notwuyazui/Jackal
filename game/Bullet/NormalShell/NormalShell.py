'''
    普通炮弹，拥有标准的伤害，对重甲单位伤害较低，速度一般，射程较短，冷却时间较短，不会爆炸
'''
from game.Bullet.BaseBullet import BaseBullet
from game.Bullet.weapon_specs import ProjectileSpec, get_projectile_spec
from typing import Tuple
from game.Parameter import Team

class NormalShell(BaseBullet):
    def __init__(self, 
                 projectile_id: str, 
                 shooter, 
                 shooter_team: Team,
                 position: Tuple[float, float] = (0.0, 0.0), 
                 velocity_direction: Tuple[float, float] = (1.0, 0.0),
                 spec: ProjectileSpec | None = None):
        spec = spec or get_projectile_spec("normal_shell")
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
        )
