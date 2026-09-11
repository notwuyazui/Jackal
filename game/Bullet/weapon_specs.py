"""Canonical projectile and weapon parameters used by the game engine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from game.Parameter import BULLET_SPEED


@dataclass(frozen=True)
class ProjectileSpec:
    """Immutable parameters for one ammunition type.

    Derived values such as speed and range are calculated here so gameplay,
    observations, and action masks cannot drift apart.
    """

    name: str
    image_path: str | None
    size: tuple[float, float]
    lifetime: float
    speed_rate: float
    damage_rate: float
    cooldown: float
    penetration: tuple[float, float, float]
    is_explosive: bool = False
    explosion_radius: float = 0.0
    explosion_damage_rate: float = 1.0
    explosion_image_path: str | None = None

    def __post_init__(self) -> None:
        if len(self.size) != 2 or any(value <= 0 for value in self.size):
            raise ValueError(f"Invalid projectile size for {self.name}: {self.size}")
        if len(self.penetration) != 3 or any(value < 0 for value in self.penetration):
            raise ValueError(
                f"Invalid projectile penetration for {self.name}: {self.penetration}"
            )
        if self.lifetime <= 0 or self.speed_rate <= 0 or self.cooldown < 0:
            raise ValueError(
                f"Invalid projectile timing for {self.name}: "
                f"lifetime={self.lifetime}, speed_rate={self.speed_rate}, "
                f"cooldown={self.cooldown}"
            )
        if self.damage_rate < 0 or self.explosion_damage_rate < 0:
            raise ValueError(f"Projectile damage rates must be >= 0 for {self.name}")
        if self.explosion_radius < 0:
            raise ValueError(f"Projectile explosion radius must be >= 0 for {self.name}")

    @property
    def speed(self) -> float:
        return float(BULLET_SPEED) * self.speed_rate

    @property
    def max_range(self) -> float:
        return self.speed * self.lifetime

    def with_overrides(self, overrides: Mapping[str, Any] | None) -> "ProjectileSpec":
        if not overrides:
            return self

        values = dict(overrides)
        absolute_speed = values.pop("speed", None)
        if absolute_speed is not None:
            if "speed_rate" in values:
                raise ValueError("Projectile overrides cannot set both 'speed' and 'speed_rate'")
            values["speed_rate"] = float(absolute_speed) / float(BULLET_SPEED)

        allowed = {
            "image_path",
            "size",
            "lifetime",
            "speed_rate",
            "damage_rate",
            "cooldown",
            "penetration",
            "is_explosive",
            "explosion_radius",
            "explosion_damage_rate",
            "explosion_image_path",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown projectile override fields for {self.name}: {unknown}")

        if "size" in values:
            values["size"] = tuple(float(value) for value in values["size"])
        if "penetration" in values:
            values["penetration"] = tuple(float(value) for value in values["penetration"])
        for field_name in (
            "lifetime",
            "speed_rate",
            "damage_rate",
            "cooldown",
            "explosion_radius",
            "explosion_damage_rate",
        ):
            if field_name in values:
                values[field_name] = float(values[field_name])

        return replace(self, **values)


PROJECTILE_SPECS: dict[str, ProjectileSpec] = {
    "normal_shell": ProjectileSpec(
        name="normal_shell",
        image_path="Bullet/NormalShell/normalshell.png",
        size=(6.0, 6.0),
        lifetime=1.2,
        speed_rate=1.0,
        damage_rate=2.5,
        cooldown=1.5,
        penetration=(1.0, 0.8, 0.6),
    ),
    "rocket_shell": ProjectileSpec(
        name="rocket_shell",
        image_path="Bullet/RocketShell/RocketShell.png",
        size=(15.0, 15.0),
        lifetime=1.5,
        speed_rate=1.2,
        damage_rate=0.75,
        cooldown=0.8,
        penetration=(0.8, 1.0, 1.2),
        is_explosive=True,
        explosion_radius=50.0,
        explosion_damage_rate=0.75,
        explosion_image_path="Bullet/RocketShell/RocketShellExplosion.png",
    ),
    "heavy_shell": ProjectileSpec(
        name="heavy_shell",
        image_path="Bullet/HeavyShell/heavyshell.png",
        size=(10.0, 10.0),
        lifetime=8.0,
        speed_rate=0.5,
        damage_rate=2.0,
        cooldown=1.2,
        penetration=(0.8, 1.0, 1.2),
        is_explosive=True,
        explosion_radius=100.0,
        explosion_damage_rate=2.0,
        explosion_image_path="Bullet/HeavyShell/HeavyShellExplosion.png",
    ),
}


def get_projectile_spec(
    ammunition: str,
    overrides: Mapping[str, Any] | None = None,
) -> ProjectileSpec:
    """Return the validated effective spec for an ammunition name."""

    name = str(ammunition).lower()
    try:
        base_spec = PROJECTILE_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ammunition type: {ammunition!r}") from exc
    return base_spec.with_overrides(overrides)
