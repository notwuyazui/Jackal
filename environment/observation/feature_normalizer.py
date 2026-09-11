"""Shared normalization functions used by observation encoders."""

import math
from typing import Any

import numpy as np


def normalize_speed(unit: Any) -> float:
    return float(unit.speed) / max(1.0, float(unit.max_speed))


def normalize_angular_speed(unit: Any) -> float:
    return float(unit.angular_speed) / max(1.0, float(unit.max_angular_speed))


def normalize_acceleration(unit: Any) -> float:
    denominator = max(
        1.0,
        abs(float(unit.max_acceleration)),
        abs(float(unit.min_acceleration)),
    )
    return float(unit.acceleration) / denominator


def bullet_time_to_impact(agent: Any, bullet: Any) -> float:
    """Return normalized time to the closest threatening bullet approach."""
    hostile = hasattr(bullet, "shooter_team") and bullet.shooter_team != agent.team
    if not hostile:
        return 1.0

    rel_x = agent.position[0] - bullet.position[0]
    rel_y = agent.position[1] - bullet.position[1]
    vel_x, vel_y = getattr(bullet, "velocity", (0.0, 0.0))
    speed_sq = vel_x * vel_x + vel_y * vel_y
    if speed_sq <= 1e-6:
        return 1.0

    closing = rel_x * vel_x + rel_y * vel_y
    if closing <= 0.0:
        return 1.0

    time_to_closest = closing / speed_sq
    closest_x = bullet.position[0] + vel_x * time_to_closest
    closest_y = bullet.position[1] + vel_y * time_to_closest
    miss_distance = math.hypot(
        agent.position[0] - closest_x,
        agent.position[1] - closest_y,
    )
    unit_radius = math.hypot(agent.size[0], agent.size[1]) / 2.0
    if miss_distance > unit_radius * 1.5:
        return 1.0
    return float(np.clip(time_to_closest / 2.0, 0.0, 1.0))
