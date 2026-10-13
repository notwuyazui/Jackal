"""Canonical unit-state serialization shared by world and standalone unit saves."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from engine.units.base_unit import BaseUnit


def unit_game_state(unit: BaseUnit) -> dict[str, Any]:
    return {
        "unit_id": int(unit.id),
        "unit_type": str(unit.unit_type),
        "team": unit.team.name,
        "position": [float(unit.position[0]), float(unit.position[1])],
        "size": list(unit.size),
        "base_collision_size": list(unit.base_collision_size),
        "armor_type": unit.armor_type.name,
        "ammunition_types": list(unit.ammunition_types),
        "ammo_switch_time": float(unit.ammo_switch_time),
        "using_ai": bool(unit.usingAI),
        "ai_intelligence_level": int(unit.ai_intelligence_level),
        "visible": bool(unit.visible),
        "sight_range": float(unit.sight_range),
        "min_sight_range": float(unit.min_sight_range),
        "communication_range": float(unit.communication_range),
        "collision_scale": float(unit.collision_scale),
        "max_speed": float(unit.max_speed),
        "max_acceleration": float(unit.max_acceleration),
        "min_acceleration": float(unit.min_acceleration),
        "max_angular_speed": float(unit.max_angular_speed),
        "turret_angular_speed": float(unit.turret_angular_speed),
        "max_health": float(unit.max_health),
        "health": float(unit.health),
        "is_alive": bool(unit.is_alive),
        "direction_angle": float(unit.direction_angle),
        "turret_direction_angle": float(unit.turret_direction_angle),
        "turret_target_angle": float(unit.turret_target_angle),
        "speed": float(unit.speed),
        "acceleration": float(unit.acceleration),
        "angular_speed": float(unit.angular_speed),
        "current_ammunition": str(unit.current_ammunition),
        "fire_cooldown": float(unit.fire_cooldown),
        "fire_cooldown_override": unit.fire_cooldown_override,
        "projectile_overrides": dict(unit.projectile_overrides),
        "ai_fire_angle_tolerance": float(unit.ai_fire_angle_tolerance),
        "is_switching_ammo": bool(unit.is_switching_ammo),
        "reload_timer": float(unit.reload_timer),
        "target_ammunition": str(unit.target_ammunition),
        "destroy_enemy_count": int(unit.destroy_enemy_count),
        "damage_dealt": float(unit.damage_dealt),
        "assist_destroy_count": int(unit.assist_destroy_count),
        "assist_damage_dealt": float(unit.assist_damage_dealt),
        "damage_received": float(unit.damage_received),
        "potential_damage": float(unit.potential_damage),
        "killed_by": unit.killed_by,
        "living_time": float(unit.living_time),
        "reward": float(unit.reward),
        "unit_collision_count": int(unit.unit_collision_count),
    }
