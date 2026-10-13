"""Jackal owns all physics. CoT consumes this read-only presentation protocol."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
from engine.parameters import Team, BULLET_DAMAGE
from engine.contracts.outcome import outcome_from_counts
from environment.scenario import build_ai_world as build_world

STATE_FILE = ROOT / "engine/scenarios/game_states/border_map_3t_vs_3t.json"
TRAINING_FILE = ROOT / "training/configs/marl2/jackal_autoaim_3v3_qmix.json"
METERS_PER_PIXEL = 0.25


def result_for(world, max_steps):
    counts = {t.name: sum(u.is_alive for u in world.iter_units() if u.team == t)
              for t in (Team.PLAYER, Team.ENEMY)}
    return outcome_from_counts(counts, world.tick, max_steps)


def unit_parameters(unit):
    weapon = unit.get_weapon_spec("normal_shell")
    return {
        "id": unit.id, "model": "m1a1", "unit_type": unit.unit_type,
        "size": list(unit.size), "collision_size": list(unit.collision_size),
        "max_speed": unit.max_speed, "max_acceleration": unit.max_acceleration,
        "min_acceleration": unit.min_acceleration, "hull_deg_s": unit.max_angular_speed,
        "turret_deg_s": unit.turret_angular_speed, "max_health": unit.max_health,
        "sight_range": unit.sight_range, "communication_range": unit.communication_range,
        # ProjectileSpec stores a multiplier, not a duration. The unit owns
        # the effective seconds, including an absolute unit-level override.
        "cooldown": unit.fire_cooldown_duration(weapon), "shell_speed": weapon.speed,
        "shell_lifetime": weapon.lifetime, "shell_range": weapon.max_range,
        "shell_size": list(weapon.size), "shell_base_damage": BULLET_DAMAGE * weapon.damage_rate,
        "armor_damage_multipliers": list(weapon.penetration),
        "ricochet": False, "splash_damage": False,
        "shell_visual": "contact_explosion", "ammo": "normal_shell",
        "ai": "EnemyAI", "ai_level": unit.ai_intelligence_level,
    }


def manifest(world, state_file=STATE_FILE, dt=0.03, seed=42, max_steps=6000):
    return {
        "protocol": "jackal-render/1", "state_file": str(Path(state_file).relative_to(ROOT.parent)),
        "state_sha256": hashlib.sha256(Path(state_file).read_bytes()).hexdigest(),
        "mode": "lockstep", "dt": dt, "seed": seed, "max_steps": max_steps, "follow_id": 2,
        "meters_per_pixel": METERS_PER_PIXEL,
        "map": json.loads(json.dumps(__import__("dataclasses").asdict(world.render_manifest(dt).map))),
        "rules": {"unit_collision": world.unit_manager.enable_unit_collision,
                  "tear_drop_vision": world.unit_manager.use_tear_drop_vision,
                  "auto_communicate": world.unit_manager.auto_communicate_enabled},
        "simulation": "Jackal BattleWorld; CoT presentation only",
        "presentation": {"2d": "pygame-offscreen-inset", "3d": "cot-browser",
                         "model": "m1a1", "camera": "turret-follow", "ground": "cot-grass"},
        "demo_overrides": {"all_units_enemy_ai": True, "ammunition": ["normal_shell"],
                           "contact_explosion": "visual_only; original NormalShell damage"},
        "units": [unit_parameters(u) for u in world.iter_units()],
    }


def export_frame(world, max_steps=6000):
    frame = world.render_snapshot().to_dict()
    frame["result"] = result_for(world, max_steps)
    frame["hash"] = hashlib.sha256(json.dumps(frame, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()[:16]
    return frame


def step_world(world, dt):
    world.step(dt)


def world_position(x, y, width=960, height=640):
    return ((x - width / 2) * METERS_PER_PIXEL, (y - height / 2) * METERS_PER_PIXEL)


def world_yaw(degrees):
    # Jackal 0° = screen up; CoT forward = local +Z. Z increases down the map.
    return math.pi - math.radians(degrees)
