"""Perception-driven tactical AI shared by every AI-controlled unit."""

from __future__ import annotations

import heapq
import math
import random
from typing import Any, Iterable, Optional, Tuple, TypeGuard

import pygame

from game.Unit.BaseUnit import BaseUnit


Vector = Tuple[float, float]


class EnemyAI:
    """Control one unit using only direct or communicated target information.

    ``INTELLIGENCE_LEVEL`` is the single global skill control. Values from 1
    through 9 progressively improve communication frequency, target memory,
    route replanning, threat avoidance, and low-health kiting. Unit and weapon
    attributes remain authoritative; the AI does not maintain separate combat
    ranges or cooldowns.
    """

    INTELLIGENCE_LEVEL = 7

    SAFE_MARGIN = 2.0
    SEPARATION_DISTANCE = 72.0
    STUCK_DISTANCE_EPSILON = 0.08
    STUCK_FRAME_LIMIT = 28
    RECOVERY_FRAMES = 32
    BULLET_THREAT_DISTANCE = 150.0
    BULLET_LOOKAHEAD_TIME = 0.65
    PAIR_MIN_INTELLIGENCE = 3
    SEARCH_SHARE_MIN_INTELLIGENCE = 4
    REGROUP_MIN_INTELLIGENCE = 5

    def __init__(
        self,
        unit: BaseUnit,
        unit_manager,
        bullet_manager,
        game_map,
        intelligence_level: Optional[int] = None,
    ) -> None:
        self.unit = unit
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        self.game_map = game_map

        level = self.INTELLIGENCE_LEVEL if intelligence_level is None else int(intelligence_level)
        if not 1 <= level <= 9:
            raise ValueError("EnemyAI intelligence_level must be between 1 and 9")
        self.intelligence_level = level

        self.ai_state = "search"
        self.fire_angle_tolerance = float(
            getattr(self.unit, "ai_fire_angle_tolerance", 10.0)
        )
        self.strafe_sign = 1.0 if self.unit.id % 2 == 0 else -1.0
        self._avoidance_sign = self.strafe_sign

        self.target_unit: Optional[BaseUnit] = None
        self.last_known_target_position: Optional[Vector] = None
        self._target_memory_age = 0
        self._target_memory_limit = 45 + 45 * level

        self._frame_count = 0
        self._last_communication_frame = -10_000
        self._communication_interval = max(6, 50 - 5 * level)
        self._tactical_interval = max(2, 4 - level // 4)
        self._last_tactical_frame = -10_000
        self._cached_known_enemies: tuple[BaseUnit, ...] = ()
        self._cached_regroup_ally: Optional[BaseUnit] = None

        # One broad spatial query is shared by perception, separation, and all
        # collision probes performed during this AI update.
        self._spatial_cache_frame = -1
        self._cached_nearby_units: tuple[BaseUnit, ...] = ()

        # Steering reacts more often than tactical planning, but a safe result
        # can be reused briefly while the requested direction barely changes.
        self._steering_interval = 2
        self._last_steering_input: Optional[Vector] = None
        self._cached_steering_direction: Optional[Vector] = None
        self._cached_steering_avoiding = False

        self._search_index = 0
        self._search_destination: Optional[Vector] = None
        self._spawn_position: Vector = (
            float(self.unit.position[0]),
            float(self.unit.position[1]),
        )
        self._searched_cells: set[tuple[int, int]] = set()
        self._passable_search_cells: list[tuple[int, int]] = []
        self._search_map_signature: Optional[tuple[int, int, int, float]] = None
        self._last_search_mark_frame = -10_000
        self._last_search_share_frame = -10_000

        self._pair_partner_id: Optional[int] = None
        self._pair_formed_frame = 0
        self._pair_split_until_frame = 0
        self._next_pair_split_frame = 0

        self._regroup_ally_id: Optional[int] = None
        self._regroup_until_frame = 0
        self._next_regroup_frame = 0
        self._path: list[Vector] = []
        self._path_goal: Optional[Vector] = None
        self._last_path_frame = -10_000
        self._path_replan_interval = max(12, 72 - 6 * level)

        self._last_position: Vector = (
            float(self.unit.position[0]),
            float(self.unit.position[1]),
        )
        self._movement_requested = False
        self._stuck_frames = 0
        self._recovery_frames = 0
        self._update_unit_radius()

    # ------------------------------------------------------------------
    # Main decision loop
    # ------------------------------------------------------------------
    def update(self) -> None:
        self._frame_count += 1
        self._update_unit_radius()
        self._update_stuck_state()
        self._refresh_nearby_unit_cache()
        self._exchange_information()

        visible_enemies = self._get_visible_enemy_units()
        has_live_target = self._is_valid_enemy(self.target_unit)
        tactical_due = (
            self._last_tactical_frame < 0
            or (self._frame_count + int(self.unit.id)) % self._tactical_interval == 0
            or (not has_live_target and bool(visible_enemies))
        )
        if tactical_due:
            known_enemies = self._get_known_enemy_units(visible_enemies)
            self._cached_known_enemies = tuple(known_enemies)
            current_target = self._select_target(known_enemies)
            self._last_tactical_frame = self._frame_count
        else:
            known_enemies = [
                enemy
                for enemy in self._cached_known_enemies
                if self._is_valid_enemy(enemy)
            ]
            current_target = (
                self.target_unit
                if has_live_target
                and any(enemy is self.target_unit for enemy in visible_enemies)
                else None
            )
        target_known_now = current_target is not None

        if not target_known_now:
            self._update_search_knowledge()

        if current_target is not None:
            self.target_unit = current_target
            self.last_known_target_position = (
                float(current_target.position[0]),
                float(current_target.position[1]),
            )
            self._target_memory_age = 0
        elif (
            self.target_unit is not None
            and self.target_unit.is_alive
            and self.last_known_target_position is not None
            and self._target_memory_age < self._target_memory_limit
        ):
            current_target = self.target_unit
            self._target_memory_age += 1
        else:
            self._clear_target_memory()

        if current_target is None:
            self._search_for_enemy()
            return

        if target_known_now:
            self._aim_at_target(current_target)
            self._try_fire(current_target, target_known_now=True)
        elif self.last_known_target_position is not None:
            self._aim_at_position(self.last_known_target_position)

        # A short, deterministic reverse-turn breaks contact with walls or unit
        # traffic. Aiming and firing above remain active during recovery.
        if self._recovery_frames > 0 and self.unit.max_speed > 0.0:
            self.ai_state = "recover"
            self._recover_from_stuck()
            return

        if tactical_due:
            self._cached_regroup_ally = self._select_regroup_ally(known_enemies)
        regroup_ally = self._cached_regroup_ally
        if (
            regroup_ally is not None
            and (
                not regroup_ally.is_alive
                or regroup_ally.team != self.unit.team
            )
        ):
            regroup_ally = None
            self._cached_regroup_ally = None
        if regroup_ally is not None:
            self.ai_state = "regroup"
            self._navigate_to(regroup_ally.position, force_path=True)
            return

        if target_known_now:
            self._engage_target(current_target)
        else:
            assert self.last_known_target_position is not None
            distance = self._distance_to_position(self.last_known_target_position)
            if distance <= max(18.0, float(self.game_map.tile_size) * 0.35):
                self._clear_target_memory()
                self._search_for_enemy()
                return
            self.ai_state = "pursue_last_known"
            self._navigate_to(self.last_known_target_position, force_path=True)

    # ------------------------------------------------------------------
    # Perception, communication, and target memory
    # ------------------------------------------------------------------
    def _exchange_information(self) -> None:
        """Actively exchange knowledge only when manager auto-communication is off."""

        if getattr(self.unit_manager, "auto_communicate_enabled", False):
            return
        if self._frame_count - self._last_communication_frame < self._communication_interval:
            return
        self._last_communication_frame = self._frame_count
        self.unit.broadcast(self.unit_manager)
        self.unit.broadcast_receive(self.unit_manager)

    def _get_visible_enemy_units(self) -> list[BaseUnit]:
        """Return direct or communicated enemies from the latest vision pass."""

        enemies: dict[int, BaseUnit] = {}
        visible_container = getattr(self.unit, "visible_units", None)
        for other in getattr(visible_container, "units", ()):
            if self._is_valid_enemy(other):
                enemies[other.id] = other
        return list(enemies.values())

    def _get_known_enemy_units(
        self,
        visible_enemies: Iterable[BaseUnit] = (),
    ) -> list[BaseUnit]:
        known = {enemy.id: enemy for enemy in visible_enemies}

        # The visible container is from the preceding vision pass. Reusing the
        # update-local spatial candidates preserves immediate target acquisition
        # without issuing a second spatial-index query.
        for other in self._nearby_units_for_update():
            if self._is_valid_enemy(other) and self.unit_manager.is_visible(
                self.game_map,
                self.unit,
                other,
            ):
                known[other.id] = other
        return list(known.values())

    def _refresh_nearby_unit_cache(self) -> None:
        tile_size = max(1.0, float(getattr(self.game_map, "tile_size", 64)))
        collision_reach = 104.0 + tile_size + max(
            float(self.unit.collision_size[0]),
            float(self.unit.collision_size[1]),
        )
        query_radius = max(
            float(self.unit.sight_range),
            self.SEPARATION_DISTANCE,
            collision_reach,
        )
        candidates = self.unit_manager.get_units_in_radius(
            self.unit.position,
            query_radius,
        )
        self._cached_nearby_units = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, BaseUnit)
                ),
                key=lambda candidate: int(candidate.id),
            )
        )
        self._spatial_cache_frame = self._frame_count

    def _nearby_units_for_update(self) -> tuple[BaseUnit, ...]:
        if self._spatial_cache_frame != self._frame_count:
            self._refresh_nearby_unit_cache()
        return self._cached_nearby_units

    def _is_valid_enemy(self, other: object) -> TypeGuard[BaseUnit]:
        return bool(
            isinstance(other, BaseUnit)
            and other is not self.unit
            and getattr(other, "is_alive", False)
            and getattr(other, "visible", True)
            and getattr(other, "team", self.unit.team) != self.unit.team
        )

    def _select_target(self, enemies: Iterable[BaseUnit]) -> Optional[BaseUnit]:
        enemies = list(enemies)
        if not enemies:
            return None
        if self.intelligence_level <= 2:
            return min(enemies, key=self._distance_to)

        weapon_range = max(1.0, float(self.unit.weapon_range()))

        def score(enemy: BaseUnit) -> tuple[float, int]:
            distance = self._distance_to(enemy)
            health_ratio = float(enemy.health) / max(1.0, float(enemy.max_health))
            value = distance + health_ratio * (8.0 * self.intelligence_level)
            if distance <= weapon_range and self.unit_manager.has_line_of_sight(
                self.game_map,
                self.unit,
                enemy,
            ):
                value -= 25.0 + 5.0 * self.intelligence_level
            if enemy is self.target_unit:
                value -= 18.0 + 3.0 * self.intelligence_level
            return value, int(enemy.id)

        return min(enemies, key=score)

    def _clear_target_memory(self) -> None:
        had_target_memory = (
            self.target_unit is not None
            or self.last_known_target_position is not None
        )
        self.target_unit = None
        self.last_known_target_position = None
        self._target_memory_age = 0
        if had_target_memory:
            self._path.clear()
            self._path_goal = None

    def _update_search_knowledge(self) -> None:
        """Record explored cells and exchange them with nearby friendly AIs."""

        self._ensure_search_map_cache()
        # Tile coverage changes much more slowly than the 60 Hz control loop.
        # Sampling it a few times per second preserves search quality without
        # turning line-of-sight bookkeeping into the dominant frame cost.
        mark_interval = max(18, 52 - 4 * self.intelligence_level)
        if self._frame_count - self._last_search_mark_frame >= mark_interval:
            self._last_search_mark_frame = self._frame_count
            sight_range = max(0.0, float(self.unit.sight_range))
            for cell in self._passable_search_cells:
                position = self._cell_center(cell)
                if math.dist(self.unit.position, position) > sight_range:
                    continue
                if self.game_map.has_line_of_sight(self.unit.position, position):
                    self._searched_cells.add(cell)

        if self.intelligence_level < self.SEARCH_SHARE_MIN_INTELLIGENCE:
            return
        share_interval = max(10, 55 - 5 * self.intelligence_level)
        if self._frame_count - self._last_search_share_frame < share_interval:
            return
        self._last_search_share_frame = self._frame_count

        for friendly_ai in self._friendly_ai_controllers():
            if friendly_ai.intelligence_level < self.SEARCH_SHARE_MIN_INTELLIGENCE:
                continue
            communication_range = max(
                float(self.unit.communication_range),
                float(friendly_ai.unit.communication_range),
            )
            if self._distance_to(friendly_ai.unit) > communication_range:
                continue
            combined = self._searched_cells | friendly_ai._searched_cells
            self._searched_cells.update(combined)
            friendly_ai._searched_cells.update(combined)

    def _ensure_search_map_cache(self) -> None:
        width = int(getattr(self.game_map, "width", 0))
        height = int(getattr(self.game_map, "height", 0))
        tile_size = float(getattr(self.game_map, "tile_size", 64))
        signature = (id(self.game_map), width, height, tile_size)
        if signature == self._search_map_signature:
            return

        self._search_map_signature = signature
        self._searched_cells.clear()
        self._passable_search_cells.clear()
        self._search_destination = None
        self._path.clear()
        self._path_goal = None
        self._spawn_position = (
            float(self.unit.position[0]),
            float(self.unit.position[1]),
        )
        for y in range(height):
            for x in range(width):
                cell = (x, y)
                if self.game_map.can_place_unit(
                    self._cell_center(cell),
                    self.unit.collision_size,
                ):
                    self._passable_search_cells.append(cell)

    def _cell_center(self, cell: tuple[int, int]) -> Vector:
        tile_size = float(getattr(self.game_map, "tile_size", 64))
        return ((cell[0] + 0.5) * tile_size, (cell[1] + 0.5) * tile_size)

    def _search_coverage(self) -> float:
        self._ensure_search_map_cache()
        if not self._passable_search_cells:
            return 1.0
        return min(1.0, len(self._searched_cells) / len(self._passable_search_cells))

    def _friendly_ai_controllers(self) -> list[EnemyAI]:
        return [
            ai
            for ai in self.unit_manager.enemy_ais
            if isinstance(ai, EnemyAI)
            and ai is not self
            and ai.unit.is_alive
            and ai.unit.team == self.unit.team
        ]

    # ------------------------------------------------------------------
    # Regrouping under pressure
    # ------------------------------------------------------------------
    def _select_regroup_ally(
        self,
        known_enemies: Iterable[BaseUnit],
    ) -> Optional[BaseUnit]:
        if (
            self.intelligence_level < self.REGROUP_MIN_INTELLIGENCE
            or self.unit.max_speed <= 0.0
        ):
            return None

        allies = [
            other
            for other in self.unit_manager.units
            if other is not self.unit
            and other.is_alive
            and other.visible
            and other.team == self.unit.team
        ]
        active_anchor = next(
            (other for other in allies if other.id == self._regroup_ally_id),
            None,
        )
        rally_distance = max(55.0, min(110.0, float(self.unit.sight_range) * 0.45))
        if (
            active_anchor is not None
            and self._frame_count < self._regroup_until_frame
            and self._distance_to(active_anchor) > rally_distance
        ):
            return active_anchor
        if self._regroup_ally_id is not None:
            self._regroup_ally_id = None
            self._regroup_until_frame = 0

        enemies = list(known_enemies)
        if (
            not enemies
            or not allies
            or self._frame_count < self._next_regroup_frame
        ):
            return None

        support_radius = max(90.0, min(220.0, float(self.unit.sight_range) * 0.75))
        local_allies = [
            ally for ally in allies if self._distance_to(ally) <= support_radius
        ]
        attackers: dict[int, BaseUnit] = {}
        for enemy in enemies:
            engagement_range = min(
                max(1.0, float(enemy.sight_range)),
                max(1.0, float(enemy.weapon_range())),
            )
            if (
                self._distance_to(enemy) <= engagement_range * 1.08
                and self.unit_manager.has_line_of_sight(
                    self.game_map,
                    enemy,
                    self.unit,
                )
            ):
                attackers[enemy.id] = enemy

        for bullet in self.bullet_manager.get_bullets_in_radius(
            self.unit.position,
            self.BULLET_THREAT_DISTANCE * 1.35,
            self.game_map,
        ):
            shooter = getattr(bullet, "shooter", None)
            if self._is_valid_enemy(shooter):
                attackers[shooter.id] = shooter

        if not attackers:
            return None

        self_power = self._combat_power(self.unit)
        friendly_power = self_power + sum(
            self._combat_power(ally) for ally in local_allies
        )
        enemy_power = sum(self._combat_power(enemy) for enemy in attackers.values())
        pressure_threshold = 2.10 - 0.065 * self.intelligence_level
        outnumbered = len(attackers) >= len(local_allies) + 2
        overpowered = enemy_power > friendly_power * pressure_threshold
        strongest_enemy = max(
            (self._combat_power(enemy) for enemy in attackers.values()),
            default=0.0,
        )
        badly_matched = (
            not local_allies
            and strongest_enemy > self_power * (pressure_threshold + 0.18)
        )
        health_ratio = float(self.unit.health) / max(1.0, float(self.unit.max_health))
        wounded_and_alone = not local_allies and health_ratio < 0.38
        if not (outnumbered or overpowered or badly_matched or wounded_and_alone):
            return None

        partner_ai = self._get_pair_partner()
        if partner_ai is not None and partner_ai.unit in allies:
            anchor = partner_ai.unit
        else:
            anchor = min(allies, key=self._regroup_anchor_score)

        if self._distance_to(anchor) <= rally_distance:
            return None
        self._regroup_ally_id = int(anchor.id)
        self._regroup_until_frame = self._frame_count + 90 + 15 * self.intelligence_level
        self._next_regroup_frame = self._regroup_until_frame + 240
        return anchor

    def _regroup_anchor_score(self, ally: BaseUnit) -> tuple[float, int]:
        nearby_support = sum(
            1
            for other in self.unit_manager.units
            if other is not ally
            and other.is_alive
            and other.team == self.unit.team
            and math.dist(other.position, ally.position) <= float(ally.sight_range) * 0.7
        )
        health_ratio = float(ally.health) / max(1.0, float(ally.max_health))
        score = self._distance_to(ally) - 45.0 * nearby_support - 35.0 * health_ratio
        return score, int(ally.id)

    @staticmethod
    def _combat_power(unit: BaseUnit) -> float:
        health_ratio = float(unit.health) / max(1.0, float(unit.max_health))
        try:
            spec = unit.get_weapon_spec()
            damage_rate = float(spec.damage_rate) / max(0.15, float(spec.cooldown))
        except (AttributeError, ValueError):
            damage_rate = 0.5
        durability = math.sqrt(max(1.0, float(unit.max_health)) / 100.0)
        return max(0.05, health_ratio) * durability * (0.7 + damage_rate)

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------
    def _engage_target(self, target: BaseUnit) -> None:
        distance = self._distance_to(target)
        has_line_of_sight = self.unit_manager.has_line_of_sight(
            self.game_map,
            self.unit,
            target,
        )
        preferred = self._preferred_combat_distance()
        health_ratio = float(self.unit.health) / max(1.0, float(self.unit.max_health))
        target_health_ratio = float(target.health) / max(1.0, float(target.max_health))
        low_health_threshold = 0.22 + 0.025 * self.intelligence_level
        disadvantaged = health_ratio + 0.10 < target_health_ratio
        should_kite = (
            self.intelligence_level >= 5
            and health_ratio < low_health_threshold
            and disadvantaged
        )

        if not has_line_of_sight:
            self.ai_state = "flank"
            self._navigate_to(target.position, force_path=True)
            return

        if should_kite:
            sustainable_range = 0.85 * min(
                max(1.0, float(self.unit.sight_range)),
                max(1.0, float(self.unit.weapon_range())),
            )
            kite_distance = min(280.0, max(preferred, sustainable_range))
            if distance < kite_distance * 0.88:
                target_dir = self._vector_to_target(target)
                strafe = (-target_dir[1] * self.strafe_sign, target_dir[0] * self.strafe_sign)
                retreat = (-target_dir[0], -target_dir[1])
                desired = self._normalize(
                    (retreat[0] + 0.45 * strafe[0], retreat[1] + 0.45 * strafe[1])
                )
                self.ai_state = "kite"
                self._move_in_direction(desired)
                return
            if distance > kite_distance * 1.15:
                self.ai_state = "kite_reengage"
                self._navigate_to(target.position)
                return

            self.ai_state = "kite_strafe"
            target_dir = self._vector_to_target(target)
            self._move_in_direction(
                (-target_dir[1] * self.strafe_sign, target_dir[0] * self.strafe_sign)
            )
            return

        too_close = max(34.0, preferred * 0.42)
        if distance < too_close and self.unit.max_speed > 0.0:
            target_dir = self._vector_to_target(target)
            self.ai_state = "make_space"
            self._move_in_direction((-target_dir[0], -target_dir[1]))
        elif distance > preferred * 1.18:
            self.ai_state = "approach"
            self._navigate_to(target.position)
        else:
            # Holding a firing position prevents endless mutual circling and lets
            # slower turrets settle. It also makes AI-vs-AI battles conclude.
            self.ai_state = "engage"
            self._brake()

    def _preferred_combat_distance(self) -> float:
        usable_range = min(
            max(1.0, float(self.unit.sight_range)),
            max(1.0, float(self.unit.weapon_range())),
        )
        return max(48.0, min(220.0, usable_range * 0.70))

    def _aim_at_target(self, target: BaseUnit) -> None:
        aim_position: Vector = (
            float(target.position[0]),
            float(target.position[1]),
        )
        if self.intelligence_level >= 6:
            projectile_speed = max(1.0, float(self.unit.get_weapon_spec().speed))
            travel_time = min(1.25, self._distance_to(target) / projectile_speed)
            lead_factor = (self.intelligence_level - 5) / 4.0
            velocity = getattr(target, "velocity", (0.0, 0.0))
            aim_position = (
                target.position[0] + velocity[0] * travel_time * lead_factor,
                target.position[1] + velocity[1] * travel_time * lead_factor,
            )
        self._aim_at_position(aim_position)

    def _aim_at_position(self, position: Vector) -> None:
        dx = position[0] - self.unit.position[0]
        dy = position[1] - self.unit.position[1]
        self.unit.turret_target_angle = (
            math.degrees(math.atan2(dy, dx)) + 90.0
        ) % 360.0

    def _try_fire(self, target: BaseUnit, *, target_known_now: bool) -> bool:
        if not target_known_now or not self.unit.can_fire():
            return False
        if self._distance_to(target) > float(self.unit.weapon_range()):
            return False
        if not self.unit_manager.has_line_of_sight(self.game_map, self.unit, target):
            return False
        angle_diff = self.unit.get_angle_difference(
            self.unit.turret_direction_angle,
            self.unit.turret_target_angle,
        )
        if abs(angle_diff) > self.fire_angle_tolerance:
            return False
        return self.bullet_manager.fire(self.unit) is not None

    # ------------------------------------------------------------------
    # Search and navigation
    # ------------------------------------------------------------------
    def _search_for_enemy(self) -> None:
        if not any(self._is_valid_enemy(other) for other in self.unit_manager.units):
            self.ai_state = "idle"
            self._brake()
            return

        partner_ai, searching_separately = self._update_search_pairing()
        if (
            partner_ai is not None
            and not searching_separately
            and self._follow_search_partner(partner_ai)
        ):
            return

        if (
            self._search_destination is None
            or self._distance_to_position(self._search_destination)
            <= max(24.0, float(self.game_map.tile_size) * 0.45)
            or self._stuck_frames > self.STUCK_FRAME_LIMIT // 2
        ):
            self._search_destination = self._choose_search_destination()
            self._path.clear()
            self._path_goal = None

        if self._search_destination is None:
            self.ai_state = "scan"
            self._brake()
            self.unit.turret_target_angle = (
                self.unit.turret_target_angle + 2.0 + self.intelligence_level * 0.25
            ) % 360.0
            return

        self.ai_state = "search_split" if searching_separately else "search"
        if partner_ai is not None and self.unit.id < partner_ai.unit.id:
            self.ai_state = "search_pair_lead"
        self._aim_at_position(self._search_destination)
        self._navigate_to(self._search_destination, force_path=True)

    def _update_search_pairing(self) -> tuple[Optional[EnemyAI], bool]:
        if (
            self.intelligence_level < self.PAIR_MIN_INTELLIGENCE
            or self.unit.max_speed <= 0.0
        ):
            return None, False

        partner_ai = self._get_pair_partner()
        if partner_ai is None:
            candidates = []
            for friendly_ai in self._friendly_ai_controllers():
                if (
                    friendly_ai.intelligence_level < self.PAIR_MIN_INTELLIGENCE
                    or friendly_ai.unit.max_speed <= 0.0
                    or friendly_ai._get_pair_partner() is not None
                ):
                    continue
                if self.unit_manager.is_visible(
                    self.game_map,
                    self.unit,
                    friendly_ai.unit,
                ):
                    candidates.append(friendly_ai)
            if candidates:
                partner_ai = min(
                    candidates,
                    key=lambda ai: (self._distance_to(ai.unit), int(ai.unit.id)),
                )
                formed_frame = max(self._frame_count, partner_ai._frame_count)
                self._pair_partner_id = int(partner_ai.unit.id)
                partner_ai._pair_partner_id = int(self.unit.id)
                self._pair_formed_frame = formed_frame
                partner_ai._pair_formed_frame = formed_frame
                next_split = formed_frame + max(
                    180,
                    360 - 15 * min(self.intelligence_level, partner_ai.intelligence_level),
                )
                self._next_pair_split_frame = next_split
                partner_ai._next_pair_split_frame = next_split
                self._search_destination = None
                partner_ai._search_destination = None

        if partner_ai is None:
            return None, False

        effective_level = min(self.intelligence_level, partner_ai.intelligence_level)
        is_split = self._frame_count < self._pair_split_until_frame
        may_split = (
            effective_level >= 6
            and self.unit.id < partner_ai.unit.id
            and self._frame_count >= self._next_pair_split_frame
            and self._search_coverage() < 0.55
        )
        if may_split:
            split_until = self._frame_count + max(90, 210 - 10 * effective_level)
            next_split = split_until + max(240, 480 - 20 * effective_level)
            self._pair_split_until_frame = split_until
            partner_ai._pair_split_until_frame = split_until
            self._next_pair_split_frame = next_split
            partner_ai._next_pair_split_frame = next_split
            self._search_destination = None
            partner_ai._search_destination = None
            self._path.clear()
            partner_ai._path.clear()
            is_split = True
        return partner_ai, is_split

    def _get_pair_partner(self) -> Optional[EnemyAI]:
        if self._pair_partner_id is None:
            return None
        partner_ai = next(
            (
                ai
                for ai in self._friendly_ai_controllers()
                if ai.unit.id == self._pair_partner_id
            ),
            None,
        )
        if (
            partner_ai is None
            or partner_ai._pair_partner_id not in (None, self.unit.id)
        ):
            self._pair_partner_id = None
            self._pair_split_until_frame = 0
            return None
        if partner_ai._pair_partner_id is None:
            partner_ai._pair_partner_id = int(self.unit.id)
        return partner_ai

    def _follow_search_partner(self, partner_ai: EnemyAI) -> bool:
        # The lower stable id leads; the other vehicle keeps a loose formation.
        if self.unit.id < partner_ai.unit.id:
            return False

        spacing = max(48.0, min(92.0, float(self.unit.sight_range) * 0.38))
        if partner_ai.target_unit is not None:
            formation_position: Vector = (
                float(partner_ai.unit.position[0]),
                float(partner_ai.unit.position[1]),
            )
        else:
            destination = partner_ai._search_destination or (
                float(partner_ai.unit.position[0]),
                float(partner_ai.unit.position[1]),
            )
            heading = self._normalize(
                (
                    destination[0] - partner_ai.unit.position[0],
                    destination[1] - partner_ai.unit.position[1],
                ),
                fallback=partner_ai._unit_forward_vector(),
            )
            side = 1.0 if self.unit.id % 2 == 0 else -1.0
            formation_position = (
                partner_ai.unit.position[0] - heading[0] * spacing
                - heading[1] * spacing * 0.35 * side,
                partner_ai.unit.position[1] - heading[1] * spacing
                + heading[0] * spacing * 0.35 * side,
            )
            if not self.game_map.can_place_unit(
                formation_position,
                self.unit.collision_size,
            ):
                formation_position = (
                    float(partner_ai.unit.position[0]),
                    float(partner_ai.unit.position[1]),
                )

        self.ai_state = "search_pair_follow"
        if self._distance_to_position(formation_position) <= 22.0:
            self._brake()
        else:
            self._aim_at_position(formation_position)
            self._navigate_to(formation_position, force_path=True)
        return True

    def _choose_search_destination(self) -> Optional[Vector]:
        self._ensure_search_map_cache()
        width = int(getattr(self.game_map, "width", 0))
        height = int(getattr(self.game_map, "height", 0))
        tile_size = float(getattr(self.game_map, "tile_size", 64))
        if width <= 0 or height <= 0 or not self._passable_search_cells:
            return None

        candidates = [
            cell
            for cell in self._passable_search_cells
            if self._distance_to_position(self._cell_center(cell)) > tile_size * 1.5
        ]
        if not candidates:
            candidates = list(self._passable_search_cells)

        left_start = self._spawn_position[0] <= width * tile_size * 0.5
        candidates.sort(
            key=lambda cell: (
                cell[1],
                cell[0] if left_start else width - 1 - cell[0],
            )
        )
        team_controllers = sorted(
            [self, *self._friendly_ai_controllers()],
            key=lambda ai: int(ai.unit.id),
        )
        team_rank = team_controllers.index(self)
        seed = (
            width * 73_856_093
            ^ height * 19_349_663
            ^ team_rank * 83_492_791
            ^ self._search_index * 2_654_435_761
        )
        rng = random.Random(seed)
        map_diagonal = max(1.0, math.hypot(width * tile_size, height * tile_size))
        teammate_destinations: list[Vector] = []
        for friendly_ai in self._friendly_ai_controllers():
            destination = friendly_ai._search_destination
            if destination is not None:
                teammate_destinations.append(destination)

        scored_cells = []
        for cell in candidates:
            position = self._cell_center(cell)
            novelty = self._local_search_novelty(cell)
            current_distance = self._distance_to_position(position) / map_diagonal
            spawn_distance = math.dist(self._spawn_position, position) / map_diagonal
            destination_overlap = sum(
                max(0.0, 1.0 - math.dist(position, other) / (tile_size * 4.0))
                for other in teammate_destinations
            )
            random_weight = max(0.18, 0.72 - 0.055 * self.intelligence_level)
            if self._search_index == 0:
                # The first patrol destination is on the far side of the map.
                score = 4.0 * spawn_distance + 1.4 * novelty
            else:
                score = (
                    (1.8 + 0.16 * self.intelligence_level) * novelty
                    + 0.55 * current_distance
                    - 1.2 * destination_overlap
                )
            score += rng.random() * random_weight
            scored_cells.append((score, cell))

        preferred_cell = max(scored_cells, key=lambda item: item[0])[1]
        self._search_index += 1
        return self._cell_center(preferred_cell)

    def _local_search_novelty(self, cell: tuple[int, int]) -> float:
        passable = set(self._passable_search_cells)
        nearby_cells = [
            (x, y)
            for y in range(cell[1] - 2, cell[1] + 3)
            for x in range(cell[0] - 2, cell[0] + 3)
            if (x, y) in passable
        ]
        if not nearby_cells:
            return 0.0
        unexplored = sum(candidate not in self._searched_cells for candidate in nearby_cells)
        return unexplored / len(nearby_cells)

    def _nearest_passable_cell(
        self,
        preferred_cell: tuple[int, int],
    ) -> Optional[tuple[int, int]]:
        width = int(getattr(self.game_map, "width", 0))
        height = int(getattr(self.game_map, "height", 0))
        tile_size = float(getattr(self.game_map, "tile_size", 64))
        max_radius = max(width, height)
        px, py = preferred_cell
        for radius in range(max_radius + 1):
            candidates = []
            for y in range(max(0, py - radius), min(height, py + radius + 1)):
                for x in range(max(0, px - radius), min(width, px + radius + 1)):
                    if max(abs(x - px), abs(y - py)) == radius:
                        candidates.append((x, y))
            candidates.sort(key=lambda cell: (cell[1], cell[0]))
            for cell in candidates:
                position = ((cell[0] + 0.5) * tile_size, (cell[1] + 0.5) * tile_size)
                if self.game_map.can_place_unit(position, self.unit.collision_size):
                    return cell
        return None

    def _navigate_to(self, destination: Vector, *, force_path: bool = False) -> None:
        desired = self._direction_to_position(destination)
        distance = self._distance_to_position(destination)
        direct_probe = min(
            distance,
            max(float(self.game_map.tile_size), abs(float(self.unit.speed)) * 1.2 + 28.0),
        )
        direct_safe = self._is_path_safe(desired, direct_probe)

        use_path = self.intelligence_level >= 3 and (force_path or not direct_safe)
        if use_path:
            self._ensure_path(destination)
            waypoint = self._current_path_waypoint()
            if waypoint is not None:
                desired = self._direction_to_position(waypoint)
        elif direct_safe:
            self._path.clear()
            self._path_goal = None

        self._move_in_direction(desired)

    def _ensure_path(self, destination: Vector) -> None:
        tile_size = float(getattr(self.game_map, "tile_size", 64))
        goal_changed = (
            self._path_goal is None
            or math.dist(self._path_goal, destination) > tile_size * 0.75
        )
        replan_due = (
            self._frame_count - self._last_path_frame >= self._path_replan_interval
        )
        stuck_replan = self._stuck_frames >= self.STUCK_FRAME_LIMIT // 2
        if self._path and not goal_changed and not replan_due and not stuck_replan:
            return
        self._path = self._build_grid_path(destination)
        self._path_goal = (float(destination[0]), float(destination[1]))
        self._last_path_frame = self._frame_count

    def _current_path_waypoint(self) -> Optional[Vector]:
        threshold = max(10.0, float(self.game_map.tile_size) * 0.30)
        while self._path and self._distance_to_position(self._path[0]) <= threshold:
            self._path.pop(0)
        return self._path[0] if self._path else None

    def _build_grid_path(self, destination: Vector) -> list[Vector]:
        width = int(getattr(self.game_map, "width", 0))
        height = int(getattr(self.game_map, "height", 0))
        tile_size = float(getattr(self.game_map, "tile_size", 64))
        if width <= 0 or height <= 0:
            return []

        start = (
            min(width - 1, max(0, int(self.unit.position[0] // tile_size))),
            min(height - 1, max(0, int(self.unit.position[1] // tile_size))),
        )
        raw_goal = (
            min(width - 1, max(0, int(destination[0] // tile_size))),
            min(height - 1, max(0, int(destination[1] // tile_size))),
        )
        goal = self._nearest_passable_cell(raw_goal)
        if goal is None or start == goal:
            return []

        def passable(cell: tuple[int, int]) -> bool:
            position = ((cell[0] + 0.5) * tile_size, (cell[1] + 0.5) * tile_size)
            return self.game_map.can_place_unit(position, self.unit.collision_size)

        frontier: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
        came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
        cost_so_far = {start: 0.0}
        while frontier:
            _, current_cost, current = heapq.heappop(frontier)
            if current == goal:
                break
            if current_cost > cost_so_far.get(current, float("inf")):
                continue
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbour = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbour[0] < width and 0 <= neighbour[1] < height):
                    continue
                if neighbour != goal and not passable(neighbour):
                    continue
                new_cost = current_cost + 1.0
                if new_cost >= cost_so_far.get(neighbour, float("inf")):
                    continue
                cost_so_far[neighbour] = new_cost
                priority = new_cost + abs(goal[0] - neighbour[0]) + abs(goal[1] - neighbour[1])
                heapq.heappush(frontier, (priority, new_cost, neighbour))
                came_from[neighbour] = current

        if goal not in came_from:
            return []
        cells = []
        current = goal
        while current != start:
            cells.append(current)
            parent = came_from[current]
            if parent is None:
                break
            current = parent
        cells.reverse()
        return [((x + 0.5) * tile_size, (y + 0.5) * tile_size) for x, y in cells]

    # ------------------------------------------------------------------
    # Steering, obstacle avoidance, and recovery
    # ------------------------------------------------------------------
    def _move_in_direction(self, desired: Vector) -> None:
        requested = self._normalize(desired)
        direction_changed = (
            self._last_steering_input is None
            or self._dot(requested, self._last_steering_input) < 0.85
        )
        steering_due = (
            self._cached_steering_direction is None
            or direction_changed
            or (self._frame_count + int(self.unit.id)) % self._steering_interval == 0
        )
        if steering_due:
            desired = self._blend_separation(requested)
            if self.intelligence_level >= 7:
                evade = self._get_bullet_evasion_direction(desired)
                if evade is not None:
                    desired = evade
                    self.ai_state = "evade_bullet"

            steered, avoiding = self._choose_safe_direction(desired)
            self._last_steering_input = requested
            self._cached_steering_direction = steered
            self._cached_steering_avoiding = avoiding
        else:
            steered = self._cached_steering_direction
            avoiding = self._cached_steering_avoiding

        if steered is None:
            self._start_recovery()
            self._recover_from_stuck()
            return
        if avoiding and self.ai_state not in {"evade_bullet", "recover"}:
            self.ai_state = f"{self.ai_state}_avoid"
        self._drive_towards_direction(steered)

    def _choose_safe_direction(self, desired: Vector) -> tuple[Optional[Vector], bool]:
        lookahead = max(
            26.0,
            min(104.0, abs(float(self.unit.speed)) * 1.25 + 28.0 + 3.0 * self.intelligence_level),
        )
        if self._is_path_safe(desired, lookahead):
            return desired, False

        angle_step = max(12, 30 - 2 * self.intelligence_level)
        forward = self._unit_forward_vector()
        for angle in range(angle_step, 181, angle_step):
            safe_candidates = []
            for offset in (
                angle * self._avoidance_sign,
                -angle * self._avoidance_sign,
            ):
                candidate = self._rotate(desired, offset)
                if not self._is_path_safe(candidate, lookahead):
                    continue
                progress = self._dot(candidate, desired)
                turn_cost = angle / 180.0
                steering_continuity = self._dot(candidate, forward)
                side_bonus = 0.06 if offset * self._avoidance_sign > 0 else 0.0
                score = (
                    2.2 * progress
                    + 0.25 * steering_continuity
                    - 0.35 * turn_cost
                    + side_bonus
                )
                safe_candidates.append((score, candidate))
            if safe_candidates:
                return max(safe_candidates, key=lambda item: item[0])[1], True
        return None, True

    def _drive_towards_direction(self, desired: Vector) -> None:
        desired_angle = (math.degrees(math.atan2(desired[1], desired[0])) + 90.0) % 360.0
        angle_diff = self.unit.get_angle_difference(self.unit.direction_angle, desired_angle)
        turn_threshold = max(2.5, 8.0 - 0.6 * self.intelligence_level)
        self.unit.set_turning(angle_diff < -turn_threshold, angle_diff > turn_threshold)

        # A vehicle keeps its current speed when acceleration is zero. Brake
        # before large turns instead of continuing straight into the obstacle.
        if abs(angle_diff) > 52.0:
            self._brake()
            return
        self.unit.set_movement(forward=True, backward=False)
        self._movement_requested = self.unit.max_speed > 0.0

    def _brake(self) -> None:
        speed = float(self.unit.speed)
        stopping_threshold = max(0.5, float(self.unit.max_acceleration) / 60.0)
        if speed > stopping_threshold:
            self.unit.set_movement(forward=False, backward=True)
            self._movement_requested = True
        elif speed < -stopping_threshold:
            self.unit.set_movement(forward=True, backward=False)
            self._movement_requested = True
        else:
            self.unit.speed = 0.0
            self.unit.set_movement(False, False)
            self._movement_requested = False

    def _update_stuck_state(self) -> None:
        moved = math.dist(self.unit.position, self._last_position)
        if self._movement_requested and self.unit.max_speed > 0.0:
            if moved <= self.STUCK_DISTANCE_EPSILON:
                self._stuck_frames += 1
            else:
                self._stuck_frames = max(0, self._stuck_frames - 2)
        else:
            self._stuck_frames = max(0, self._stuck_frames - 1)
        self._last_position = (
            float(self.unit.position[0]),
            float(self.unit.position[1]),
        )
        self._movement_requested = False
        if self._stuck_frames >= self.STUCK_FRAME_LIMIT:
            self._start_recovery()

    def _start_recovery(self) -> None:
        if self.unit.max_speed <= 0.0:
            return
        self._recovery_frames = self.RECOVERY_FRAMES
        self._stuck_frames = 0
        self._avoidance_sign *= -1.0
        self._path.clear()
        self._path_goal = None
        self._cached_steering_direction = None
        self._last_steering_input = None

    def _recover_from_stuck(self) -> None:
        if self._recovery_frames <= 0:
            return
        self._recovery_frames -= 1
        turn_right = self._avoidance_sign > 0.0
        self.unit.set_turning(left=not turn_right, right=turn_right)
        self.unit.set_movement(forward=False, backward=True)
        self._movement_requested = True

    def _is_path_safe(self, direction: Vector, distance: float) -> bool:
        direction = self._normalize(direction)
        step = max(6.0, min(16.0, float(self.game_map.tile_size) * 0.25))
        samples = max(1, int(math.ceil(max(0.0, distance) / step)))
        for index in range(1, samples + 1):
            probe_distance = min(distance, index * step)
            position = (
                self.unit.position[0] + direction[0] * probe_distance,
                self.unit.position[1] + direction[1] * probe_distance,
            )
            if not self._is_position_safe(position):
                return False
        return True

    def _is_position_safe(self, position: Vector) -> bool:
        width = float(self.unit.collision_size[0]) + 2.0 * self.SAFE_MARGIN
        height = float(self.unit.collision_size[1]) + 2.0 * self.SAFE_MARGIN
        if not self.game_map.can_place_unit(position, (width, height)):
            return False
        if not getattr(self.unit_manager, "enable_unit_collision", False):
            return True
        rect = pygame.Rect(0, 0, int(math.ceil(width)), int(math.ceil(height)))
        rect.center = (round(position[0]), round(position[1]))
        for other in self._nearby_units_for_update():
            if other is self.unit or not other.is_alive:
                continue
            other_box = other.collision_box
            if other_box is not None and rect.colliderect(other_box):
                return False
        return True

    def _blend_separation(self, desired: Vector) -> Vector:
        repel_x = 0.0
        repel_y = 0.0
        for other in self._nearby_units_for_update():
            if other is self.unit or not getattr(other, "is_alive", False):
                continue
            dx = self.unit.position[0] - other.position[0]
            dy = self.unit.position[1] - other.position[1]
            distance = math.hypot(dx, dy)
            if distance <= 1e-6 or distance >= self.SEPARATION_DISTANCE:
                continue
            weight = (self.SEPARATION_DISTANCE - distance) / self.SEPARATION_DISTANCE
            repel_x += dx / distance * weight
            repel_y += dy / distance * weight
        separation_weight = 0.55 + 0.06 * self.intelligence_level
        mixed = (
            desired[0] + repel_x * separation_weight,
            desired[1] + repel_y * separation_weight,
        )
        return self._normalize(mixed, fallback=desired)

    # ------------------------------------------------------------------
    # Projectile threat avoidance
    # ------------------------------------------------------------------
    def _get_bullet_evasion_direction(self, desired: Vector) -> Optional[Vector]:
        threats = self._detect_threatening_bullets()
        if not threats:
            return None
        bullet = min(threats, key=lambda item: math.dist(item.position, self.unit.position))
        velocity = self._normalize(
            (float(bullet.velocity[0]), float(bullet.velocity[1]))
        )
        candidates = ((-velocity[1], velocity[0]), (velocity[1], -velocity[0]))
        safe_candidates = [
            candidate
            for candidate in candidates
            if self._is_path_safe(candidate, max(28.0, float(self.unit.max_speed) * 0.65))
        ]
        if not safe_candidates:
            return None
        return max(safe_candidates, key=lambda candidate: self._dot(candidate, desired))

    def _detect_threatening_bullets(self) -> list[Any]:
        threats = []
        nearby = self.bullet_manager.get_bullets_in_radius(
            self.unit.position,
            self.BULLET_THREAT_DISTANCE,
            self.game_map,
        )
        for bullet in nearby:
            if bullet.shooter_team == self.unit.team or not bullet.is_active:
                continue
            vx, vy = bullet.velocity
            speed_sq = vx * vx + vy * vy
            if speed_sq <= 1e-9:
                continue
            to_unit = (
                self.unit.position[0] - bullet.position[0],
                self.unit.position[1] - bullet.position[1],
            )
            time_to_closest = self._dot(to_unit, (vx, vy)) / speed_sq
            if not 0.0 < time_to_closest <= self.BULLET_LOOKAHEAD_TIME:
                continue
            closest = (
                bullet.position[0] + vx * time_to_closest,
                bullet.position[1] + vy * time_to_closest,
            )
            bullet_radius = max(
                float(getattr(bullet.bounding_box, "width", 0)),
                float(getattr(bullet.bounding_box, "height", 0)),
            ) / 2.0
            if math.dist(closest, self.unit.position) <= self.unit_radius + bullet_radius:
                threats.append(bullet)
        return threats

    # ------------------------------------------------------------------
    # Small geometry helpers
    # ------------------------------------------------------------------
    def _update_unit_radius(self) -> None:
        width, height = self.unit.collision_size
        self.unit_radius = math.hypot(width, height) / 2.0

    def _distance_to(self, target: BaseUnit) -> float:
        return self._distance_to_position(target.position)

    def _distance_to_position(self, position: Vector) -> float:
        return math.dist(self.unit.position, position)

    def _vector_to_target(self, target: BaseUnit) -> Vector:
        return self._direction_to_position(target.position)

    def _direction_to_position(self, position: Vector) -> Vector:
        return self._normalize(
            (
                position[0] - self.unit.position[0],
                position[1] - self.unit.position[1],
            )
        )

    def _unit_forward_vector(self) -> Vector:
        radians = math.radians(self.unit.direction_angle - 90.0)
        return math.cos(radians), math.sin(radians)

    @staticmethod
    def _dot(first: Vector, second: Vector) -> float:
        return first[0] * second[0] + first[1] * second[1]

    @staticmethod
    def _rotate(vector: Vector, angle_degrees: float) -> Vector:
        radians = math.radians(angle_degrees)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        return (
            vector[0] * cosine - vector[1] * sine,
            vector[0] * sine + vector[1] * cosine,
        )

    @staticmethod
    def _normalize(vector: Vector, fallback: Vector = (1.0, 0.0)) -> Vector:
        length = math.hypot(vector[0], vector[1])
        if length <= 1e-9:
            return fallback
        return vector[0] / length, vector[1] / length
