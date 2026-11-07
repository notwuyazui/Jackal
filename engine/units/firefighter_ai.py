"""Border recording: alternating three-second spray and repositioning bursts."""
from __future__ import annotations

import math
from random import Random

from engine.units.base_unit import BaseUnit
from engine.units.enemy_ai import EnemyAI


class FirefighterDirector:
    # The closer recording formation still drives a visible 13 m approach
    # before the first synchronized spray burst.
    ADVANCE_DISTANCE = 52.0
    RETARGET_INTERVAL = 5.0
    ARRIVAL_TOLERANCE = 1.0
    APPROACH_SPEED = 12.0
    STANDOFF_DISTANCE = 185.0
    PHASE_DURATION = 3.0
    # The new house is outside this old arena. Vehicles never enter the scenery.
    MOVEMENT_BOUNDS = (64.0, 64.0, 896.0, 576.0)

    def __init__(self, crew: list[BaseUnit], fire_sites: list[BaseUnit], *, seed: int = 42):
        self.crew = sorted(crew, key=lambda unit: unit.position[1])
        self.MOVEMENT_BOUNDS = self.crew[0].movement_bounds or self.MOVEMENT_BOUNDS
        self.sites = sorted(fire_sites, key=lambda unit: unit.position[1])
        self.paired = {unit.id: site for unit, site in zip(self.crew, self.sites)}
        self.targets = dict(self.paired)
        self.staging = {unit.id: (unit.position[0] + self.ADVANCE_DISTANCE, unit.position[1])
                        for unit in self.crew}
        self.started_at: float | None = None
        self.selection_round = 0
        self._rng = Random(seed)
        self._motion_rng = Random(seed ^ 0xF17E)
        self.flank_signs = {unit.id: (-1 if index == 0 else 1)
                            for index, unit in enumerate(self.crew)}
        self.motion_round = -1
        self.elapsed = 0.0
        self._tick = -1

    def update_tick(self, tick: int) -> None:
        # Shared simulation time and one draw per vehicle per five-second
        # interval make random choices independent of controller update order.
        if tick == self._tick:
            return
        self._tick = tick
        now = min(unit.living_time for unit in self.crew)
        staged = all(math.dist(unit.position, self.staging[unit.id]) <= self.ARRIVAL_TOLERANCE
                     and abs(unit.speed) < .1 and math.hypot(*unit.velocity) < .1
                     for unit in self.crew)
        if self.started_at is None and staged:
            self.started_at = now
        if self.started_at is None:
            return
        self.elapsed = max(0, now - self.started_at)
        motion_round = int((self.elapsed + 1e-9) // (2 * self.PHASE_DURATION))
        if motion_round > self.motion_round:
            self.motion_round = motion_round
            for unit in self.crew:
                # A separate RNG keeps left/right choices independent of aim.
                # Flip on most repositionings, occasionally keep the same side.
                if self._motion_rng.random() < .75:
                    self.flank_signs[unit.id] *= -1
        selection_round = int((now - self.started_at + 1e-9) // self.RETARGET_INTERVAL)
        live_sites = [site for site in self.sites if site.is_alive]
        while self.selection_round < selection_round:
            if live_sites:
                for unit in self.crew:
                    # Sampling with replacement deliberately allows retaining
                    # the current target as well as choosing either other site.
                    self.targets[unit.id] = self._rng.choice(live_sites)
            self.selection_round += 1


class FirefighterAI(EnemyAI):
    def __init__(self, *args, director: FirefighterDirector, **kwargs):
        super().__init__(*args, **kwargs)
        self.director = director

    def _is_position_safe(self, position) -> bool:
        left, top, right, bottom = self.director.MOVEMENT_BOUNDS
        half_w = self.unit.collision_size[0] / 2 + self.SAFE_MARGIN + 2
        half_h = self.unit.collision_size[1] / 2 + self.SAFE_MARGIN + 2
        if not (left + half_w <= position[0] <= right - half_w and
                top + half_h <= position[1] <= bottom - half_h):
            return False
        return super()._is_position_safe(position)

    def _approach_flank(self) -> None:
        # Blend a forward approach with tangential motion. Near the fire site
        # this becomes a gentle flank rather than a forward/reverse shuttle.
        dx = self.target_unit.position[0] - self.unit.position[0]
        dy = self.target_unit.position[1] - self.unit.position[1]
        distance = max(1, math.hypot(dx, dy))
        radial = max(.15, min(1, (distance - self.director.STANDOFF_DISTANCE) / 100))
        lateral = .85
        flank_sign = self.director.flank_signs[self.unit.id]
        requested = self._normalize((
            dx / distance * radial - dy / distance * lateral * flank_sign,
            dy / distance * radial + dx / distance * lateral * flank_sign,
        ))
        self._refresh_nearby_unit_cache()
        direction, _ = self._choose_safe_direction(self._blend_separation(requested))
        if direction is None:
            self._brake()
            return
        heading = (math.degrees(math.atan2(direction[1], direction[0])) + 90) % 360
        error = self.unit.get_angle_difference(self.unit.direction_angle, heading)
        turn_speed = min(35, self.unit.max_angular_speed)
        self.unit.angular_speed = max(-turn_speed, min(turn_speed, error * 2))
        desired_speed = min(self.director.APPROACH_SPEED, self.unit.max_speed) * max(.35, math.cos(math.radians(error)))
        forward = (math.cos(math.radians(self.unit.direction_angle - 90)),
                   math.sin(math.radians(self.unit.direction_angle - 90)))
        if not self._is_path_safe(forward, max(10, self.unit.speed ** 2 /
                                             (2 * max(1, abs(self.unit.min_acceleration))) + 4)):
            # Turn in place before advancing along a safe heading at the edge.
            desired_speed = 0
        self.unit.set_movement(self.unit.speed < desired_speed - .25,
                               self.unit.speed > desired_speed + .25)
        self._movement_requested = True

    def update(self) -> None:
        self._fire_requested = False
        self._frame_count += 1
        self.director.update_tick(self.unit_manager.current_tick)
        self.unit.set_turning(False, False)
        paired = self.director.paired[self.unit.id]
        if self.director.started_at is None:
            self.ai_state = 'approach_fire'
            self.target_unit = paired
            self._aim_at_target(paired)
            remaining = self.director.staging[self.unit.id][0] - self.unit.position[0]
            braking_distance = self.unit.speed ** 2 / (2 * max(1, abs(self.unit.min_acceleration)))
            if remaining > braking_distance + self.director.ARRIVAL_TOLERANCE:
                self.unit.set_movement(forward=True)
            else:
                self._brake()
            return

        self.target_unit = self.director.targets[self.unit.id]
        if self.target_unit.is_alive:
            self._aim_at_target(self.target_unit)
            phase = int((self.director.elapsed + 1e-9) // self.director.PHASE_DURATION) % 2
            if phase == 0:
                self.ai_state = 'spray_burst'
                self._brake()
                # Keep spraying through nozzle turns during the spray phase.
                self._fire_requested = self.unit.can_fire()
            else:
                self.ai_state = 'reposition'
                self._approach_flank()
