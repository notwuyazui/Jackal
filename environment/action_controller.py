"""Translate discrete reinforcement-learning actions into game commands."""

from __future__ import annotations

import math
from typing import Sequence

from game.BattleState import UnitSnapshot, WorldSnapshot
from game.BattleWorld import BattleWorld


_CHASSIS_COMMANDS = (
    (False, False, False, False),
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (False, False, False, True),
    (True, False, True, False),
    (True, False, False, True),
    (False, True, True, False),
    (False, True, False, True),
)


class ActionController:
    """Own action-space rules while BattleWorld owns simulation commands."""

    def __init__(
        self,
        *,
        auto_aim: bool,
        fire_angle_tolerance: float | None,
        auto_fire_when_ready: bool,
    ) -> None:
        self.auto_aim = bool(auto_aim)
        self.fire_angle_tolerance = fire_angle_tolerance
        self.auto_fire_when_ready = bool(auto_fire_when_ready)
        self.n_actions = 10 if self.auto_aim else 28

    def available_actions(
        self,
        world: BattleWorld,
        snapshot: WorldSnapshot,
        agent_id: int,
    ) -> list[int]:
        available = [0] * self.n_actions
        agent = snapshot.allies[agent_id]
        if not agent.alive:
            available[0] = 1
        elif self.auto_aim:
            available[:9] = [1] * 9
            available[9] = int(
                world.can_unit_fire(agent.unit_id)
                and self._has_fire_target(agent, snapshot.enemies)
            )
        else:
            available[:] = [1] * self.n_actions
        return available

    def apply(
        self,
        world: BattleWorld,
        snapshot: WorldSnapshot,
        actions: Sequence[int],
    ) -> None:
        enemies = snapshot.enemies
        for agent_id, agent in enumerate(snapshot.allies):
            if not agent.alive:
                continue
            action = actions[agent_id]
            if self.auto_aim:
                self._apply_auto_aim(world, agent, enemies, action)
            else:
                self._apply_manual(world, agent, action)

    @staticmethod
    def _set_chassis(world: BattleWorld, unit_id: int, action: int) -> None:
        world.set_unit_chassis(unit_id, _CHASSIS_COMMANDS[action])

    def _apply_auto_aim(
        self,
        world: BattleWorld,
        agent: UnitSnapshot,
        enemies: Sequence[UnitSnapshot],
        action: int,
    ) -> None:
        self._set_chassis(world, agent.unit_id, action if 0 <= action < 9 else 0)
        if (
            action == 9
            and world.can_unit_fire(agent.unit_id)
            and self._has_fire_target(agent, enemies)
        ):
            world.set_unit_fire(agent.unit_id)

        closest = min(
            (
                enemy
                for enemy in enemies
                if enemy.alive and agent.can_see_unit(enemy.unit_id)
            ),
            key=lambda enemy: _distance(agent, enemy),
            default=None,
        )
        turret_target_angle = (
            agent.direction_angle if closest is None else _target_angle(agent, closest)
        )
        world.set_unit_turret_target_angle(agent.unit_id, turret_target_angle)

        if (
            self.auto_fire_when_ready
            and action < 9
            and world.can_unit_fire(agent.unit_id)
            and self._has_fire_target(agent, enemies)
        ):
            world.set_unit_fire(agent.unit_id)

    def _apply_manual(
        self,
        world: BattleWorld,
        agent: UnitSnapshot,
        action: int,
    ) -> None:
        if action < 27:
            self._set_chassis(world, agent.unit_id, action % 9)
            turret_action = action // 9
            if turret_action == 1:
                target_angle = agent.turret_direction_angle - 15.0
            elif turret_action == 2:
                target_angle = agent.turret_direction_angle + 15.0
            else:
                target_angle = agent.turret_direction_angle
            world.set_unit_turret_target_angle(agent.unit_id, target_angle)
            return

        self._set_chassis(world, agent.unit_id, 0)
        if action == 27:
            world.set_unit_turret_target_angle(
                agent.unit_id,
                agent.turret_direction_angle,
            )
            if world.can_unit_fire(agent.unit_id):
                world.set_unit_fire(agent.unit_id)

    def _has_fire_target(
        self,
        agent: UnitSnapshot,
        enemies: Sequence[UnitSnapshot],
    ) -> bool:
        for enemy in enemies:
            if not enemy.alive or not agent.can_see_unit(enemy.unit_id):
                continue
            if _distance(agent, enemy) > agent.weapon_range:
                continue
            if self.fire_angle_tolerance is not None:
                angle_difference = abs(
                    _angle_difference(
                        agent.turret_direction_angle,
                        _target_angle(agent, enemy),
                    )
                )
                if angle_difference > float(self.fire_angle_tolerance):
                    continue
            return True
        return False


def _distance(source, target) -> float:
    return math.hypot(
        target.position[0] - source.position[0],
        target.position[1] - source.position[1],
    )


def _target_angle(source, target) -> float:
    return (
        math.degrees(
            math.atan2(
                target.position[1] - source.position[1],
                target.position[0] - source.position[0],
            )
        )
        + 90
    ) % 360


def _angle_difference(angle1: float, angle2: float) -> float:
    return (angle2 - angle1 + 180.0) % 360.0 - 180.0
