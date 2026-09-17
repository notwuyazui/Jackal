"""Translate discrete reinforcement-learning actions into game commands."""

from __future__ import annotations

import math
from typing import Any, Sequence

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
        agents: Sequence[Any],
        enemies: Sequence[Any],
        agent_id: int,
    ) -> list[int]:
        available = [0] * self.n_actions
        agent = agents[agent_id]
        if not agent.is_alive:
            available[0] = 1
        elif self.auto_aim:
            available[:9] = [1] * 9
            available[9] = int(
                agent.can_fire() and self._has_fire_target(world, agent, enemies)
            )
        else:
            available[:] = [1] * self.n_actions
        return available

    def apply(
        self,
        world: BattleWorld,
        agents: Sequence[Any],
        enemies: Sequence[Any],
        actions: Sequence[int],
    ) -> None:
        for agent_id, agent in enumerate(agents):
            if not agent.is_alive:
                continue
            action = actions[agent_id]
            if self.auto_aim:
                self._apply_auto_aim(world, agent, enemies, action)
            else:
                self._apply_manual(world, agent, action)

    @staticmethod
    def _set_chassis(agent, action: int) -> None:
        forward, backward, left, right = _CHASSIS_COMMANDS[action]
        agent.set_movement(forward=forward, backward=backward)
        agent.set_turning(left=left, right=right)

    def _apply_auto_aim(
        self,
        world: BattleWorld,
        agent,
        enemies: Sequence[Any],
        action: int,
    ) -> None:
        self._set_chassis(agent, action if 0 <= action < 9 else 0)
        if (
            action == 9
            and agent.can_fire()
            and self._has_fire_target(world, agent, enemies)
        ):
            world.fire_weapon(agent)

        closest = min(
            (
                enemy
                for enemy in enemies
                if enemy.is_alive and world.is_visible(agent, enemy)
            ),
            key=lambda enemy: _distance(agent, enemy),
            default=None,
        )
        agent.turret_target_angle = (
            agent.direction_angle if closest is None else _target_angle(agent, closest)
        )

        if (
            self.auto_fire_when_ready
            and action < 9
            and agent.can_fire()
            and self._has_fire_target(world, agent, enemies)
        ):
            world.fire_weapon(agent)

    def _apply_manual(self, world: BattleWorld, agent, action: int) -> None:
        if action < 27:
            self._set_chassis(agent, action % 9)
            turret_action = action // 9
            if turret_action == 1:
                agent.turret_target_angle = agent.turret_direction_angle - 15.0
            elif turret_action == 2:
                agent.turret_target_angle = agent.turret_direction_angle + 15.0
            else:
                agent.turret_target_angle = agent.turret_direction_angle
            return

        self._set_chassis(agent, 0)
        if action == 27:
            agent.turret_target_angle = agent.turret_direction_angle
            if agent.can_fire():
                world.fire_weapon(agent)

    def _has_fire_target(
        self,
        world: BattleWorld,
        agent,
        enemies: Sequence[Any],
    ) -> bool:
        for enemy in enemies:
            if not enemy.is_alive or not world.is_visible(agent, enemy):
                continue
            if _distance(agent, enemy) > agent.weapon_range():
                continue
            if not world.has_line_of_sight(agent, enemy):
                continue
            if self.fire_angle_tolerance is not None:
                angle_difference = abs(
                    agent.get_angle_difference(
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
