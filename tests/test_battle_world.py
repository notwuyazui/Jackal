import unittest
from typing import Any, cast

from game.core import BattleWorld


class _RecordingMap:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def update(self, delta_time: float) -> None:
        self.calls.append(f"map:{delta_time}")


class _RecordingUnit:
    is_alive = True

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def _update_vision(self, unit_manager, bullet_manager, game_map) -> None:
        self.calls.append("vision")


class _RecordingUnitManager:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.units = [_RecordingUnit(calls)]

    def add_unit(self, unit, bullet_manager, game_map) -> None:
        self.calls.append("add_unit")

    def update(self, delta_time, unit_manager, bullet_manager, game_map) -> None:
        self.calls.append(f"units:{delta_time}")

    def clear(self) -> None:
        self.units.clear()


class _RecordingBulletManager:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.bullets = []

    def add_bullet(self, bullet) -> None:
        self.calls.append("add_bullet")

    def update(self, delta_time, unit_manager, game_map) -> None:
        self.calls.append(f"bullets:{delta_time}")

    def clear(self) -> None:
        self.bullets.clear()


class BattleWorldTests(unittest.TestCase):
    def test_step_has_one_authoritative_update_order(self) -> None:
        calls: list[str] = []
        world = BattleWorld(
            cast(Any, _RecordingMap(calls)),
            cast(Any, _RecordingUnitManager(calls)),
            cast(Any, _RecordingBulletManager(calls)),
        )

        world.step(0.25)

        self.assertEqual(
            calls,
            ["map:0.25", "units:0.25", "bullets:0.25", "vision"],
        )
        self.assertEqual(world.tick, 1)
        self.assertEqual(world.elapsed_time, 0.25)

    def test_non_positive_delta_time_is_rejected(self) -> None:
        calls: list[str] = []
        world = BattleWorld(cast(Any, _RecordingMap(calls)))

        with self.assertRaises(ValueError):
            world.step(0.0)


if __name__ == "__main__":
    unittest.main()
