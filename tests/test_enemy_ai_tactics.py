"""Focused checks for communication-limited combat cooperation and maneuvers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from game.BattleWorld import BattleWorld
from game.Map.GameMap import GameMap
from game.Parameter import Team
from game.Unit.EnemyAI import EnemyAI
from game.Unit.UnitManager import UnitManager


class EnemyAITacticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = BattleWorld(
            GameMap(["x" * 15] + ["x" + "o" * 13 + "x"] * 8 + ["x" * 15]),
            unit_manager=UnitManager(enable_unit_collision=True, auto_communicate=False,
                                     use_tear_drop_vision=False),
        )

    def add(self, unit_id, position, *, team=Team.PLAYER, level=5, using_ai=True):
        return self.world.create_unit(
            "tank", team, position, unit_id=unit_id, using_ai=using_ai,
            ai_intelligence_level=level, sight_range=400.0, communication_range=400.0,
            initial_heading=90.0 if team == Team.PLAYER else 270.0,
        )

    def ai(self, unit_id) -> EnemyAI:
        return next(ai for ai in self.world.unit_manager.enemy_ais if ai.unit.id == unit_id)

    def prepare(self) -> None:
        self.world.refresh_vision()
        for ai in self.world.unit_manager.enemy_ais:
            ai._refresh_nearby_unit_cache()
            ai._cached_combat_allies = tuple(
                unit for unit in ai._nearby_units_for_update()
                if unit is not ai.unit and unit.team == ai.unit.team and unit.is_alive
            )

    def test_nearby_level_five_ais_share_wounded_target_and_lock_it(self):
        self.add(1, (300.0, 220.0))
        self.add(2, (300.0, 320.0))
        healthy = self.add(100, (490.0, 220.0), team=Team.ENEMY, using_ai=False)
        wounded = self.add(101, (500.0, 320.0), team=Team.ENEMY, using_ai=False)
        wounded.health = 30.0
        self.prepare()
        enemies = [healthy, wounded]
        self.assertIs(self.ai(1)._select_target(enemies), wounded)
        self.assertIs(self.ai(2)._select_target(enemies), wounded)
        healthy.health = 1.0
        self.assertIs(self.ai(1)._select_target(enemies), wounded)
        wounded.is_alive = False
        self.assertIs(self.ai(2)._select_target([healthy]), healthy)

    def test_focus_orders_do_not_reveal_unknown_enemies(self):
        self.add(1, (300.0, 220.0))
        self.add(2, (300.0, 320.0))
        visible = self.add(100, (490.0, 220.0), team=Team.ENEMY, using_ai=False)
        hidden = self.add(101, (750.0, 500.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        self.ai(1)._focus_target_id = hidden.id
        self.ai(1)._focus_lock_until_frame = 100
        self.assertIs(self.ai(1)._select_target([visible]), visible)

    def test_no_sharing_outside_communication_range_or_at_low_level(self):
        one = self.add(1, (300.0, 220.0), level=3)
        two = self.add(2, (300.0, 320.0))
        enemy = self.add(100, (490.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        self.ai(1)._select_target([enemy])
        self.assertIsNone(self.ai(2)._focus_target_id)
        self.ai(1).intelligence_level = 5
        one.communication_range = two.communication_range = 20.0
        self.ai(1)._select_target([enemy])
        self.assertIsNone(self.ai(2)._focus_target_id)

    def test_unreachable_focus_does_not_prevent_local_fire(self):
        self.add(1, (280.0, 220.0))
        self.add(2, (450.0, 320.0))
        local = self.add(100, (460.0, 220.0), team=Team.ENEMY, using_ai=False)
        distant = self.add(101, (710.0, 320.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._focus_target_id = distant.id
        ai._focus_lock_until_frame = 100
        self.assertIs(ai._select_target([local, distant]), local)

    def test_effective_uncrowded_fire_still_holds_position(self):
        self.add(1, (300.0, 220.0))
        enemy = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._update_combat_feedback(enemy)
        ai._plan_combat_movement(enemy)
        ai._engage_target(enemy)
        self.assertIsNone(ai._combat_destination)
        self.assertEqual(ai.ai_state, "engage")
        self.assertFalse(ai._movement_requested)

    def test_sustained_ineffective_fire_plans_safe_bounded_move(self):
        self.add(1, (300.0, 220.0))
        enemy = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._update_combat_feedback(enemy)
        ai._frame_count = 250
        ai._shots_since_progress = 4
        with patch.object(ai, "_build_grid_path", side_effect=AssertionError("no extra A*")):
            ai._plan_combat_movement(enemy)
        destination = ai._combat_destination
        self.assertIsNotNone(destination)
        assert destination is not None
        self.assertTrue(self.world.can_move_unit(ai.unit.id, destination))
        self.assertEqual(ai._combat_move_reason, "reposition_stalled")
        ai._frame_count = ai._combat_move_until_frame
        ai._engage_target(enemy)
        self.assertIsNone(ai._combat_destination)
        ai._plan_combat_movement(enemy)
        self.assertIsNone(ai._combat_destination)

    def test_target_damage_resets_stall_detection(self):
        self.add(1, (300.0, 220.0))
        enemy = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._update_combat_feedback(enemy)
        ai._frame_count = 250
        ai._shots_since_progress = 4
        enemy.health -= 5.0
        ai._update_combat_feedback(enemy)
        ai._plan_combat_movement(enemy)
        self.assertEqual(ai._shots_since_progress, 0)
        self.assertIsNone(ai._combat_destination)

    def test_only_committed_shots_count_toward_ineffective_fire(self):
        self.add(1, (300.0, 220.0))
        enemy = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._update_combat_feedback(enemy)
        ai._frame_count = 1000
        ai._plan_combat_movement(enemy)
        self.assertIsNone(ai._combat_destination)
        ai._fire_requested = True
        with patch.object(self.world.bullet_manager, "fire", return_value=None):
            self.assertFalse(ai.commit_fire())
        self.assertEqual(ai._shots_since_progress, 0)
        ai._fire_requested = True
        with patch.object(self.world.bullet_manager, "fire", return_value=object()):
            self.assertTrue(ai.commit_fire())
        self.assertEqual(ai._shots_since_progress, 1)

    def test_stalled_target_can_be_replaced_by_another_reachable_enemy(self):
        self.add(1, (300.0, 220.0))
        self.add(2, (300.0, 320.0))
        stalled = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        alternative = self.add(101, (500.0, 320.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._focus_target_id = stalled.id
        ai._focus_lock_until_frame = 1000
        ai._update_combat_feedback(stalled)
        ai._frame_count = 250
        ai._shots_since_progress = 4
        ai._plan_combat_movement(stalled)
        self.assertIs(ai._select_target([stalled, alternative]), alternative)

    def test_crowded_pair_only_one_unit_yields(self):
        self.add(1, (300.0, 220.0))
        self.add(2, (300.0, 250.0))
        enemy = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        for ai in self.world.unit_manager.enemy_ais:
            ai._plan_combat_movement(enemy)
        self.assertIsNone(self.ai(1)._combat_destination)
        self.assertIsNotNone(self.ai(2)._combat_destination)
        self.assertEqual(self.ai(2)._combat_move_reason, "spread_formation")

    def test_combat_spacing_is_mirrored_not_biased_to_one_side(self):
        left_target = self.add(1, (360.0, 220.0))
        self.add(2, (360.0, 250.0))
        right_target = self.add(100, (600.0, 220.0), team=Team.ENEMY)
        self.add(101, (600.0, 250.0), team=Team.ENEMY)
        self.prepare()
        self.ai(2)._plan_combat_movement(right_target)
        self.ai(101)._plan_combat_movement(left_target)
        left = self.ai(2)._combat_destination
        right = self.ai(101)._combat_destination
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        assert left is not None and right is not None
        self.assertAlmostEqual(left[0] + right[0], 960.0)
        self.assertAlmostEqual(left[1], right[1])

    def test_recent_focus_damage_regroups_before_critical_health(self):
        unit = self.add(1, (350.0, 220.0))
        ally = self.add(2, (160.0, 220.0))
        enemies = [self.add(100 + i, (550.0, 200.0 + i * 60.0),
                            team=Team.ENEMY, using_ai=False) for i in range(2)]
        self.prepare()
        unit.health = 65.0
        ai = self.ai(1)
        ai._recent_damage_fraction = 0.12
        self.assertIs(ai._select_regroup_ally(enemies), ally)

    def test_combat_update_reuses_one_nearby_spatial_query(self):
        self.add(1, (300.0, 220.0))
        self.add(2, (300.0, 320.0))
        self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        manager = self.world.unit_manager
        with patch.object(manager, "get_units_in_radius", wraps=manager.get_units_in_radius) as query:
            self.ai(1).update()
        self.assertEqual(query.call_count, 1)

    def test_friend_blocking_fire_lane_prompts_short_reposition(self):
        self.add(1, (300.0, 220.0))
        self.add(2, (350.0, 220.0))
        enemy = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._plan_combat_movement(enemy)
        self.assertIsNotNone(ai._combat_destination)
        self.assertEqual(ai._combat_move_reason, "clear_fire_lane")

    def test_low_level_and_immobile_units_do_not_maneuver(self):
        unit = self.add(1, (300.0, 220.0), level=3)
        enemy = self.add(100, (500.0, 220.0), team=Team.ENEMY, using_ai=False)
        self.prepare()
        ai = self.ai(1)
        ai._frame_count = 1000
        ai._shots_since_progress = 10
        ai._plan_combat_movement(enemy)
        self.assertIsNone(ai._combat_destination)
        ai.intelligence_level = 9
        unit.max_speed = 0.0
        ai._plan_combat_movement(enemy)
        self.assertIsNone(ai._combat_destination)


if __name__ == "__main__":
    unittest.main()
