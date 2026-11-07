"""Recording-scene requirements: inert fire sites and low-damage water."""
import tempfile
import unittest
from pathlib import Path
import math

from engine.battle_world import BattleWorld
from engine.parameters import BULLET_SPEED, Team
from environment.scenario import build_ai_world
from training.run_game_ai import parse_args, resolve_settings


class FirefighterTests(unittest.TestCase):
    def test_three_second_spray_move_cycles_and_five_second_aim_choices(self):
        world = build_ai_world(ai_intelligence_level=5, normal_shell_only=False)
        ais = sorted((ai for ai in world.unit_manager.enemy_ais if ai.unit.team == Team.PLAYER),
                     key=lambda ai: ai.unit.id)
        director = ais[0].director
        starts = {ai.unit.id: ai.unit.position for ai in ais}
        first_shots, shot_times, moving_y = {}, {}, {}
        assignments, signs = {}, {ai.unit.id: set() for ai in ais}
        phase_shots = {}
        for _ in range(3300):
            world.step(.01)
            if director.started_at is None:
                self.assertFalse(any(e['kind'] == 'fire' for e in world.bullet_manager.events))
                continue
            phase = int((director.elapsed + 1e-9) // 3)
            targets = tuple(ai.target_unit.id for ai in ais)
            previous = assignments.setdefault(director.selection_round, targets)
            self.assertEqual(targets, previous, 'aim changes only every five seconds')
            self.assertEqual(director.selection_round, int((director.elapsed + 1e-9) // 5))
            for ai in ais:
                unit = ai.unit
                signs[unit.id].add(director.flank_signs[unit.id])
                self.assertGreaterEqual(unit.speed, 0, 'never reverse back and forth')
                self.assertEqual(ai.ai_state, 'reposition' if phase % 2 else 'spray_burst')
                self.assertGreaterEqual(unit.collision_box.left, unit.movement_bounds[0])
                self.assertLessEqual(unit.collision_box.right, unit.movement_bounds[2])
                self.assertGreaterEqual(unit.collision_box.top, unit.movement_bounds[1])
                self.assertLessEqual(unit.collision_box.bottom, unit.movement_bounds[3])
                if phase % 2:
                    moving_y.setdefault((unit.id, phase), []).append((unit.position, ai.target_unit.position))
                elif (director.elapsed + 1e-9) % 6 > .3:
                    self.assertEqual(unit.speed, 0, 'stop while spraying after gentle braking')
            for event in world.bullet_manager.events:
                if event['kind'] != 'fire':
                    continue
                uid = event['source_id']
                self.assertEqual(phase % 2, 0, 'no shots in the three-second movement phases')
                first_shots.setdefault(uid, (world.elapsed_time, world.get_unit(uid).position))
                shot_times.setdefault(uid, []).append(world.elapsed_time)
                phase_shots[(uid, phase)] = phase_shots.get((uid, phase), 0) + 1
        self.assertEqual(len(set(t for t, _ in first_shots.values())), 1)
        for uid, (time, position) in first_shots.items():
            self.assertTrue(1.5 < time < 3)
            self.assertGreater(math.dist(position, starts[uid]), 50)
            for phase in range(0, 10, 2):
                self.assertEqual(phase_shots[(uid, phase)], 30)
            intervals = [round(b - a, 6) for a, b in zip(shot_times[uid], shot_times[uid][1:])]
            self.assertEqual(set(intervals), {.1, 3.1})
            self.assertEqual(signs[uid], {-1, 1}, 'choose both left and right flanks')
            displacements = []
            for (vehicle, _), values in moving_y.items():
                if vehicle != uid:
                    continue
                start, target = values[0]
                end, _ = values[-1]
                dx, dy = target[0] - start[0], target[1] - start[1]
                displacements.append((dx * (end[1] - start[1]) - dy * (end[0] - start[0])) / math.hypot(dx, dy))
            self.assertTrue(any(v < -2 for v in displacements) and any(v > 2 for v in displacements),
                            'movement actually changes left/right direction')
        self.assertEqual(assignments[0], (100, 101, 102))
        self.assertTrue(any(a != b for before, after in zip(assignments.values(), list(assignments.values())[1:])
                            for a, b in zip(before, after)))
        self.assertEqual(world.unit_manager.unit_collision_count, 0)

    def test_old_arena_is_a_physical_limit_even_with_external_movement(self):
        world = build_ai_world(ai_intelligence_level=5, normal_shell_only=False)
        unit = world.get_unit(2)
        unit.position = (unit.movement_bounds[2] - 10, (unit.movement_bounds[1] + unit.movement_bounds[3]) / 2)
        unit.direction_angle = 90
        unit.speed = unit.max_speed
        unit.velocity = (unit.max_speed, 0)
        unit.acceleration = 0
        for _ in range(100):
            unit.update(.01, world.unit_manager, world.bullet_manager, world.game_map)
            self.assertLessEqual(unit.position[0] + unit.collision_size[0] / 2, unit.movement_bounds[2])
        self.assertGreater(world.game_map.width * world.game_map.tile_size, 960)
        self.assertTrue(all(site.position[0] > 960 for site in world.iter_units() if site.team == Team.ENEMY))

    def test_target_randomness_is_seeded_and_repeatable(self):
        def selections(seed):
            world = build_ai_world(seed=seed, ai_intelligence_level=5, normal_shell_only=False)
            ais = [ai for ai in world.unit_manager.enemy_ais if ai.unit.team == Team.PLAYER]
            director = ais[0].director
            result = []
            previous = -1
            for _ in range(1200):
                world.step(.01)
                if director.started_at is not None and director.selection_round != previous:
                    previous = director.selection_round
                    result.append(tuple(ai.target_unit.id for ai in ais))
            return result
        first = selections(42)
        self.assertEqual(first, selections(42))
        self.assertNotEqual(first, selections(7))

    def test_fire_site_aim_stays_exact_at_every_ai_level(self):
        for level in (1, 5, 9):
            world = build_ai_world(ai_intelligence_level=level, normal_shell_only=False)
            ai = next(ai for ai in world.unit_manager.enemy_ais if ai.unit.id == 2)
            target = world.get_unit(101)
            for frame in range(360):
                ai._frame_count = frame
                ai._aim_at_target(target)
                ai.unit._update_turret_direction(.01)
                expected = (math.degrees(math.atan2(target.position[1] - ai.unit.position[1],
                                                   target.position[0] - ai.unit.position[0])) + 90) % 360
                self.assertAlmostEqual(ai.unit.turret_target_angle, expected)
            self.assertAlmostEqual(ai.unit.turret_direction_angle, expected)

    def test_fire_site_lock_prevents_combat_retargeting(self):
        world = build_ai_world(ai_intelligence_level=5, normal_shell_only=False)
        ai = next(ai for ai in world.unit_manager.enemy_ais if ai.unit.id == 2)
        sites = [unit for unit in world.iter_units() if unit.team == Team.ENEMY]
        target = world.get_unit(101)
        ai.unit.position = (480, 320)
        ai.target_unit = target
        ai._ineffective_target_id = target.id
        ai._avoid_target_until_frame = 1000
        self.assertIs(ai._select_target(sites), target)
        target.is_alive = False
        self.assertIsNot(ai._select_target([unit for unit in sites if unit.is_alive]), target)

    def test_fire_sites_reject_ai_and_external_control(self):
        world = build_ai_world(ai_intelligence_level=9, normal_shell_only=False)
        sites = [unit for unit in world.iter_units() if unit.team == Team.ENEMY]
        poses = [(unit.position, unit.direction_angle, unit.turret_direction_angle) for unit in sites]
        for site in sites:
            self.assertEqual(site.health, 999999)
            self.assertFalse(world.set_unit_movement(site.id, forward=True))
            self.assertFalse(world.set_unit_turning(site.id, right=True))
            self.assertFalse(world.set_unit_turret_target_position(site.id, (200, 200)))
            self.assertFalse(world.set_unit_turret_target_angle(site.id, 123))
            self.assertIsNone(world.set_unit_fire(site.id))
            # Also reject commands written directly by a controller.
            site.speed = site.acceleration = site.angular_speed = 100
            site.velocity = (100, 100)
            site.turret_target_angle = 180
        world.step(.03)
        self.assertEqual(poses, [(u.position, u.direction_angle, u.turret_direction_angle) for u in sites])
        for site in sites:
            self.assertEqual((site.speed, site.angular_speed, site.velocity), (0, 0, (0, 0)))
            site.usingAI = False
            self.assertIsNone(site.fire())
        # The static constraint survives a state round trip.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.json'
            world.save_game_state(path)
            restored = BattleWorld().load_game_state(path)
            self.assertTrue(all(u.is_static for u in restored.iter_units() if u.team == Team.ENEMY))

    def test_water_parameters_and_twenty_second_recording(self):
        world = build_ai_world(ai_intelligence_level=5, normal_shell_only=False)
        allies = [unit for unit in world.iter_units() if unit.team == Team.PLAYER]
        sites = [unit for unit in world.iter_units() if unit.team == Team.ENEMY]
        initial = [(unit.position, unit.direction_angle, unit.turret_direction_angle) for unit in sites]
        for ally in allies:
            self.assertEqual(ally.ammunition_types, ['normal_shell'])
            self.assertEqual(ally.fire_cooldown_duration(), .1)
            water = ally.fire()
            self.assertEqual(water.speed, BULLET_SPEED * .45)
            self.assertEqual(water._calculate_damage(sites[0]), 1)
            self.assertFalse(water.is_explosive)
            self.assertEqual(ally.weapon_range(), 2430)
            self.assertEqual(ally.turret_angular_speed, 75)
            for _ in range(9):
                ally._update_fire_cooldown(.01)
            self.assertFalse(ally.can_fire())
            ally._update_fire_cooldown(.01)
            self.assertTrue(ally.can_fire())
        shots = {unit.id: 0 for unit in world.iter_units()}
        settings = resolve_settings(parse_args(['--render', '3D']))
        self.assertEqual(settings.dt, .01)
        for _ in range(2100):
            world.step(settings.dt)
            for event in world.bullet_manager.events:
                if event['kind'] == 'fire':
                    shots[event['source_id']] += 1
        self.assertEqual(initial, [(u.position, u.direction_angle, u.turret_direction_angle) for u in sites])
        self.assertTrue(all(u.is_alive and u.health > 990000 for u in sites))
        self.assertTrue(all(shots[u.id] == 0 for u in sites))
        self.assertTrue(all(90 <= shots[u.id] <= 110 for u in allies))
        self.assertGreater(sum(u.damage_received for u in sites), 0)
