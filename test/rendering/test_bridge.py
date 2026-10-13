import os
import io
import threading
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from environment.presentation.adapter import (STATE_FILE, build_world, export_frame, manifest, result_for,
                     step_world, unit_parameters, world_position, world_yaw)
from engine.battle_world import BattleWorld
from engine.projectiles.normal_shell.normal_shell import NormalShell
from engine.parameters import Team


class BridgeTests(unittest.TestCase):
    def test_render_api_and_browser_preserve_the_environment_result(self):
        from render.api import RenderSession
        from render.render3D.bridge import BrowserRenderer
        from render.transport.stream import SnapshotStream
        world = build_world()
        manifest = world.render_manifest(.03)
        renderer = object.__new__(BrowserRenderer)
        renderer.hub = SnapshotStream(show_inset=False, max_steps=6000)
        renderer.vite = Mock()
        renderer.vite.poll.return_value = None
        renderer.open_browser = False
        renderer.follow_missing = False
        view = object.__new__(RenderSession)
        view.backend, view.closed, view.manifest = renderer, False, manifest
        view._deployed, view.scene_id = False, None
        result = {'finished': True, 'winner': 'DRAW', 'reason': 'environment_limit',
                  'alive': {'PLAYER': 3, 'ENEMY': 3}}
        try:
            renderer.hub.initialize(manifest)
            view.render(world.render_snapshot(), result)
            self.assertEqual(renderer.hub.latest['result'], result)
            self.assertTrue(renderer.hub.latest['paused'])
        finally:
            renderer.hub.close()

    def test_setup_failure_releases_socket_and_log_before_process_start(self):
        from render.render3D.bridge import BrowserRenderer
        log = io.StringIO()
        with patch('render.transport.server.LocalDemoServer') as server, \
                patch('tempfile.TemporaryFile', return_value=log), \
                patch('render.render3D.bridge.subprocess.Popen', side_effect=OSError('launch failed')):
            with self.assertRaisesRegex(OSError, 'launch failed'):
                BrowserRenderer(view_port=1234)
            server.return_value.server_close.assert_called_once()
            self.assertTrue(log.closed)
        with patch('render.transport.server.LocalDemoServer') as server, \
                patch('render.render3D.bridge.available_view_port', side_effect=OSError('no port')):
            with self.assertRaisesRegex(OSError, 'no port'):
                BrowserRenderer()
            server.return_value.server_close.assert_called_once()

    def test_process_monitor_preserves_failure_log_and_wakes_the_stream(self):
        from render.render3D.bridge import BrowserRenderer
        from render.transport.stream import SnapshotStream
        renderer = object.__new__(BrowserRenderer)
        renderer.closed = False
        renderer.hub = SnapshotStream(show_inset=False)
        renderer.log = io.StringIO('Vite failed to bind port')
        renderer.log_lock = threading.Lock()
        renderer.vite = Mock()
        renderer.vite.poll.return_value = 1
        renderer.window_watch_stop = Mock()
        renderer.window_watch_stop.wait.return_value = False
        try:
            renderer._watch_service()
            with self.assertRaisesRegex(RuntimeError, 'Vite failed to bind port'):
                renderer.poll()
        finally:
            renderer.hub.close()

    def test_missing_follow_id_selects_an_ally_even_if_enemy_is_first(self):
        from render.render3D.bridge import BrowserRenderer
        from render.transport.stream import SnapshotStream
        world = build_world()
        world.unit_manager.units[1].id = 20
        world.unit_manager.units[:] = world.unit_manager.units[3:] + world.unit_manager.units[:3]
        renderer = object.__new__(BrowserRenderer)
        renderer.hub = SnapshotStream(show_inset=False)
        renderer.vite = Mock()
        renderer.vite.poll.return_value = None
        renderer.closed = False
        renderer.open_browser = False
        renderer.view_port = 1234
        try:
            renderer.initialize(world.render_manifest(.03))
            renderer.render(world.render_snapshot())
            self.assertEqual(renderer.hub.info['follow_id'], 1)
        finally:
            renderer.hub.close()

    def test_original_training_stats_are_preserved_when_attaching_ai(self):
        source = BattleWorld()
        source.load_game_state(STATE_FILE)
        adapted = build_world()
        self.assertEqual(len(adapted.unit_manager.enemy_ais), 6)
        fields = ("position", "size", "collision_size", "max_speed", "max_acceleration",
                  "min_acceleration", "max_angular_speed", "turret_angular_speed",
                  "max_health", "sight_range", "communication_range")
        for original, actual in zip(source.iter_units(), adapted.iter_units()):
            for field in fields:
                self.assertEqual(getattr(original, field), getattr(actual, field), (actual.id, field))
            self.assertEqual(original.get_weapon_spec("normal_shell"),
                             actual.get_weapon_spec("normal_shell"))
            self.assertEqual(actual.ammunition_types, ["normal_shell"])
            self.assertEqual(original.fire_cooldown_duration(), actual.fire_cooldown_duration())
        config = manifest(adapted)
        self.assertEqual(config["map"]["size"], [960, 640])
        self.assertEqual(config["follow_id"], 2)
        self.assertEqual([u["model"] for u in config["units"]], ["m1a1"] * 6)
        self.assertEqual(world_position(220, 320), (-65, 0))
        self.assertAlmostEqual(world_yaw(90), 1.5707963267948966)

    def test_observing_and_exporting_does_not_change_any_battle_step(self):
        plain, observed = build_world(observe=False), build_world()
        for tick in range(701):
            left, right = export_frame(plain, 700), export_frame(observed, 700)
            self.assertEqual(left["units"], right["units"], tick)
            self.assertEqual(left["result"], right["result"], tick)
            strip_ids = lambda bullets: [{k: v for k, v in b.items() if k != "id"} for b in bullets]
            self.assertEqual(strip_ids(left["bullets"]), strip_ids(right["bullets"]), tick)
            # Reading/exporting the same frame twice must have no side effects.
            self.assertEqual(right, export_frame(observed, 700))
            if right["result"]["finished"]:
                break
            step_world(plain, .03)
            step_world(observed, .03)
        self.assertTrue(right["result"]["finished"])
        self.assertEqual(right["result"]["reason"], "timeout")
        self.assertEqual(tick, 700)
        self.assertEqual(right["result"]["winner"], "DRAW")

    def test_firing_intervals_respect_each_units_effective_cooldown(self):
        world = build_world()
        shots = {unit.id: [] for unit in world.iter_units()}
        cooldowns = {unit.id: unit.fire_cooldown_duration() for unit in world.iter_units()}
        for _ in range(700):
            step_world(world, .03)
            frame = export_frame(world, 700)
            for event in frame["events"]:
                if event["kind"] == "fire":
                    shots[event["source_id"]].append(frame["time"])
        for unit_id, times in shots.items():
            self.assertGreater(len(times), 1, unit_id)
            intervals = [b - a for a, b in zip(times, times[1:])]
            self.assertGreaterEqual(min(intervals), cooldowns[unit_id] - 1e-8, unit_id)

    def test_exported_cooldown_is_seconds_and_matches_the_actual_shot(self):
        unit = build_world().unit_manager.units[0]
        # The new ProjectileSpec intentionally has no absolute cooldown field.
        self.assertFalse(hasattr(unit.get_weapon_spec(), "cooldown"))
        for base, rate, override, expected in (
            (2.0, .5, None, 1.0),
            (2.0, 2.0, None, 4.0),
            (2.0, 0.0, None, 0.0),
            (2.0, 2.0, .35, .35),
            (2.0, 2.0, 0.0, 0.0),
            (1.2, .8, 1.5, 1.5),
        ):
            with self.subTest(base=base, rate=rate, override=override), \
                    patch("engine.units.base_unit.BULLET_COOLDOWN", base):
                unit.configure_weapon(fire_cooldown=override,
                    projectile_overrides={"normal_shell": {"cooldown_rate": rate}})
                unit.fire_cooldown = 0.0
                self.assertEqual(unit_parameters(unit)["cooldown"], expected)
                bullet = unit.fire()
                self.assertIsNotNone(bullet)
                self.assertEqual(bullet.cooldown, expected)
                self.assertEqual(unit.fire_cooldown, expected)

    def test_extended_evaluation_can_finish_beyond_the_training_horizon(self):
        world = build_world()
        for _ in range(6000):
            step_world(world, .03)
            result = result_for(world, 6000)
            if world.tick == 700:
                self.assertFalse(result["finished"])
            if result["finished"]:
                break
        self.assertGreater(world.tick, 700)
        self.assertLess(world.tick, 6000)
        self.assertIn(result["reason"], ("elimination", "mutual_elimination"))
        self.assertIn(0, result["alive"].values())

    def test_normal_shell_contact_is_single_target_and_does_not_ricochet(self):
        world = build_world()
        shooter = world.unit_manager.units[0]
        target, neighbour = world.unit_manager.units[3:5]
        target.position = (300, 320)
        neighbour.position = (320, 320)
        bullet = NormalShell("test-contact", shooter, Team.PLAYER, position=target.position)
        self.assertEqual(bullet.speed, 300)
        self.assertEqual(bullet.size, (6., 6.))
        self.assertEqual(bullet.max_lifetime, 1.2)
        self.assertFalse(bullet.is_explosive)
        bullet._handle_unit_collision(target, world.unit_manager)
        self.assertFalse(bullet.is_active)
        self.assertEqual(target.health, 92)
        self.assertEqual(neighbour.health, 100)
        self.assertEqual(bullet.collided_objects, [target])


if __name__ == "__main__":
    unittest.main()
