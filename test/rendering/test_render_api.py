"""Cross-layer rendering and extensible map deployment regressions."""
import ast
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from environment import JackalEnv
from environment.presentation.adapter import build_world
from render.api import apply_map, create_inset_renderer, create_renderer, create_viewer, list_maps, list_models, load_map
from render.render3D.maps.registry import CATALOG, list_maps as read_catalog, resolve_scene
from render.transport.server import handler_for
from render.transport.stream import SnapshotStream

ROOT = Path(__file__).resolve().parents[2]


class RenderAPITests(unittest.TestCase):
    def test_unknown_model_is_rejected_before_opening_a_browser(self):
        self.assertEqual(list_models(), ('m1a1',))
        data = json.loads((CATALOG / 'border-verdant.json').read_text())
        data['unit_models']['tank'] = 'missing-model'
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'broken.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'missing-model'):
                read_catalog(Path(directory))

    def test_model_catalog_requires_an_existing_adapter(self):
        from render.render3D.models.registry import list_models as read_models
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'broken.json').write_text('{"id":"new-tank","adapter":"absent"}')
            with self.assertRaisesRegex(ValueError, 'adapter does not exist'):
                read_models(Path(directory))

    def test_draw_present_and_rgb_access_obey_the_session_lifecycle(self):
        world = build_world()
        frame = world.render_snapshot()
        view = create_renderer('RGB_ARRAY')
        try:
            for action in (lambda: view.draw(frame), view.present, view.rgb_array):
                with self.assertRaisesRegex(RuntimeError, 'initialize'):
                    action()
            view.initialize(world.render_manifest(.03))
            view.draw(frame)
            view.close()
            for action in (lambda: view.draw(frame), view.present, view.rgb_array):
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    action()
        finally:
            view.close()

    def test_environment_releases_renderers_even_if_video_finalization_fails(self):
        env = JackalEnv()
        env.renderer, env._video_renderer, env.video_writer = Mock(), Mock(), Mock()
        writer = env.video_writer
        writer.release.side_effect = RuntimeError('encoder failed')
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            env.close()
        env.renderer.close.assert_called_once()
        env._video_renderer.close.assert_called_once()
        env.close()
        writer.release.assert_called_once()
        for action in (env.reset, lambda: env.step([0]), env.render):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                action()

    def test_deployment_uses_actual_positions_and_preserves_world(self):
        world = build_world()
        original = world.render_snapshot()
        scene = apply_map(world)
        tank = next(u for u in scene["initial_units"] if u["id"] == 2)
        self.assertEqual((tank["x"], tank["y"], tank["z"]), (-65., 0., 0.))
        self.assertEqual(tank["model"], "m1a1")
        self.assertEqual(tank["turret_yaw"], math.pi - math.radians(original.units[1].turret_heading))
        self.assertEqual(world.render_snapshot(), original)

    def test_second_map_is_registered_loaded_and_scaled(self):
        self.assertEqual({s.id for s in list_maps()}, {"border-verdant", "border-small-verdant"})
        world = load_map("border-small-verdant")
        self.assertEqual(world.render_manifest(.03).map.size, (640, 512))
        scene = apply_map(world)
        self.assertEqual(scene["id"], "border-small-verdant")
        self.assertEqual(len(scene["initial_units"]), 4)
        self.assertEqual([(u["x"], u["z"]) for u in scene["initial_units"]],
                         [(-40., -16.), (-40., 16.), (40., -16.), (40., 16.)])

    def test_map_validation_precedes_browser_creation(self):
        world = build_world(ROOT / "engine/scenarios/game_states/valley_map_3t2a_vs_3t2a.json")
        with patch("render.render3D.bridge.BrowserRenderer") as browser:
            with self.assertRaisesRegex(ValueError, "--render 2D"):
                create_viewer(world, "3D")
            browser.assert_not_called()

    def test_duplicate_layout_requires_explicit_scene_selection(self):
        world = build_world()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            data = json.loads((CATALOG / "border-verdant.json").read_text())
            (path / "first.json").write_text(json.dumps(data))
            data["id"] = "another-theme"
            (path / "second.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "scene_id"):
                resolve_scene(world.render_manifest(.03), catalog=path)
            self.assertEqual(resolve_scene(world.render_manifest(.03), "another-theme", catalog=path).id,
                             "another-theme")

    def test_invalid_catalog_reports_the_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text('{"id":"broken","layout":{"tile_size":64,"tiles":[]}}')
            with self.assertRaisesRegex(ValueError, "broken.json"):
                read_catalog(Path(directory))

    def test_2d_offscreen_facade_draws_without_mutating_physics(self):
        world = build_world()
        original = world.render_snapshot()
        with create_viewer(world, "rgb_array", width=480, height=320) as view:
            self.assertTrue(view.poll())
            view.render(original)
            self.assertEqual(view.rgb_array().shape, (320, 480, 3))
            self.assertEqual(world.render_snapshot(), original)
        self.assertFalse(view.poll())
        view.close()  # Cleanup is idempotent.

    def test_none_backend_remains_headless(self):
        self.assertIsNone(create_renderer(None))

    def test_transport_rejects_nonobject_payload_and_unsupported_restart(self):
        stream = SnapshotStream(show_inset=False)
        try:
            for path, payload, status in (("/api/window", [], 400), ("/api/restart", {}, 409)):
                data = json.dumps(payload).encode()
                handler = object.__new__(handler_for(stream))
                handler.path = path
                handler.headers = {"Content-Length": str(len(data))}
                handler.rfile = io.BytesIO(data)
                response = {}
                handler.respond = lambda body, status=200: response.update(body=body, status=status)
                handler.do_POST()
                self.assertEqual(response["status"], status)
        finally:
            stream.close()

    def test_quit_event_is_handled_by_the_unified_2d_api(self):
        import pygame
        with create_viewer(build_world(), "2D") as view:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
            self.assertFalse(view.poll())

    def test_stream_limit_is_relative_to_restored_episode_tick(self):
        world = build_world()
        for _ in range(4):
            world.step(.03)
        stream = SnapshotStream(show_inset=False, max_steps=2)
        try:
            stream.initialize(world.render_manifest(.03))
            stream.publish(world.render_snapshot())
            self.assertFalse(stream.latest["result"]["finished"])
            world.step(.03); world.step(.03)
            stream.publish(world.render_snapshot())
            self.assertEqual(stream.latest["result"]["reason"], "timeout")
            self.assertEqual(stream.info["max_steps"], 6)
        finally:
            stream.close()

    def test_production_callers_use_the_public_api(self):
        paths = list((ROOT / "environment").rglob("*.py")) + list((ROOT / "training").rglob("*.py"))
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                for module in modules:
                    if module == "render" or module.startswith("render."):
                        self.assertEqual(module, "render.api", str(path))

    def test_renderer_does_not_depend_on_environment_or_training(self):
        for path in (ROOT / "render").rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                modules = ([n.name for n in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                self.assertFalse(any(m.split('.')[0] in {"environment", "training"} for m in modules), str(path))

    def test_inset_options_do_not_leak_to_other_renderers(self):
        import render.render2D.options as options
        before = options.DRAW_SIGHT_RANGE, options.DRAW_HEALTH_BAR
        inset = create_inset_renderer(960, 640)
        try:
            self.assertEqual((options.DRAW_SIGHT_RANGE, options.DRAW_HEALTH_BAR), before)
        finally:
            inset.close()


class ActionValidationTests(unittest.TestCase):
    def test_invalid_batch_does_not_change_any_command_tick_or_step(self):
        env = JackalEnv(n_agents=3, n_enemies=3)
        try:
            env.reset(seed=42)
            def state():
                return [u.to_game_state() for u in env.world.iter_units()], env.world.render_snapshot()
            before = state()
            for actions in ([1], [1, 2, 10], [1, 2, -1], [1, 2, 2.5], [1, 2, True]):
                with self.assertRaises(ValueError):
                    env.step(actions)
                self.assertEqual(env.steps, 0)
                self.assertEqual(state(), before)
        finally:
            env.close()

    def test_nonfinite_timestep_and_invalid_horizon_are_rejected(self):
        for value in (float('nan'), float('inf'), 0, -1):
            with self.assertRaises(ValueError):
                JackalEnv(fixed_delta_time=value)
        for value in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                JackalEnv(max_steps=value)

    def test_unit_save_writes_the_canonical_record_and_can_be_read(self):
        unit = build_world().get_unit(2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/unit.json"
            self.assertTrue(unit.save_to_file(path))
            self.assertEqual(json.loads(path.read_text()), unit.to_game_state())
            self.assertEqual(unit.load_from_file(path), unit.to_game_state())


if __name__ == "__main__":
    unittest.main()
