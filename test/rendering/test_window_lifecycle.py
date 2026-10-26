"""Window closure releases frame waits and preserves reloads/other windows."""
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from environment.presentation.adapter import build_world
from environment.presentation.server import handler_for
from render.api import SnapshotStream, ViewerClosed
from render.render3D.launch import SafariWindow, BrowserProcessWindow, _windows_browser, open_cot_window
from render.api import RenderSession
from training.run_game_ai import parse_args, resolve_settings, run_battle


class WindowLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.hub = SnapshotStream(show_inset=False)

    def tearDown(self):
        self.hub.close()

    def request(self, payload):
        data = json.dumps(payload).encode()
        handler = object.__new__(handler_for(self.hub))
        handler.path = '/api/window'
        handler.headers = {'Content-Length': str(len(data))}
        handler.rfile = io.BytesIO(data)
        response = {}
        handler.respond = lambda body, status=200: response.update(payload=body, status=status)
        handler.do_POST()
        return response

    def test_close_unblocks_first_frame_before_ack_or_startup_timeout(self):
        world = build_world()
        self.hub.lockstep = True
        self.hub.startup_timeout = 120
        self.hub.initialize(world.render_manifest(.03))
        errors = []
        def publish():
            try:
                self.hub.publish(world.render_snapshot())
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=publish, daemon=True)
        worker.start()
        with self.hub.condition:
            ready = self.hub.condition.wait_for(lambda: self.hub.latest is not None, timeout=2)
        try:
            self.assertTrue(ready)
            self.hub.request_viewer_close()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ViewerClosed)
            self.assertEqual(world.tick, 0)
        finally:
            self.hub.close()
            worker.join(timeout=2)

    def test_window_close_and_reload_grace(self):
        with patch('render.transport.stream.threading.Timer') as timer:
            self.hub.window_event('open', 'page-one')
            self.hub.window_event('close', 'page-one')
            self.assertFalse(self.hub.viewer_closed)
            finish_close = timer.call_args.args[1]
            self.hub.window_event('open', 'reloaded-page')
            timer.return_value.cancel.assert_called_once()
            finish_close()  # Even an already executing callback from the old page is harmless.
            self.assertFalse(self.hub.viewer_closed)
            self.hub.window_event('close', 'reloaded-page')
            timer.call_args.args[1]()
            self.assertTrue(self.hub.viewer_closed)

    def test_old_page_cannot_close_current_window(self):
        self.hub.window_event('open', 'current-page')
        with patch('render.transport.stream.threading.Timer') as timer:
            self.hub.window_event('close', 'old-page')
            timer.assert_not_called()
        self.assertFalse(self.hub.viewer_closed)

    def test_late_open_from_departing_page_does_not_cancel_its_close(self):
        with patch('render.transport.stream.threading.Timer') as timer:
            self.hub.window_event('close', 'departing-page')
            self.hub.window_event('open', 'departing-page')
            timer.return_value.cancel.assert_not_called()
            timer.call_args.args[1]()
        self.assertTrue(self.hub.viewer_closed)

    def test_window_endpoint_authenticates_and_validates_requests(self):
        response = self.request({'view_token': 'old-token', 'action': 'close', 'client_id': 'page'})
        self.assertEqual(response['status'], 403)
        self.assertIsNone(self.hub.window_client)
        response = self.request({'view_token': self.hub.view_token, 'action': 'open', 'client_id': 'page'})
        self.assertEqual(response['status'], 200)
        self.assertEqual(self.hub.window_client, 'page')
        response = self.request({'view_token': self.hub.view_token, 'action': 'close', 'client_id': ''})
        self.assertEqual(response['status'], 400)
        self.assertFalse(self.hub.viewer_closed)

    def test_browser_poll_reflects_close_after_battle_finished(self):
        from render.render3D.bridge import BrowserRenderer
        renderer = object.__new__(BrowserRenderer)
        renderer.hub, renderer.closed = self.hub, False
        viewer = object.__new__(RenderSession)
        viewer.backend, viewer.closed = renderer, False
        self.assertTrue(viewer.poll())
        self.hub.request_viewer_close()
        self.assertFalse(viewer.poll())

    def test_closing_browser_during_http_response_does_not_report_an_error(self):
        handler = object.__new__(handler_for(self.hub))
        handler.send_response = handler.send_header = handler.end_headers = Mock()
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError()
        handler.respond({'ok': True})

    def test_test_entry_cleans_up_on_close_while_waiting_for_render(self):
        args = parse_args([])
        viewer = RenderSession('rgb_array')
        with patch('environment.evaluation.create_viewer', return_value=viewer), \
                patch.object(viewer, 'render', side_effect=ViewerClosed()), \
                patch.object(viewer, 'close', wraps=viewer.close) as close:
            self.assertIsNone(run_battle(args, resolve_settings(args)))
            close.assert_called_once()
        self.assertTrue(viewer.closed)

    def test_test_entry_cleans_up_when_result_window_closes(self):
        args = parse_args(['--max-steps', '1'])
        viewer = RenderSession('rgb_array')
        with patch('environment.evaluation.create_viewer', return_value=viewer), \
                patch.object(viewer, 'render'), \
                patch.object(viewer, 'poll', side_effect=[True, False]), \
                patch.object(viewer, 'pace', return_value=True), \
                patch.object(viewer, 'close', wraps=viewer.close) as close:
            result = run_battle(args, resolve_settings(args))
            close.assert_called_once()
        self.assertEqual(result['tick'], 1)
        self.assertTrue(viewer.closed)

    @patch('render.render3D.launch.sys.platform', 'darwin')
    @patch('render.render3D.launch.subprocess.run')
    def test_native_launcher_returns_the_new_window_id(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout='314\n')
        window = open_cot_window('http://127.0.0.1:1234/')
        self.assertEqual(window.identifier, 314)
        self.assertIn('make new document', run.call_args.args[0][2])
        self.assertIn('return battleWindowID', run.call_args.args[0][2])
        self.assertIn('"AXFullScreen"', run.call_args.args[0][2])

    @patch('render.render3D.launch.sys.platform', 'darwin')
    @patch('render.render3D.launch.subprocess.run')
    def test_safari_fullscreen_denial_keeps_the_exact_new_window_watchable(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout='314\t-1743\n')
        with patch('sys.stdout', new_callable=io.StringIO) as output:
            window = open_cot_window('http://127.0.0.1:1234/')
        self.assertIsInstance(window, SafariWindow)
        self.assertEqual(window.identifier, 314)
        self.assertIn('辅助功能', output.getvalue())
        self.assertIn('点击战斗画面', output.getvalue())

    @patch('render.render3D.launch.sys.platform', 'win32')
    @patch('render.render3D.launch._windows_browser', return_value='C:/Edge/msedge.exe')
    @patch('render.render3D.launch.subprocess.Popen')
    def test_windows_launches_an_isolated_fullscreen_window_and_cleans_its_profile(self, popen, browser):
        popen.return_value.poll.return_value = None
        url = 'http://127.0.0.1:1234/?token=test&inset=1'
        window = open_cot_window(url)
        self.assertIsInstance(window, BrowserProcessWindow)
        arguments = popen.call_args.args[0]
        self.assertEqual(arguments[0], browser.return_value)
        self.assertEqual(arguments[-1], url)
        self.assertIn('--new-window', arguments)
        self.assertIn('--start-fullscreen', arguments)
        profile = Path(window.profile.name)
        self.assertTrue(profile.is_dir())
        self.assertIn(f'--user-data-dir={profile}', arguments)
        try:
            self.assertTrue(window.is_open())
            popen.return_value.poll.return_value = 0
            self.assertFalse(window.is_open())
        finally:
            window.close()
        self.assertFalse(profile.exists())
        popen.return_value.terminate.assert_not_called()

    def test_owned_browser_shutdown_terminates_only_its_process_before_profile_cleanup(self):
        process, profile = Mock(), Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('browser', 2), 0]
        window = BrowserProcessWindow(process, profile)
        window.close()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        profile.cleanup.assert_called_once()

    @patch('render.render3D.launch.sys.platform', 'win32')
    @patch('render.render3D.launch._windows_browser', return_value='C:/Edge/msedge.exe')
    def test_windows_launch_failure_releases_the_temporary_profile(self, browser):
        with patch('render.render3D.launch.tempfile.TemporaryDirectory') as profile, \
                patch('render.render3D.launch.subprocess.Popen', side_effect=OSError('cannot launch')), \
                patch('sys.stdout', new_callable=io.StringIO):
            self.assertIsNone(open_cot_window('http://127.0.0.1:1234/'))
        profile.return_value.cleanup.assert_called_once()

    @patch('render.render3D.launch.shutil.which', return_value=None)
    def test_windows_browser_search_finds_a_standard_edge_installation(self, which):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / 'Microsoft/Edge/Application/msedge.exe'
            binary.parent.mkdir(parents=True)
            binary.touch()
            with patch.dict(os.environ, {'PROGRAMFILES(X86)': directory}):
                self.assertEqual(_windows_browser(), str(binary))

    @patch('render.render3D.launch.subprocess.run')
    def test_native_watch_targets_one_window_and_tolerates_script_errors(self, run):
        window = SafariWindow(314)
        run.side_effect = [subprocess.CompletedProcess([], 0, stdout='true\n'),
                           subprocess.CompletedProcess([], 0, stdout='false\n'),
                           subprocess.TimeoutExpired('osascript', 3)]
        self.assertTrue(window.is_open())
        self.assertEqual(run.call_args.args[0][-1], '314')
        self.assertFalse(window.is_open())
        self.assertIsNone(window.is_open())


if __name__ == '__main__':
    unittest.main()
