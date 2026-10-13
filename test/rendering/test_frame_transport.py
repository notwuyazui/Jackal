"""First-frame and recovery transport tests without a network dependency."""
import base64
import hashlib
import io
import json
import threading
import unittest
from environment.presentation.adapter import build_world
from environment.presentation.server import handler_for
from render.api import SnapshotStream


class FrameTransportTests(unittest.TestCase):
    def setUp(self):
        self.world = build_world()
        self.hub = SnapshotStream()
        self.hub.initialize(self.world.render_manifest(.03))

    def tearDown(self):
        self.hub.close()

    def request(self, path):
        handler = object.__new__(handler_for(self.hub))
        handler.path = path
        response = {}
        handler.respond = lambda payload, status=200: response.update(payload=payload, status=status)
        handler.do_GET()
        return response

    def test_http_bootstraps_exact_pose_and_png_without_sse(self):
        self.hub.publish(self.world.render_snapshot())
        response = self.request(f'/api/frame?token={self.hub.view_token}')
        self.assertEqual(response['status'], 200)
        frame = response['payload']
        self.assertEqual(frame['version'], 0)
        self.assertEqual(frame['hash'], self.hub.latest['hash'])
        image = base64.b64decode(frame['jackal_png'].split(',', 1)[1])
        self.assertEqual(hashlib.sha256(image).hexdigest(), frame['jackal_png_sha256'])

    def test_http_recovery_receives_next_pose_while_world_waits_for_ack(self):
        self.hub.lockstep = True
        self.hub.startup_timeout = 2
        worker = threading.Thread(target=lambda: self.hub.publish(self.world.render_snapshot()))
        worker.start()
        try:
            frame = self.request(f'/api/frame?token={self.hub.view_token}')['payload']
            self.hub.acknowledge({'view_token': self.hub.view_token, 'version': frame['version'],
                'tick': frame['tick'], 'hash': frame['hash'],
                'jackal_inset': {'png_sha256': frame['jackal_png_sha256']}})
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(self.hub.acks['3d']['tick'], 0)
        finally:
            self.hub.close()
            worker.join(timeout=2)

    def test_wrong_window_token_is_rejected_before_waiting_for_a_frame(self):
        response = self.request('/api/frame?token=old-window')
        self.assertEqual(response['status'], 403)
        self.assertEqual(response['payload']['error'], 'stale CoT window')

    def test_renderer_error_endpoint_authenticates_and_unblocks_owner(self):
        for token, status in (('wrong', 403), (self.hub.view_token, 200)):
            data = json.dumps({'view_token': token, 'message': 'WebGL initialization failed'}).encode()
            handler = object.__new__(handler_for(self.hub))
            handler.path = '/api/error'
            handler.headers = {'Content-Length': str(len(data))}
            handler.rfile = io.BytesIO(data)
            response = {}
            handler.respond = lambda payload, status=200: response.update(status=status)
            handler.do_POST()
            self.assertEqual(response['status'], status)
            if token == 'wrong':
                self.assertIsNone(self.hub.failure)
        with self.assertRaisesRegex(RuntimeError, 'WebGL initialization failed'):
            self.hub.publish(self.world.render_snapshot())
        response = self.request(f'/api/frame?token={self.hub.view_token}')
        self.assertEqual(response['status'], 503)
        self.assertIn('WebGL initialization failed', response['payload']['error'])

    def test_pure_3d_bootstrap_does_not_include_a_2d_image(self):
        self.hub.close()
        self.hub = SnapshotStream(show_inset=False)
        self.hub.initialize(self.world.render_manifest(.03))
        self.hub.publish(self.world.render_snapshot())
        frame = self.request(f'/api/frame?token={self.hub.view_token}')['payload']
        self.assertEqual(frame['jackal_png'], '')
        self.assertEqual(frame['jackal_png_sha256'], '')

    def test_sse_flushes_connection_header_without_waiting_for_a_pose(self):
        self.hub.close()
        handler = object.__new__(handler_for(self.hub))
        handler.path = f'/api/events?view=3d&token={self.hub.view_token}'
        handler.wfile = io.BytesIO()
        handler.send_response = lambda *_: None
        handler.send_header = lambda *_: None
        handler.end_headers = lambda: None
        handler.do_GET()
        self.assertEqual(handler.wfile.getvalue(), b'retry: 1000\n: connected\n\nevent:closed\ndata:{}\n\n')


if __name__ == '__main__':
    unittest.main()
