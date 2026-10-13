import unittest
import threading
from environment.presentation.adapter import build_world
from render.api import SnapshotStream
from render.transport.stream import ViewerClosed
from unittest.mock import Mock

class SnapshotStreamTests(unittest.TestCase):
    def test_environment_result_is_authoritative_and_copied_before_publication(self):
        world = build_world()
        hub = SnapshotStream(show_inset=False, max_steps=6000)
        result = {'finished': True, 'winner': 'DRAW', 'reason': 'evaluation_limit',
                  'alive': {'PLAYER': 3, 'ENEMY': 3}}
        try:
            hub.initialize(world.render_manifest(.03))
            hub.publish(world.render_snapshot(), result)
            self.assertEqual(hub.latest['result'], result)
            self.assertTrue(hub.latest['paused'])
            result['alive']['PLAYER'] = 99
            self.assertEqual(hub.latest['result']['alive']['PLAYER'], 3)
            hub.acknowledge(self.receipt(hub, hub.latest))
            self.assertEqual(hub.acks['3d']['tick'], 0)
        finally:
            hub.close()

    @staticmethod
    def receipt(hub, frame, **values):
        return {'view_token': hub.view_token, 'version': frame['version'],
                'tick': frame['tick'], 'hash': frame['hash'],
                'jackal_inset': {'png_sha256': frame['jackal_png_sha256']}, **values}

    def test_delayed_ack_cannot_regress_current_confirmation(self):
        world = build_world()
        hub = SnapshotStream(show_inset=False)
        try:
            hub.initialize(world.render_manifest(.03))
            hub.publish(world.render_snapshot())
            old = hub.latest
            world.step(.03)
            hub.publish(world.render_snapshot())
            current = hub.latest
            hub.acknowledge(self.receipt(hub, current, performance={'fps': 60}))
            hub.acknowledge(self.receipt(hub, old, performance={'fps': 1}))
            self.assertEqual(hub.acks['3d']['version'], current['version'])
            self.assertEqual(hub.validation['performance'], {'fps': 60})
            self.assertEqual(hub.ack_cursor, current['event_cursor'])
        finally:
            hub.close()

    def test_invalid_event_cursor_cannot_remove_events_or_confirm_a_frame(self):
        world = build_world()
        hub = SnapshotStream(show_inset=False)
        try:
            hub.initialize(world.render_manifest(.03))
            hub.publish(world.render_snapshot())
            events = tuple(hub.events)
            for cursor in (-1, .5, True, hub.latest['event_cursor'] + 1):
                with self.assertRaisesRegex(ValueError, 'event cursor'):
                    hub.acknowledge(self.receipt(hub, hub.latest, event_cursor=cursor))
                self.assertEqual(tuple(hub.events), events)
                self.assertEqual(hub.acks, {})
        finally:
            hub.close()

    def test_close_releases_wait_and_prevents_further_publishing(self):
        self.check_interrupted_publish('close', ViewerClosed)

    def test_service_failure_releases_wait_with_the_original_error(self):
        self.check_interrupted_publish('fail', RuntimeError)

    def check_interrupted_publish(self, action, error_type):
        world = build_world()
        hub = SnapshotStream(show_inset=False, lockstep=True, startup_timeout=120)
        hub.initialize(world.render_manifest(.03))
        errors = []
        def publish():
            try:
                hub.publish(world.render_snapshot())
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=publish, daemon=True)
        worker.start()
        try:
            with hub.condition:
                self.assertTrue(hub.condition.wait_for(lambda: hub.latest is not None, timeout=2))
            if action == 'fail':
                hub.fail('test WebGL failure')
            else:
                hub.close()
            worker.join(timeout=2)
            self.assertFalse(worker.is_alive())
            self.assertIsInstance(errors[0], error_type)
            if action == 'fail':
                self.assertIn('test WebGL failure', str(errors[0]))
            version = hub.version
            with self.assertRaises(error_type):
                hub.publish(world.render_snapshot())
            self.assertEqual(hub.version, version)
        finally:
            hub.close()
            worker.join(timeout=2)

    def test_closed_stream_cannot_be_reinitialized_or_close_resources_twice(self):
        hub = SnapshotStream(show_inset=False)
        renderer = Mock()
        hub.renderer = renderer
        hub.close(); hub.close()
        renderer.close.assert_called_once()
        with self.assertRaises(ViewerClosed):
            hub.initialize(build_world().render_manifest(.03))

    def test_lockstep_releases_wait_only_after_matching_frame_ack(self):
        world = build_world()
        hub = SnapshotStream(lockstep=True, startup_timeout=2, frame_timeout=.02)
        try:
            hub.initialize(world.render_manifest(.03))
            def acknowledge_first_frame():
                with hub.condition:
                    hub.condition.wait_for(lambda: hub.latest is not None, timeout=2)
                    frame = hub.latest
                if frame is not None:
                    hub.acknowledge({'view': '3d', 'view_token': hub.view_token,
                        'version': frame['version'], 'tick': frame['tick'], 'hash': frame['hash'],
                        'jackal_inset': {'png_sha256': frame['jackal_png_sha256']}})
            viewer = threading.Thread(target=acknowledge_first_frame)
            viewer.start()
            hub.publish(world.render_snapshot())
            viewer.join(timeout=2)
            self.assertFalse(viewer.is_alive())
            self.assertEqual(hub.acks['3d']['tick'], 0)
            world.step(.03)
            with self.assertRaisesRegex(TimeoutError, 'STEP 1 within 0.02 seconds'):
                hub.publish(world.render_snapshot())
        finally:
            hub.close()

    def test_first_frame_has_separate_startup_timeout(self):
        world = build_world()
        hub = SnapshotStream(lockstep=True, startup_timeout=.02, frame_timeout=2)
        try:
            hub.initialize(world.render_manifest(.03))
            with self.assertRaisesRegex(TimeoutError, 'STEP 0 within 0.02 seconds'):
                hub.publish(world.render_snapshot())
        finally:
            hub.close()

    def test_slow_viewer_keeps_events_while_pose_queue_is_bounded(self):
        world=build_world(); hub=SnapshotStream()
        try:
            hub.initialize(world.render_manifest(.03))
            for _ in range(100):
                world.step(.03);hub.publish(world.render_snapshot())
            self.assertEqual(len(hub.frames),8)
            self.assertEqual(hub.latest['tick'],100)
            seqs=[e['seq'] for e in hub.latest['pending_events']]
            self.assertGreater(len(seqs),0)
            self.assertEqual(seqs,sorted(set(seqs)))
            latest=hub.latest
            hub.acknowledge({'view':'3d','view_token':hub.view_token,
                'version':latest['version'],'tick':latest['tick'],'hash':latest['hash'],
                'event_cursor':latest['event_cursor'],
                'jackal_inset':{'png_sha256':latest['jackal_png_sha256']}})
            self.assertEqual(len(hub.events),0)
            self.assertEqual(hub.ack_cursor,world.bullet_manager.event_sequence)
        finally: hub.close()

    def test_reset_restarts_event_cursor_and_requires_new_run(self):
        hub=SnapshotStream()
        try:
            world=build_world();hub.initialize(world.render_manifest(.03))
            world.step(.03);hub.publish(world.render_snapshot());run=hub.run_id
            hub.initialize(build_world().render_manifest(.03))
            self.assertEqual(hub.run_id,run+1)
            self.assertEqual(hub.ack_cursor,0)
            self.assertIsNone(hub.latest)
            self.assertEqual(len(hub.events),0)
        finally: hub.close()
