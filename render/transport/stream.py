"""Read-only, bounded presentation stream for an existing RL environment.

The newest pose replaces older poses; unacknowledged events remain queued.
No world or step function is available to this transport.
"""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import base64, hashlib, json, secrets, threading
from engine.contracts.outcome import battle_result


class ViewerClosed(Exception):
    """The presentation window or its owning session was closed."""


class SnapshotStream:
    def __init__(self, *, lockstep=False, max_steps=6000, follow_id=2, seed=42,
                 startup_timeout=120, frame_timeout=45, show_inset=True,
                 controller='EnemyAI Lv.5', scene_id=None):
        self.condition = threading.Condition()
        self.lockstep, self.max_steps, self.follow_id, self.seed = lockstep, max_steps, follow_id, seed
        self.startup_timeout, self.frame_timeout = startup_timeout, frame_timeout
        self.show_inset, self.controller = show_inset, controller
        self.scene_id = scene_id
        self.start_tick = None
        self.version = -1
        self.run_id = 0
        self.view_token = secrets.token_urlsafe(12)
        self.latest = None
        self.png = ''
        self.acks = {}
        self.validation = {'errors': []}
        self.frames = deque(maxlen=8)
        self.events = deque()
        self.ack_cursor = 0
        self.renderer = None
        self.info = {}
        self.closed = False
        self.failure = None
        self.viewer_closed = False
        self.window_client = None
        self.window_close_timer = None
        self.restart_requested = False

    def fail(self, message):
        """Surface a renderer/service failure to the owner, including a waiting step."""
        with self.condition:
            if self.closed or self.viewer_closed or self.failure is not None:
                return
            self.failure = str(message)
            self.validation['errors'].append(self.failure)
            self.condition.notify_all()

    def check_active(self):
        with self.condition:
            if self.closed or self.viewer_closed:
                raise ViewerClosed()
            if self.failure is not None:
                raise RuntimeError(self.failure)

    def request_viewer_close(self):
        # Wake a lockstep publisher immediately. SDL resources are released by
        # the owner thread in close(), never by a browser/server callback.
        with self.condition:
            self.viewer_closed = True
            if self.window_close_timer:
                self.window_close_timer.cancel()
                self.window_close_timer = None
            self.condition.notify_all()

    def window_event(self, action, client):
        with self.condition:
            if self.closed or self.viewer_closed:
                return
            if action == 'open':
                # Reloads create a new client before the close grace expires.
                # A delayed request from the departing page cannot reopen it.
                if client == self.window_client and self.window_close_timer:
                    return
                self.window_client = client
                if self.window_close_timer:
                    self.window_close_timer.cancel()
                    self.window_close_timer = None
            elif self.window_client in (None, client):
                self.window_client = client
                if self.window_close_timer:
                    self.window_close_timer.cancel()
                def finish_close():
                    with self.condition:
                        if self.window_client == client and not self.closed:
                            self.request_viewer_close()
                self.window_close_timer = threading.Timer(2, finish_close)
                self.window_close_timer.daemon = True
                self.window_close_timer.start()

    def initialize(self, manifest):
        from render.render3D.maps.registry import resolve_scene
        scene = resolve_scene(manifest, self.scene_id)
        with self.condition:
            self.check_active()
            if self.renderer:
                self.renderer.close()
            self.renderer = None
            if self.show_inset:
                from render.render2D.inset import InsetRenderer
                self.renderer = InsetRenderer(*manifest.map.size)
                self.renderer.initialize(manifest)
            self.run_id += 1
            self.latest = None
            self.events.clear()
            self.frames.clear()
            self.acks.clear()
            self.ack_cursor = 0
            self.start_tick = None
            self.info = {**asdict(manifest), 'seed': self.seed, 'max_steps': self.max_steps,
                         'follow_id': self.follow_id, 'meters_per_pixel': scene.meters_per_pixel,
                         'scene': scene.to_dict(),
                         'show_inset': self.show_inset, 'controller': self.controller,
                         'mode': 'lockstep' if self.lockstep else 'async'}

    def publish(self, snapshot, result=None):
        self.check_active()
        if not self.info:
            raise RuntimeError('Call initialize(manifest) before publishing')
        if self.start_tick is None:
            self.start_tick = snapshot.tick
            self.info['max_steps'] = self.start_tick + self.max_steps
        frame = snapshot.to_dict()
        # The environment owns termination. Raw snapshot clients can still use
        # the shared rules when no environment result has been supplied.
        frame['result'] = (battle_result(snapshot, self.info['max_steps'])
                           if result is None else deepcopy(result))
        finished = frame['result']['finished']
        frame['hash'] = hashlib.sha256(json.dumps(frame, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()[:16]
        png = self.renderer.png(snapshot, frame) if self.renderer else b''
        with self.condition:
            self.check_active()
            last = self.events[-1]['seq'] if self.events else self.ack_cursor
            for event in frame['events']:
                if event['seq'] > last:
                    self.events.append(event)
                    last = event['seq']
            if len(self.events) > 4096:
                raise BufferError('Browser event queue exceeded 4096; restart the viewer before continuing')
            self.version += 1
            self.latest = {**frame, 'version': self.version, 'run_id': self.run_id,
                           'manifest': self.info, 'paused': finished, 'speed': 1,
                           'jackal_png_sha256': hashlib.sha256(png).hexdigest() if self.show_inset else '',
                           'pending_events': list(self.events)}
            self.frames.append(self.latest)
            self.png = base64.b64encode(png).decode()
            self.condition.notify_all()
            if self.lockstep:
                # Loading native models, terrain and shaders takes longer than
                # presenting a frame in an already initialized browser.
                timeout = self.frame_timeout if self.acks else self.startup_timeout
                version = self.version
                ready = self.condition.wait_for(lambda: self.closed or self.viewer_closed or self.failure is not None or
                    self.acks.get('3d', {}).get('version', -1) >= version, timeout=timeout)
                self.check_active()
                if not ready:
                    raise TimeoutError(f'Browser did not acknowledge STEP {snapshot.tick} within {timeout} seconds')

    def acknowledge(self, payload):
        with self.condition:
            if self.closed or self.viewer_closed or payload.get('view_token') != self.view_token:
                return
            frame = next((f for f in self.frames if f['version'] == payload.get('version')), None)
            if frame is None:
                return
            # SSE and HTTP recovery can acknowledge out of order. An older
            # receipt must never undo a newer confirmation or its diagnostics.
            if frame['version'] <= self.acks.get('3d', {}).get('version', -1):
                return
            if payload.get('hash') != frame['hash'] or payload.get('tick') != frame['tick']:
                raise ValueError('Browser frame hash/tick mismatch')
            inset = payload.get('jackal_inset', {})
            if self.show_inset and inset.get('png_sha256') != frame['jackal_png_sha256']:
                raise ValueError('Browser 2D inset does not match the snapshot')
            cursor = payload.get('event_cursor', frame['event_cursor'])
            if type(cursor) is not int or not 0 <= cursor <= frame['event_cursor']:
                raise ValueError('Invalid browser event cursor')
            self.ack_cursor = max(self.ack_cursor, cursor)
            while self.events and self.events[0]['seq'] <= self.ack_cursor:
                self.events.popleft()
            self.acks['3d'] = {'version': frame['version'], 'tick': frame['tick']}
            self.validation['tick'] = frame['tick']
            self.validation['performance'] = payload.get('performance', {})
            self.validation['scene_id'] = payload.get('scenery', {}).get('map_id')
            self.condition.notify_all()

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
            if self.window_close_timer:
                self.window_close_timer.cancel()
                self.window_close_timer = None
            self.condition.notify_all()
            renderer, self.renderer = self.renderer, None
        if renderer:
            renderer.close()
