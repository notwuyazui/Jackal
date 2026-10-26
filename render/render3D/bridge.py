"""Renderer interface for MARL evaluation; consumes immutable engine snapshots."""
from pathlib import Path
from contextlib import ExitStack
import os, shutil, socket, subprocess, threading, time


def available_view_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def find_node():
    configured = os.environ.get('JACKAL_NODE')
    candidates = [configured, shutil.which('node'),
                  str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise RuntimeError('Node.js not found; install Node or set JACKAL_NODE')


class BrowserRenderer:
    def __init__(self, *, lockstep=False, max_steps=6000, follow_id=2,
                 seed=42, port=0, view_port=0, open_browser=True,
                 startup_timeout=120, frame_timeout=45, show_inset=True,
                 controller='EnemyAI Lv.5', scene_id=None):
        from render.transport.stream import SnapshotStream
        from render.transport.server import LocalDemoServer, handler_for
        root = Path(__file__).resolve().parent
        if not (root/'node_modules/vite/bin/vite.js').is_file():
            raise RuntimeError('请先在 render/render3D 中执行 npm ci，安装浏览器渲染依赖')
        node = find_node()
        self.hub = SnapshotStream(lockstep=lockstep, max_steps=max_steps, follow_id=follow_id,
                                  seed=seed, startup_timeout=startup_timeout, frame_timeout=frame_timeout,
                                  show_inset=show_inset, controller=controller, scene_id=scene_id)
        self.open_browser = open_browser
        self.opened = False
        self.closed = False
        self.window_watch_stop = threading.Event()
        self.window_thread = None
        self.log_lock = threading.Lock()
        self.follow_missing = False
        # Every setup step can fail, including temporary-file creation and port
        # allocation. Roll back already owned sockets/processes on any failure.
        with ExitStack() as resources:
            self.server = LocalDemoServer(('127.0.0.1', port), handler_for(self.hub))
            resources.callback(self.server.server_close)
            self.api_port = self.server.server_port
            self.view_port = view_port or available_view_port()
            import tempfile
            self.log = resources.enter_context(tempfile.TemporaryFile(mode='w+', prefix='jackal-browser-', suffix='.log'))
            env = {**os.environ, 'JACKAL_API_PORT': str(self.api_port), 'JACKAL_VIEW_PORT': str(self.view_port)}
            self.vite = subprocess.Popen([node, str(root/'node_modules/vite/bin/vite.js'),
                                         '--config', str(root/'vite.config.ts')], cwd=root,
                                        env=env, stdout=self.log, stderr=subprocess.STDOUT)
            resources.callback(self._stop_vite)
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            resources.callback(self._stop_server)
            self.service_thread = threading.Thread(target=self._watch_service, daemon=True)
            self.service_thread.start()
            resources.callback(self.window_watch_stop.set)
            self.resources = resources.pop_all()

    def _stop_vite(self):
        if self.vite.poll() is None:
            self.vite.terminate()
            try:
                self.vite.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.vite.kill()
                self.vite.wait()

    def _stop_server(self):
        self.server.shutdown()
        self.server_thread.join(timeout=2)

    def _service_failure(self):
        with self.log_lock:
            self.log.seek(0)
            return '3D 服务已退出：\n' + self.log.read()[-4000:]

    def _watch_service(self):
        while not self.window_watch_stop.wait(.2):
            if self.vite.poll() is not None:
                self.hub.fail(self._service_failure())
                return

    @property
    def url(self):
        return (f'http://127.0.0.1:{self.view_port}/?view=3d&token={self.hub.view_token}'
                f'&inset={int(self.hub.show_inset)}')

    def initialize(self, manifest):
        self.hub.initialize(manifest)
        # Team membership is in the first snapshot, not unit parameters.
        self.follow_missing = not any(u.id == self.hub.follow_id for u in manifest.units)
        print(f"Viewer: {self.url}", flush=True)

    def render(self, frame, result=None):
        if self.vite.poll() is not None:
            self.hub.fail(self._service_failure())
        self.hub.check_active()
        if self.follow_missing:
            followed = next((u for u in frame.units if u.team == 'PLAYER' and u.alive),
                            next((u for u in frame.units if u.team == 'PLAYER'), frame.units[0]))
            self.hub.follow_id = followed.id
            self.hub.info['follow_id'] = followed.id
            self.follow_missing = False
        if self.open_browser and not self.opened:
            from render.render3D.launch import open_cot_window
            url = self.url
            def show():
                import urllib.request
                deadline = time.monotonic() + 20
                while not self.window_watch_stop.is_set() and time.monotonic() < deadline:
                    try:
                        urllib.request.urlopen(url, timeout=.5).close()
                        if self.window_watch_stop.is_set():
                            return
                        window = open_cot_window(url)
                        if window is not None:
                            # Native window identity survives reloads and does
                            # not depend on SSE or an unload beacon arriving.
                            try:
                                while not self.window_watch_stop.wait(.5):
                                    if window.is_open() is False:
                                        self.hub.request_viewer_close()
                                        return
                            finally:
                                close = getattr(window, 'close', None)
                                if close is not None:
                                    close()
                        return
                    except OSError:
                        self.window_watch_stop.wait(.1)
                    except Exception as error:
                        self.hub.fail(f'无法打开 3D 窗口：{error}')
                        return
                if not self.window_watch_stop.is_set():
                    self.hub.fail('3D 服务在 20 秒内未就绪，请检查 Node/Vite 启动日志')
            self.window_thread = threading.Thread(target=show, daemon=True)
            self.window_thread.start()
            self.opened = True
        self.hub.publish(frame, result)

    def poll(self):
        if self.closed or self.hub.closed or self.hub.viewer_closed:
            return False
        self.hub.check_active()
        return True

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.window_watch_stop.set()
        try:
            self.hub.close()
        finally:
            self.service_thread.join(timeout=2)
            if self.window_thread:
                self.window_thread.join(timeout=3)
            self.resources.close()
