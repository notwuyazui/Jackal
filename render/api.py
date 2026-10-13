"""The single rendering API for environments, evaluation and interactive play.

    with create_viewer(world, "2D3D", dt=.03) as view:
        while view.poll():
            view.render(world.render_snapshot())

Only immutable manifests/frames reach the backends. Window creation, HUD, input,
browser transport and cleanup belong here, not to training/evaluation scripts.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import TYPE_CHECKING

from engine.contracts.render import BattleManifest, FrameSnapshot
from render.transport.stream import SnapshotStream, ViewerClosed

if TYPE_CHECKING:
    from engine.battle_world import BattleWorld
    from render.render2D.pygame_renderer import PygameRenderer
    from render.render3D.bridge import BrowserRenderer

_ALIASES = {"human": "2D", "rgb_array": "RGB_ARRAY", "browser": "2D3D",
            "browser_lockstep": "2D3D"}


def _render_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise ValueError(f'Unknown render mode: {mode!r}')
    selected = _ALIASES.get(mode.lower(), mode.upper())
    if selected not in {'2D', '3D', '2D3D', 'RGB_ARRAY'}:
        raise ValueError(f'Unknown render mode: {mode!r}')
    return selected


class RenderSession:
    """One owner for a visible/offscreen renderer, including its window lifecycle."""
    def __init__(self, mode: str = "2D3D", width: int = 960, height: int = 640,
                 *, hud: bool = False, scene_id: str | None = None, **options):
        selected = _render_mode(mode)
        self.mode, self.scene_id, self.show_hud = selected, scene_id, hud
        self.closed = False
        self.manifest: BattleManifest | None = None
        self.scene: dict | None = None
        self._font = None
        self._mouse_pos = None
        self._deployed = False
        self._window_closed = False
        self._deadline = None
        self.backend: PygameRenderer | BrowserRenderer
        if selected in {"2D", "RGB_ARRAY"}:
            from render.render2D.pygame_renderer import PygameRenderer
            self.backend = PygameRenderer(width, height, visible=selected == "2D", **options)
        else:
            from render.render3D.bridge import BrowserRenderer
            options.setdefault("lockstep", mode.lower() != 'browser')
            options["show_inset"] = selected == "2D3D"
            self.backend = BrowserRenderer(scene_id=scene_id, **options)

    def initialize(self, manifest: BattleManifest) -> None:
        if self.closed:
            raise RuntimeError("RenderSession is closed")
        if self.mode in {"3D", "2D3D"}:
            from render.render3D.maps.registry import resolve_scene
            self.scene = resolve_scene(manifest, self.scene_id).to_dict()
        self.backend.initialize(manifest)
        self.manifest = manifest
        self._deployed = False

    def _check_ready(self) -> BattleManifest:
        if self.closed:
            raise RuntimeError('RenderSession is closed')
        if self.manifest is None:
            raise RuntimeError('Call initialize(manifest) before rendering')
        return self.manifest

    def poll(self) -> bool:
        """False means the user closed the window (or pressed Esc in 2D)."""
        if self.closed:
            return False
        from render.render3D.bridge import BrowserRenderer
        if isinstance(self.backend, BrowserRenderer):
            return self.backend.poll()
        if self.mode == "RGB_ARRAY":
            return True
        import pygame
        from engine.parameters import Direction
        directions = {pygame.K_UP: Direction.UP, pygame.K_DOWN: Direction.DOWN,
                      pygame.K_LEFT: Direction.LEFT, pygame.K_RIGHT: Direction.RIGHT}
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                return False
            if event.type == pygame.KEYDOWN and event.key in directions:
                self.backend.camera.move(directions[event.key])
        return True

    def render(self, frame: FrameSnapshot, result: dict | None = None):
        manifest = self._check_ready()
        from render.render3D.bridge import BrowserRenderer
        if isinstance(self.backend, BrowserRenderer):
            if not self._deployed:
                from render.render3D.maps.registry import deployment
                self.scene = deployment(manifest, frame, self.scene_id)
                self.backend.hub.info['scene'] = self.scene
                self._deployed = True
            return self.backend.render(frame, result)
        surface = self.backend.draw(frame, mouse_pos=self._mouse_pos)
        if self.show_hud and result is not None:
            import pygame
            if self._font is None:
                self._font = pygame.font.Font(None, 22)
            status = ({"PLAYER": "BLUE WINS", "ENEMY": "RED WINS", "DRAW": "DRAW"}
                      .get(result["winner"], "RUNNING"))
            text = (f"{frame.time:.2f}s  STEP {frame.tick}  Blue {result['alive']['PLAYER']} : "
                    f"{result['alive']['ENEMY']} Red  {status}")
            surface.blit(self._font.render(text, True, (240, 240, 240)), (10, 12))
        self.backend.present()
        return surface

    def _notify_window_closed(self) -> bool:
        if not self._window_closed:
            self._window_closed = True
            print('[View] 用户关闭测试窗口', flush=True)
        return False

    def poll_window(self) -> bool:
        """Interactive closure is a normal status, handled inside rendering."""
        if self.closed or self._window_closed:
            return False
        try:
            return True if self.poll() else self._notify_window_closed()
        except ViewerClosed:
            return self._notify_window_closed()

    def show_battle(self, frame: FrameSnapshot, result: dict) -> bool:
        """Present a battle without exposing window exceptions to callers."""
        if self.closed or self._window_closed:
            return False
        try:
            self.render(frame, result)
        except ViewerClosed:
            return self._notify_window_closed()
        if self._deadline is None:
            self._deadline = time.monotonic()
        return True

    def pace(self, interval: float) -> bool:
        """Wait between physics frames while continuing to process closure."""
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Frame interval must be finite and positive')
        now = time.monotonic()
        previous = now if self._deadline is None else self._deadline
        self._deadline = max(previous + interval, now - interval)
        while self.poll_window():
            remaining = self._deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(.01, remaining))
        return False

    def wait_for_close(self) -> None:
        """Keep the final result visible; window UI and messages stay here."""
        if not self.poll_window():
            return
        print('对局已结束，保留结果窗口。关闭测试窗口即可退出（2D 也支持 Esc）。', flush=True)
        while self.poll_window():
            time.sleep(.05)

    def draw(self, frame: FrameSnapshot, **options):
        """Draw offscreen, for video or interactive play; render() presents a frame."""
        self._check_ready()
        from render.render2D.pygame_renderer import PygameRenderer
        if not isinstance(self.backend, PygameRenderer):
            raise ValueError("draw/rgb_array are available in 2D and rgb_array modes")
        return self.backend.draw(frame, **options)

    def present(self) -> None:
        self._check_ready()
        from render.render2D.pygame_renderer import PygameRenderer
        if isinstance(self.backend, PygameRenderer):
            self.backend.present()

    def rgb_array(self):
        self._check_ready()
        from render.render2D.pygame_renderer import PygameRenderer
        if not isinstance(self.backend, PygameRenderer):
            raise ValueError("Use a separate rgb_array renderer for 3D video recording")
        return self.backend.rgb_array()

    def keyboard_controls(self, world: BattleWorld, action):
        self._check_ready()
        from render.render2D.pygame_renderer import PygameRenderer
        from render.render2D.input import PCControl
        if not isinstance(self.backend, PygameRenderer) or self.mode != "2D":
            raise ValueError("Keyboard controls require a visible 2D renderer")
        import pygame
        self._mouse_pos = pygame.mouse.get_pos()
        return PCControl(world, action, self.backend.camera)

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.backend.close()

    def diagnostics(self) -> dict:
        """Read-only presentation diagnostics, including browser GPU resource counts."""
        from render.render3D.bridge import BrowserRenderer
        if isinstance(self.backend, BrowserRenderer):
            with self.backend.hub.condition:
                return {"mode": self.mode, "closed": self.closed, **deepcopy(self.backend.hub.validation)}
        return {"mode": self.mode, "closed": self.closed,
                "viewport": (self.backend.viewport_width, self.backend.viewport_height)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def create_renderer(mode: str | None, width: int = 960, height: int = 640, **options) -> RenderSession | None:
    """Environment factory. None preserves a fully headless environment."""
    return None if mode is None else RenderSession(mode, width, height, **options)


def create_viewer(world: BattleWorld, mode: str = "2D3D", *, dt: float = .03,
                  width: int = 960, height: int = 640, controller: str = "EnemyAI Lv.5",
                  scene_id: str | None = None, **options) -> RenderSession:
    """Validate first, then open and initialize the requested view."""
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    selected = _render_mode(mode)
    manifest = world.render_manifest(dt)
    scene = apply_map(world, scene_id, dt=dt) if selected in {"3D", "2D3D"} else None
    backend_options = ({"title": f"Jackal · {controller}",
                        **{key: value for key, value in options.items()
                           if key in {"title", "draw_sight_range", "draw_health_bar"}}} if selected in {"2D", "RGB_ARRAY"}
                       else {**options, "controller": controller})
    view = RenderSession(mode, width, height, hud=True, scene_id=scene_id, **backend_options)
    try:
        view.initialize(manifest)
        view.scene = scene
        return view
    except Exception:
        view.close()
        raise


def create_video_writer(*args, **kwargs):
    from render.render2D.video import create_video_writer as factory
    return factory(*args, **kwargs)


def create_inset_renderer(width: int, height: int):
    """Compatibility for the recorded demo; no separate SDL window is created."""
    from render.render2D.inset import InsetRenderer
    return InsetRenderer(width, height)


def open_browser_window(url: str):
    from render.render3D.launch import open_cot_window
    return open_cot_window(url)


def apply_map(world: BattleWorld, scene_id: str | None = None, *, dt: float = .03) -> dict:
    from render.render3D.maps.registry import apply_map as apply
    return apply(world, scene_id, dt=dt)


def list_maps():
    from render.render3D.maps.registry import list_maps as catalog
    return catalog()


def list_models() -> tuple[str, ...]:
    from render.render3D.models.registry import list_models as catalog
    return catalog()


def load_map(scene_id: str, **options) -> BattleWorld:
    from render.render3D.maps.registry import load_map as load
    return load(scene_id, **options)


def create_http_handler(stream):
    """Compatibility boundary for the optional recorded demo."""
    from render.transport.server import handler_for
    return handler_for(stream)


def create_http_server(address, handler):
    from render.transport.server import LocalDemoServer
    return LocalDemoServer(address, handler)
