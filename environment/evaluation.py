"""Evaluation facades: training sees observations, actions and results.

Engine snapshots and rendering/window objects remain private to this layer.
BattleSession handles demonstrations; ModelEvaluationSession handles scored runs.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
import math
import operator
from pathlib import Path

from engine.contracts.outcome import outcome_from_counts
from engine.parameters import Team
from engine.battle_world import BattleWorld
from environment.jackal_env import JackalEnv
from environment.marl_adapter import JackalMultiAgentEnv
from environment.scenario import DEFAULT_GAME_STATE, build_ai_world
from render.api import RenderSession, ViewerClosed, create_viewer


class EvaluationStopped(RuntimeError):
    """A user closed the evaluation view; an unfinished episode has no score."""


class ModelEvaluationSession(JackalMultiAgentEnv):
    """Multi-episode evaluation with video and window ownership in this layer."""

    def __init__(self, env_config: dict):
        if env_config.get('name', 'jackal') != 'jackal':
            raise ValueError(f"Unsupported evaluation environment: {env_config['name']}")
        super().__init__(env_config)

    def _ensure_view_open(self) -> None:
        if self.env.renderer is not None and not self.env.renderer.poll_window():
            raise EvaluationStopped('window_closed')

    def _with_view(self, operation, *args, **kwargs):
        self._ensure_view_open()
        try:
            return operation(*args, **kwargs)
        except ViewerClosed as exc:
            # Rendering owns the closure message and reports it only once.
            if self.env.renderer is not None:
                self.env.renderer.poll_window()
            raise EvaluationStopped('window_closed') from exc

    def reset(self, seed: int | None = None):
        return self._with_view(super().reset, seed=seed)

    def step(self, actions):
        return self._with_view(super().step, actions)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class BattleSession:
    """Own one model-controlled or EnemyAI-controlled evaluation episode."""

    def __init__(self, *, state_file: str | Path = DEFAULT_GAME_STATE,
                 env_config: dict | None = None, seed: int = 42,
                 dt: float = .03, max_steps: int = 6000,
                 ai_intelligence: int = 5, enemy_intelligence: int | None = None,
                 render_mode: str | None = '2D3D', controller: str = 'EnemyAI Lv.5',
                 render_options: dict | None = None):
        self.dt = float(dt)
        self.max_steps = operator.index(max_steps)
        if not math.isfinite(self.dt) or self.dt <= 0 or self.max_steps <= 0:
            raise ValueError('dt and max_steps must be positive and finite')
        self.seed = int(seed)
        self.state_file = state_file
        self.ai_intelligence, self.enemy_intelligence = ai_intelligence, enemy_intelligence
        self.render_mode, self.controller = render_mode, controller
        self._options = deepcopy(render_options or {})
        self._env: JackalEnv | None = None
        self._world: BattleWorld | None = None
        self._viewer: RenderSession | None = None
        self._closed = False
        self._start_tick = 0
        self._episode_return = 0.
        self._env_config = deepcopy(env_config) if env_config is not None else None
        if self._env_config is not None:
            # The facade owns the view. The RL environment stays headless.
            config = {**self._env_config, 'headless': True, 'render_mode': None,
                      'use_video': False, 'seed': self.seed,
                      'fixed_delta_time': self.dt, 'max_steps': self.max_steps}
            config.pop('render_options', None)
            self._env = JackalEnv(**config)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError('BattleSession is closed')

    def _require_world(self) -> BattleWorld:
        self._require_open()
        if self._world is None:
            raise RuntimeError('Call reset() before evaluating a battle')
        return self._world

    def _model_env(self) -> JackalEnv:
        self._require_open()
        if self._env is None:
            raise ValueError('Model observations/actions require env_config')
        return self._env

    def get_env_info(self) -> dict:
        return self._model_env().get_env_info()

    def get_avail_actions(self):
        return self._model_env().get_avail_actions()

    def reset(self, seed: int | None = None):
        self._require_open()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if seed is not None:
            self.seed = int(seed)
        self._episode_return = 0.
        if self._env is not None:
            observations = self._env.reset(seed=self.seed)
            self._world = self._env.world
        else:
            self._world = build_ai_world(self.state_file, self.seed,
                ai_intelligence_level=self.ai_intelligence,
                enemy_intelligence_level=self.enemy_intelligence, normal_shell_only=False)
            observations = (None, None)
        self._start_tick = self._world.tick
        return observations

    def get_battle_result(self) -> dict:
        world = self._require_world()
        if self._env is not None:
            return self._env.get_battle_result()
        counts = {team.name: sum(unit.is_alive for unit in world.iter_units() if unit.team == team)
                  for team in (Team.PLAYER, Team.ENEMY)}
        return outcome_from_counts(counts, world.tick - self._start_tick, self.max_steps)

    def step(self, actions=None):
        world = self._require_world()
        if self.get_battle_result()['finished']:
            raise RuntimeError('Battle has finished; reset before stepping again')
        if self._env is not None:
            if actions is None:
                raise ValueError('Model-controlled evaluation requires agent actions')
            obs, state, reward, done, info = self._env.step(actions)
            self._episode_return += float(reward)
        else:
            if actions is not None:
                raise ValueError('EnemyAI evaluation does not accept model actions')
            world.step(self.dt)
            obs, state, reward, info = None, None, 0., {}
            done = self.get_battle_result()['finished']
        # Evaluation augments info; the training environment's original info
        # schema/rewards remain unchanged for existing runners and baselines.
        return obs, state, reward, done, {**info, 'battle_result': self.get_battle_result()}

    def render(self) -> bool:
        world = self._require_world()
        if self.render_mode is None:
            return True
        if self._viewer is None:
            options = self._env_config or {}
            view_options = deepcopy(self._options)
            view_options.setdefault('width', int(options.get('viewport_width', 960)))
            view_options.setdefault('height', int(options.get('viewport_height', 640)))
            view_options.update(dt=self.dt, controller=self.controller,
                                max_steps=self.max_steps, seed=self.seed)
            self._viewer = create_viewer(world, self.render_mode, **view_options)
        return self._viewer.show_battle(world.render_snapshot(), self.get_battle_result())

    def is_open(self) -> bool:
        self._require_world()
        return self._viewer.poll_window() if self._viewer is not None else True

    def wait_frame(self, speed: float = 1.) -> bool:
        self._require_world()
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError('Playback speed must be finite and positive')
        return self._viewer.pace(self.dt / speed) if self._viewer is not None else True

    def wait_for_close(self) -> None:
        self._require_world()
        if self._viewer is not None:
            self._viewer.wait_for_close()

    def get_report(self) -> dict:
        world = self._require_world()
        frame = world.render_snapshot()
        report = {**self.get_battle_result(), 'tick': frame.tick, 'time': frame.time,
                  'render': self.render_mode, 'controller': self.controller, 'seed': self.seed}
        if self._env is not None:
            report['episode_return'] = self._episode_return
        return report

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with ExitStack() as cleanup:
            if self._env is not None:
                cleanup.callback(self._env.close)
            if self._viewer is not None:
                cleanup.callback(self._viewer.close)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()
