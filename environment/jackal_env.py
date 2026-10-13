import os
import datetime
import math
import operator
from random import Random
from typing import Optional

import engine.rules as GameMode
from engine.parameters import BULLET_SPEED, DEFAULT_AI_INTELLIGENCE_LEVEL, Team
from engine.maps.game_map import GameMap
from engine.projectiles.bullet_manager import BulletManager
from engine.contracts.state import WorldSnapshot
from engine.contracts.outcome import outcome_from_counts
from engine.battle_world import BattleWorld
from engine.units.unit_manager import UnitManager
from environment.action_controller import ActionController
from environment.observation import ObservationConfig, ObservationManager
from render.api import create_renderer, create_video_writer
from environment.reward import RewardManager, default_reward_config, merge_reward_config
from environment.scenario import (
    ScenarioConfig,
    build_episode,
    create_scenario_map,
    default_positions,
    normalize_unit_types,
)

class JackalEnv:
    def __init__(
        self,
        headless=True,
        fixed_delta_time=0.03,
        use_video=False,
        video_dir="videos",
        auto_aim=True,
        map_name="border",
        map_file=None,
        map_data=None,
        game_state_file=None,
        map_tile_size=64,
        viewport_width=960,
        viewport_height=640,
        include_map_features=False,
        include_obs_map_features=None,
        include_state_map_features=None,
        obs_map_grid_size=5,
        obs_map_cell_size=64,
        state_map_grid_size=10,
        reward_config=None,
        n_agents=1,
        n_enemies=1,
        ally_positions=None,
        enemy_positions=None,
        max_steps=500,
        enemy_use_ai=True,
        ally_unit_types=None,
        enemy_unit_types=None,
        unit_type_names=None,
        include_unit_type_onehot=False,
        ally_unit_scales=None,
        enemy_unit_scales=None,
        ally_unit_type_scales=None,
        enemy_unit_type_scales=None,
        agent_fire_cooldown_max=None,
        enemy_ai_fire_cooldown_max=None,
        agent_fire_cooldown_by_type=None,
        enemy_ai_fire_cooldown_by_type=None,
        bullet_overrides_by_unit_type=None,
        enemy_ai_fire_angle_tolerance=None,
        auto_aim_fire_angle_tolerance=None,
        auto_aim_auto_fire_when_ready=False,
        ally_initial_headings=None,
        enemy_initial_headings=None,
        unit_sight_range=400.0,
        position_jitter=0.0,
        heading_jitter=0.0,
        enable_unit_collision=None,
        use_tear_drop_vision=None,
        auto_communicate=None,
        collision_scale=1.0,
        enemy_ai_intelligence_level=DEFAULT_AI_INTELLIGENCE_LEVEL,
        seed: Optional[int] = None,
        render_mode: str | None = None,
        render_options: dict | None = None,
    ):
        self.headless = headless
        self.delta_time = float(fixed_delta_time)
        if not math.isfinite(self.delta_time) or self.delta_time <= 0:
            raise ValueError("fixed_delta_time must be finite and > 0")
        try:
            max_steps = operator.index(max_steps)
        except TypeError as exc:
            raise ValueError("max_steps must be a positive integer") from exc
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self._rng = Random(None if seed is None else int(seed))
        self._episode_seed: Optional[int] = None
        
        self.use_video = use_video
        self.video_dir = video_dir
        self.video_writer = None
        self._closed = False
        self.auto_aim = auto_aim
        self.enable_unit_collision = bool(
            GameMode.ENABLE_UNIT_COLLISION
            if enable_unit_collision is None
            else enable_unit_collision
        )
        self.use_tear_drop_vision = bool(
            GameMode.USE_TEAR_DROP_VISION
            if use_tear_drop_vision is None
            else use_tear_drop_vision
        )
        self.auto_communicate = bool(
            GameMode.AUTO_COMMUNICATE
            if auto_communicate is None
            else auto_communicate
        )
        self.collision_scale = float(collision_scale)
        if not math.isfinite(self.collision_scale) or self.collision_scale <= 0.0:
            raise ValueError("collision_scale must be finite and > 0")
        self.include_obs_map_features = bool(include_map_features) if include_obs_map_features is None else bool(include_obs_map_features)
        self.include_state_map_features = bool(include_map_features) if include_state_map_features is None else bool(include_state_map_features)
        self.obs_map_grid_size = int(obs_map_grid_size)
        self.obs_map_cell_size = float(obs_map_cell_size)
        self.state_map_grid_size = int(state_map_grid_size)
        self.map_feature_dim = 4
        self.reward_config = default_reward_config()
        if reward_config:
            merge_reward_config(self.reward_config, reward_config)
        

        self.viewport_width = int(viewport_width)
        self.viewport_height = int(viewport_height)
        if self.viewport_width <= 0 or self.viewport_height <= 0:
            raise ValueError("viewport dimensions must be > 0")

        self.game_state_file = (
            None if game_state_file is None else str(game_state_file)
        )
        resolved_map_name = str(map_name or "border").lower()
        resolved_map_tile_size = int(map_tile_size)
        state_allies = []
        state_enemies = []
        if self.game_state_file:
            # The environment only asks the world boundary to restore state;
            # it never constructs maps, units, or bullets from the JSON itself.
            probe_world = BattleWorld(
                unit_manager=UnitManager(
                    enable_unit_collision=self.enable_unit_collision,
                    use_tear_drop_vision=self.use_tear_drop_vision,
                    auto_communicate=self.auto_communicate,
                )
            )
            probe_world.load_game_state(
                self.game_state_file,
                using_ai_by_team={
                    Team.PLAYER: False,
                    Team.ENEMY: bool(enemy_use_ai),
                },
                ai_intelligence_by_team={
                    Team.ENEMY: int(enemy_ai_intelligence_level),
                },
            )
            self.enable_unit_collision = (
                probe_world.unit_manager.enable_unit_collision
            )
            self.use_tear_drop_vision = (
                probe_world.unit_manager.use_tear_drop_vision
            )
            self.auto_communicate = (
                probe_world.unit_manager.auto_communicate_enabled
            )
            map_layout = probe_world.game_map
            state_allies = [
                unit
                for unit in probe_world.unit_manager.units
                if unit.team == Team.PLAYER
            ]
            state_enemies = [
                unit
                for unit in probe_world.unit_manager.units
                if unit.team == Team.ENEMY
            ]
        else:
            map_layout = create_scenario_map(
                map_name=resolved_map_name,
                map_file=map_file,
                map_data=map_data,
                map_tile_size=resolved_map_tile_size,
            )
        self.world_width, self.world_height = map_layout.get_map_size()
        if self.world_width <= 0 or self.world_height <= 0:
            raise ValueError("map dimensions must be > 0")
        random_builtin_map = (
            not self.game_state_file
            and map_file is None
            and not map_data
            and resolved_map_name.removesuffix("_map") == "random"
        )
        self._initial_map: Optional[GameMap] = (
            None
            if self.game_state_file or random_builtin_map
            else map_layout
        )
        self.render_mode = render_mode or ("rgb_array" if use_video else "human" if not headless else None)
        if self.render_mode and self.render_mode.upper() in {"2D", "3D", "2D3D"}:
            self.render_mode = self.render_mode.upper()
        browser_mode = self.render_mode in {"browser", "browser_lockstep", "3D", "2D3D"}
        self.n_agents = len(state_allies) if self.game_state_file else int(n_agents)
        self.n_enemies = len(state_enemies) if self.game_state_file else int(n_enemies)
        if self.n_agents <= 0:
            raise ValueError("n_agents must be >= 1")
        if self.n_enemies <= 0:
            raise ValueError("n_enemies must be >= 1")
        self.enemy_ai_intelligence_level = int(enemy_ai_intelligence_level)
        if not 1 <= self.enemy_ai_intelligence_level <= 9:
            raise ValueError("enemy_ai_intelligence_level must be between 1 and 9")

        if self.game_state_file:
            ally_positions = [tuple(unit.position) for unit in state_allies]
            enemy_positions = [tuple(unit.position) for unit in state_enemies]
            ally_unit_types = [unit.unit_type for unit in state_allies]
            enemy_unit_types = [unit.unit_type for unit in state_enemies]
        else:
            ally_positions = (
                [tuple(pos) for pos in ally_positions]
                if ally_positions is not None
                else default_positions(self.n_agents, enemy=False)
            )
            enemy_positions = (
                [tuple(pos) for pos in enemy_positions]
                if enemy_positions is not None
                else default_positions(self.n_enemies, enemy=True)
            )
        self.max_steps = int(max_steps)

        if unit_type_names is not None:
            self.unit_type_names = list(unit_type_names)
        elif self.game_state_file:
            self.unit_type_names = list(
                dict.fromkeys(
                    [unit.unit_type for unit in state_allies + state_enemies]
                )
            )
        else:
            self.unit_type_names = ["tank", "archie"]
        self.include_unit_type_onehot = bool(include_unit_type_onehot)
        self.unit_type_dim = len(self.unit_type_names) if self.include_unit_type_onehot else 0
        ally_unit_types = normalize_unit_types(
            ally_unit_types,
            self.n_agents,
            self.unit_type_names,
        )
        enemy_unit_types = normalize_unit_types(
            enemy_unit_types,
            self.n_enemies,
            self.unit_type_names,
        )

        self._world: Optional[BattleWorld] = None
        self._snapshot: Optional[WorldSnapshot] = None

        fire_cooldown = (
            float(agent_fire_cooldown_max)
            if agent_fire_cooldown_max is not None
            else 1.5
        )
        self.action_controller = ActionController(
            auto_aim=self.auto_aim,
            fire_angle_tolerance=auto_aim_fire_angle_tolerance,
            auto_fire_when_ready=auto_aim_auto_fire_when_ready,
            delta_time=self.delta_time,
        )
        self.n_actions = self.action_controller.n_actions
        self.scenario_config = ScenarioConfig(
            map_name=resolved_map_name,
            map_file=map_file,
            map_data=map_data,
            game_state_file=self.game_state_file,
            map_tile_size=resolved_map_tile_size,
            arena_size=(self.world_width, self.world_height),
            ally_positions=ally_positions,
            enemy_positions=enemy_positions,
            ally_unit_types=ally_unit_types,
            enemy_unit_types=enemy_unit_types,
            enemy_use_ai=bool(enemy_use_ai),
            enemy_ai_intelligence_level=self.enemy_ai_intelligence_level,
            ally_unit_scales=ally_unit_scales or {},
            enemy_unit_scales=enemy_unit_scales or {},
            ally_unit_type_scales=ally_unit_type_scales or {},
            enemy_unit_type_scales=enemy_unit_type_scales or {},
            agent_fire_cooldown=fire_cooldown,
            enemy_fire_cooldown=enemy_ai_fire_cooldown_max,
            agent_fire_cooldown_by_type=agent_fire_cooldown_by_type or {},
            enemy_fire_cooldown_by_type=enemy_ai_fire_cooldown_by_type or {},
            projectile_overrides_by_type=bullet_overrides_by_unit_type or {},
            enemy_fire_angle_tolerance=enemy_ai_fire_angle_tolerance,
            ally_initial_headings=ally_initial_headings or (),
            enemy_initial_headings=enemy_initial_headings or (),
            sight_range=float(unit_sight_range),
            position_jitter=float(position_jitter),
            heading_jitter=float(heading_jitter),
            collision_scale=self.collision_scale,
            enable_unit_collision=self.enable_unit_collision,
            use_tear_drop_vision=self.use_tear_drop_vision,
            auto_communicate=self.auto_communicate,
        )
        self.max_obs_bullets = 3
        self.max_state_bullets = 10
        self.obs_sight_range = (
            max(float(unit.sight_range) for unit in state_allies)
            if state_allies
            else float(unit_sight_range)
        )
        self.bullet_norm_speed = max(1.0, float(BULLET_SPEED))

        observation_config = ObservationConfig(
            world_width=float(self.world_width),
            world_height=float(self.world_height),
            n_agents=self.n_agents,
            n_enemies=self.n_enemies,
            unit_type_names=tuple(self.unit_type_names),
            include_unit_type_onehot=self.include_unit_type_onehot,
            max_obs_bullets=self.max_obs_bullets,
            max_state_bullets=self.max_state_bullets,
            sight_range=self.obs_sight_range,
            bullet_norm_speed=self.bullet_norm_speed,
            max_steps=self.max_steps,
            include_obs_map_features=self.include_obs_map_features,
            include_state_map_features=self.include_state_map_features,
            obs_map_grid_size=self.obs_map_grid_size,
            obs_map_cell_size=self.obs_map_cell_size,
            state_map_grid_size=self.state_map_grid_size,
            map_feature_dim=self.map_feature_dim,
        )
        self.observation_manager = ObservationManager(observation_config)
        self.reward_manager = RewardManager(
            auto_aim=self.auto_aim,
            max_steps=self.max_steps,
            arena_size=(float(self.world_width), float(self.world_height)),
            config=self.reward_config,
        )
        
        if self.use_video:
            os.makedirs(self.video_dir, exist_ok=True)

        # Open optional windows only after all constructor validation succeeds.
        self.renderer = create_renderer(self.render_mode, self.viewport_width,
                                        self.viewport_height, **({"max_steps": max_steps, **(render_options or {})} if browser_mode else (render_options or {})))

        self._video_renderer = self.renderer
        if use_video and browser_mode:
            try:
                self._video_renderer = create_renderer("rgb_array", self.viewport_width, self.viewport_height)
            except Exception:
                if self.renderer is not None:
                    self.renderer.close()
                raise


    @property
    def world(self) -> BattleWorld:
        if self._world is None:
            raise RuntimeError("Environment must be reset before accessing the battle world")
        return self._world

    @property
    def game_map(self) -> GameMap:
        """Compatibility access for existing integrations; prefer environment APIs."""

        return self.world.game_map

    @property
    def unit_manager(self) -> UnitManager:
        """Compatibility access for existing integrations; prefer environment APIs."""

        return self.world.unit_manager

    @property
    def bullet_manager(self) -> BulletManager:
        """Compatibility access for existing integrations; prefer environment APIs."""

        return self.world.bullet_manager

    @property
    def snapshot(self) -> WorldSnapshot:
        if self._snapshot is None:
            raise RuntimeError("Environment must be reset before accessing a snapshot")
        return self._snapshot

    def set_reward_config(self, reward_config):
        if reward_config:
            merge_reward_config(self.reward_config, reward_config)

    def reset(self, seed: Optional[int] = None):
        """Start an episode, optionally reproducing it from an exact seed."""

        self._require_open()
        if seed is not None:
            actual_seed = int(seed)
            self._rng.seed(actual_seed)
        else:
            actual_seed = self._rng.getrandbits(63)
        self._episode_seed = actual_seed
        episode_rng = Random(actual_seed)

        self.steps = 0
        self.reward_manager.reset()
        initial_map = self._initial_map
        self._initial_map = None
        episode = build_episode(
            self.scenario_config,
            game_map=initial_map,
            rng=episode_rng,
        )
        self._world = episode.world
        if self.renderer is not None:
            self.renderer.initialize(self._world.render_manifest(self.delta_time))
        if self._video_renderer is not None and self._video_renderer is not self.renderer:
            self._video_renderer.initialize(self._world.render_manifest(self.delta_time))
        self.agents = episode.allies
        self.enemies = episode.enemies
        self._snapshot = episode.world.snapshot()
        if self.render_mode in {"human", "browser", "browser_lockstep", "2D", "3D", "2D3D"}:
            self.render()
            
        if self.use_video:
            if self.video_writer is not None:
                self.video_writer.release()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            video_path = os.path.join(self.video_dir, f"episode_{timestamp}.mp4")
            fps = 1.0 / self.delta_time
            self.video_writer = create_video_writer(
                video_path,
                fps,
                (self.viewport_width, self.viewport_height),
            )
            self._render_to_video()
            
        return (
            self.observation_manager.get_observations(self.snapshot),
            self.observation_manager.get_state(self.snapshot),
        )

    def get_avail_agent_actions(self, agent_id):
        return self.action_controller.available_actions(
            self.world,
            self.snapshot,
            agent_id,
        )
        
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def step(self, actions):
        self._require_open()
        self.action_controller.apply(
            self.world,
            self.snapshot,
            actions,
        )
        self.steps += 1

        pre_snapshot = self.snapshot

        # --- 2. 物理更新：由游戏层统一编排地图、AI/单位、子弹和视野 ---
        self.world.step(self.delta_time)
        
        # --- 3. 视频录制 ---
        if self.use_video:
            self._render_to_video()

        self._snapshot = self.world.snapshot()
        reward, info = self.reward_manager.calculate(
            pre_snapshot,
            self.snapshot,
            actions,
        )
        info["blocked_by_unit"] = tuple(
            bool(agent.blocked_by_unit) for agent in self.snapshot.allies
        )
        info["unit_collision_count"] = tuple(
            int(agent.unit_collision_count) for agent in self.snapshot.allies
        )
        info["episode_seed"] = self._episode_seed
        done = self._check_done()
        if self.render_mode in {"human", "browser", "browser_lockstep", "2D", "3D", "2D3D"}:
            self.render()
        
        return (
            self.observation_manager.get_observations(self.snapshot),
            self.observation_manager.get_state(self.snapshot),
            reward,
            done,
            info,
        )
    
    def _check_done(self):
        return self.get_battle_result()['finished']

    def get_battle_result(self) -> dict:
        """Return episode-relative termination without exposing engine objects."""
        self._require_open()
        counts = {'PLAYER': sum(unit.alive for unit in self.snapshot.allies),
                  'ENEMY': sum(unit.alive for unit in self.snapshot.enemies)}
        return outcome_from_counts(counts, self.steps, self.max_steps)

    def get_env_info(self):
        """Return stable metadata without exposing environment components."""

        return {
            "n_agents": self.n_agents,
            "n_enemies": self.n_enemies,
            "n_actions": self.n_actions,
            "state_shape": self.observation_manager.state_dim(),
            "obs_shape": self.observation_manager.observation_dim(),
            "episode_limit": self.max_steps,
            "unit_type_dim": self.unit_type_dim,
            "obs_map_dim": self.observation_manager.observation_map_dim(),
            "state_map_dim": self.observation_manager.state_map_dim(),
            "world_width": self.world_width,
            "world_height": self.world_height,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "enable_unit_collision": self.enable_unit_collision,
            "use_tear_drop_vision": self.use_tear_drop_vision,
            "auto_communicate": self.auto_communicate,
        }

    def get_runtime_info(self):
        """Return lightweight episode diagnostics through the environment API."""

        return {
            "steps": self.steps,
            "world_tick": self.snapshot.tick,
            "active_allies": sum(unit.alive for unit in self.snapshot.allies),
            "active_enemies": sum(unit.alive for unit in self.snapshot.enemies),
            "active_bullets": sum(bullet.active for bullet in self.snapshot.bullets),
            "blocked_by_unit": tuple(
                bool(agent.blocked_by_unit) for agent in self.snapshot.allies
            ),
            "unit_collision_count": tuple(
                int(agent.unit_collision_count) for agent in self.snapshot.allies
            ),
            "episode_seed": self._episode_seed,
        }

    def get_obs(self):
        """Return local observations through the stable environment API."""
        return self.observation_manager.get_observations(self.snapshot)

    def get_state(self):
        """Return centralized state through the stable environment API."""
        return self.observation_manager.get_state(self.snapshot)

    def _render_to_video(self):
        if self.video_writer is None or self._video_renderer is None:
            return
        self._video_renderer.draw(self.world.render_snapshot())
        self.video_writer.write(self._video_renderer.rgb_array())

    def render(self):
        """Render the current immutable snapshot through the selected unified view."""

        self._require_open()
        if self.renderer is None:
            return None
        return self.renderer.render(self.world.render_snapshot(), self.get_battle_result())

    def _require_open(self):
        if self._closed:
            raise RuntimeError('JackalEnv is closed')

    def close(self):
        if self._closed:
            return
        self._closed = True
        from contextlib import ExitStack
        # ExitStack still releases the other owned resources if one fails.
        with ExitStack() as cleanup:
            if self._video_renderer is not None and self._video_renderer is not self.renderer:
                cleanup.callback(self._video_renderer.close)
            if self.renderer is not None:
                cleanup.callback(self.renderer.close)
            if self.video_writer is not None:
                writer, self.video_writer = self.video_writer, None
                cleanup.callback(writer.release)
