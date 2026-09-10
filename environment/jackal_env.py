import pygame
import os
import random
import numpy as np
import math
import cv2
import datetime

from game.Parameter import BULLET_DAMAGE, BULLET_SPEED, Team, UNIT_MIN_SIGHT_RATIO
from game.Unit.Tank.Tank import create_tank, create_enemy_tank
from game.Unit.Archie.Archie import create_archie, create_enemy_archie
from game.Map.GameMap import (
    create_border_map,
    create_empty_map,
    create_maze_map,
    create_random_map,
    create_valley_map,
    create_spindle_map,
    create_corridor_map,
    create_map_from_file,
    create_map_from_strings,
)
from game.Bullet.BulletManager import BulletManager

# 引入单位管理器
from game.Unit.UnitManager import UnitManager
# from game.GameMode import Team


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
        map_tile_size=64,
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
    ):
        self.headless = headless
        self.delta_time = fixed_delta_time
        
        self.use_video = use_video
        self.video_dir = video_dir
        self.video_writer = None
        self.has_enemy_kill = False
        self.episode_reward_so_far = 0.0

        self.auto_aim = auto_aim
        self.map_name = str(map_name or "border").lower()
        self.map_file = map_file
        self.map_data = map_data
        self.map_tile_size = int(map_tile_size)
        self.include_obs_map_features = bool(include_map_features) if include_obs_map_features is None else bool(include_obs_map_features)
        self.include_state_map_features = bool(include_map_features) if include_state_map_features is None else bool(include_state_map_features)
        self.obs_map_grid_size = int(obs_map_grid_size)
        self.obs_map_cell_size = float(obs_map_cell_size)
        self.state_map_grid_size = int(state_map_grid_size)
        self.map_feature_dim = 4
        self.reward_config = {
            "auto_aim": {
                "enemy_limit_scale": 0.1,
                "agent_limit_scale": 0.1,
                "enemy_kill_bonus": 20.0,
                "agent_killed_penalty": 10.0,
                "approach_scale": 6.0,
                "alive_adv_scale": 8.0,
                "step_penalty": 0.01,
                "reward_clip_abs": 5.0,
                "win_bonus": 30.0,
                "fast_win_bonus": 0.0,
                "fast_win_reference_steps": 0,
                "lose_penalty": 30.0,
                "timeout_penalty": 30.0,
                "no_kill_timeout_penalty": 0.0,
                "timeout_return_penalty_scale": 0.0,
            },
            "manual_aim": {
                "enemy_limit_scale": 0.12,
                "agent_limit_scale": 0.12,
                "enemy_kill_bonus": 20.0,
                "agent_killed_penalty": 12.0,
                "aim_good_angle": 8.0,
                "aim_ok_angle": 20.0,
                "aim_good_reward": 0.04,
                "aim_ok_reward": 0.015,
                "aim_bad_penalty": 0.01,
                "fire_good_angle": 12.0,
                "fire_good_reward": 0.08,
                "fire_bad_penalty": 0.06,
                "step_penalty": 0.01,
                "fire_action_id": 27,
                "win_bonus": 30.0,
                "fast_win_bonus": 0.0,
                "fast_win_reference_steps": 0,
                "lose_penalty": 30.0,
                "timeout_penalty": 30.0,
                "no_kill_timeout_penalty": 0.0,
                "timeout_return_penalty_scale": 0.0,
            }
        }
        if reward_config:
            self._deep_update(self.reward_config, reward_config)
        
        if self.headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
            
        pygame.init()
        
        self.screen_width, self.screen_height = 960, 640
        
        if not self.headless:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Jackal MARL Environment")
        else:
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.n_agents = int(n_agents)
        self.n_enemies = int(n_enemies)
        if self.n_agents <= 0:
            raise ValueError("n_agents must be >= 1")
        if self.n_enemies <= 0:
            raise ValueError("n_enemies must be >= 1")

        self.ally_positions = (
            [tuple(pos) for pos in ally_positions]
            if ally_positions is not None
            else self._build_default_positions(self.n_agents, is_enemy=False)
        )
        self.enemy_positions = (
            [tuple(pos) for pos in enemy_positions]
            if enemy_positions is not None
            else self._build_default_positions(self.n_enemies, is_enemy=True)
        )
        self.max_steps = int(max_steps)
        self.enemy_use_ai = bool(enemy_use_ai)

        self.unit_type_names = list(unit_type_names) if unit_type_names is not None else ["tank", "archie"]
        self.include_unit_type_onehot = bool(include_unit_type_onehot)
        self.unit_type_dim = len(self.unit_type_names) if self.include_unit_type_onehot else 0
        self.ally_unit_types = self._normalize_unit_types(ally_unit_types, self.n_agents)
        self.enemy_unit_types = self._normalize_unit_types(enemy_unit_types, self.n_enemies)

        # 单位属性缩放接口（用于任务难度调节/课程学习）
        self.ally_unit_scales = ally_unit_scales or {}
        self.enemy_unit_scales = enemy_unit_scales or {}
        self.ally_unit_type_scales = ally_unit_type_scales or {}
        self.enemy_unit_type_scales = enemy_unit_type_scales or {}

        # 开火参数接口（默认保持原行为）
        self.agent_fire_cooldown_max = agent_fire_cooldown_max
        self.enemy_ai_fire_cooldown_max = enemy_ai_fire_cooldown_max
        self.agent_fire_cooldown_by_type = agent_fire_cooldown_by_type or {}
        self.enemy_ai_fire_cooldown_by_type = enemy_ai_fire_cooldown_by_type or {}
        self.bullet_overrides_by_unit_type = bullet_overrides_by_unit_type or {}
        self.enemy_ai_fire_angle_tolerance = enemy_ai_fire_angle_tolerance
        self.auto_aim_fire_angle_tolerance = auto_aim_fire_angle_tolerance
        self.auto_aim_auto_fire_when_ready = bool(auto_aim_auto_fire_when_ready)
        self.ally_initial_headings = ally_initial_headings or []
        self.enemy_initial_headings = enemy_initial_headings or []
        self.unit_sight_range = float(unit_sight_range)
        self.position_jitter = float(position_jitter)
        self.heading_jitter = float(heading_jitter)
        
        self.fire_cooldown_max = (
            float(agent_fire_cooldown_max)
            if agent_fire_cooldown_max is not None
            else 1.5
        )
        self.agent_fire_cooldowns = {i: 0.0 for i in range(self.n_agents)}
        self.max_obs_bullets = 3
        self.max_state_bullets = 10
        self.obs_sight_range = self.unit_sight_range
        self.bullet_norm_speed = max(1.0, float(BULLET_SPEED))
        self.normal_shell_range = float(BULLET_SPEED) * 1.2
        
        if self.use_video:
            os.makedirs(self.video_dir, exist_ok=True)

    def _deep_update(self, base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _create_game_map(self):
        if self.map_data:
            return create_map_from_strings(self.map_data, tile_size=self.map_tile_size)
        if self.map_file:
            game_map = create_map_from_file(self.map_file, tile_size=self.map_tile_size)
            if game_map is None:
                raise ValueError(f"Failed to load map_file: {self.map_file}")
            return game_map

        if self.map_name == "border":
            return create_border_map()
        if self.map_name == "empty":
            return create_empty_map()
        if self.map_name == "maze":
            return create_maze_map()
        if self.map_name == "valley":
            return create_valley_map()
        if self.map_name in ("spindle", "spindle_map"):
            return create_spindle_map()
        if self.map_name in ("corridor", "corridor_map"):
            return create_corridor_map()
        if self.map_name == "random":
            return create_random_map()

        raise ValueError(f"Unknown map_name: {self.map_name}")

    def set_reward_config(self, reward_config):
        if reward_config:
            self._deep_update(self.reward_config, reward_config)

    def _build_default_positions(self, count, is_enemy=False):
        """为多智能体任务构建可复现的默认出生点。"""
        x = 740 if is_enemy else 220
        start_y = 220
        spacing_y = 100
        return [(x, start_y + i * spacing_y) for i in range(count)]

    def _normalize_unit_types(self, unit_types, count):
        if unit_types is None:
            return ["tank"] * count
        types = [str(unit_type).lower() for unit_type in unit_types]
        if len(types) > count:
            types = types[:count]
        if len(types) < count:
            types.extend(["tank"] * (count - len(types)))
        for unit_type in types:
            if unit_type not in self.unit_type_names:
                raise ValueError(
                    f"Unknown unit_type={unit_type!r}; expected one of {self.unit_type_names}"
                )
        return types

    def _unit_type_onehot(self, unit_or_type):
        if not self.include_unit_type_onehot:
            return []
        unit_type = unit_or_type if isinstance(unit_or_type, str) else getattr(unit_or_type, "unit_type", "tank")
        onehot = [0.0] * self.unit_type_dim
        if unit_type in self.unit_type_names:
            onehot[self.unit_type_names.index(unit_type)] = 1.0
        return onehot

    def _create_unit_by_type(self, unit_type, unit_id, team, position, using_ai):
        unit_type = str(unit_type).lower()
        if unit_type == "tank":
            if team == Team.ENEMY:
                return create_enemy_tank(unit_id, position=position, usingAI=using_ai)
            return create_tank(unit_id, team, position=position, usingAI=using_ai)
        if unit_type == "archie":
            if team == Team.ENEMY:
                return create_enemy_archie(unit_id, position=position, usingAI=using_ai)
            return create_archie(unit_id, team, position=position, usingAI=using_ai)
        raise ValueError(f"Unsupported unit_type={unit_type!r}")

    def _unit_type_scale_cfg(self, side_cfg, unit_type):
        if not side_cfg:
            return {}
        return side_cfg.get(unit_type, side_cfg.get(str(unit_type).lower(), {})) or {}

    def _jitter_position(self, position):
        if self.position_jitter <= 0.0:
            return position
        x = float(position[0]) + random.uniform(-self.position_jitter, self.position_jitter)
        y = float(position[1]) + random.uniform(-self.position_jitter, self.position_jitter)
        margin = 50.0
        x = float(np.clip(x, margin, self.screen_width - margin))
        y = float(np.clip(y, margin, self.screen_height - margin))
        return (x, y)

    def _apply_initial_heading_jitter(self, unit):
        if self.heading_jitter <= 0.0:
            return
        delta = random.uniform(-self.heading_jitter, self.heading_jitter)
        unit.direction_angle = unit.normalize_angle(unit.direction_angle + delta)
        unit.turret_direction_angle = unit.normalize_angle(unit.turret_direction_angle + delta)
        unit.turret_target_angle = unit.turret_direction_angle
        unit.velocity = unit.cal_velocity()
        unit._update_bounding_box()

    def _initial_heading(self, headings, idx):
        if idx >= len(headings):
            return None
        return float(headings[idx])

    def _set_unit_heading(self, unit, heading):
        heading = unit.normalize_angle(float(heading))
        unit.direction_angle = heading
        unit.turret_direction_angle = heading
        unit.turret_target_angle = heading
        unit.velocity = unit.cal_velocity()
        unit._update_bounding_box()

    def _apply_unit_scales(self, unit, scale_cfg):
        """按配置缩放单位关键属性；缺省键不修改。"""
        if not scale_cfg:
            return

        speed_scale = float(scale_cfg.get("speed", 1.0))
        acc_scale = float(scale_cfg.get("acceleration", 1.0))
        turn_scale = float(scale_cfg.get("turn", 1.0))
        turret_turn_scale = float(scale_cfg.get("turret_turn", 1.0))
        health_scale = float(scale_cfg.get("health", 1.0))

        unit.max_speed *= speed_scale
        unit.max_acceleration *= acc_scale
        unit.min_acceleration *= acc_scale
        unit.max_angular_speed *= turn_scale
        unit.turret_angular_speed *= turret_turn_scale
        unit.max_health *= health_scale
        unit.health = min(unit.health * health_scale, unit.max_health)

    def _apply_unit_sight_range(self, unit):
        unit.sight_range = self.unit_sight_range
        unit.min_sight_range = UNIT_MIN_SIGHT_RATIO * unit.sight_range
        unit.communication_range = unit.sight_range

    def _agent_fire_cooldown_max(self, agent):
        unit_type = str(getattr(agent, "unit_type", "tank")).lower()
        if unit_type in self.agent_fire_cooldown_by_type:
            return float(self.agent_fire_cooldown_by_type[unit_type])
        return self.fire_cooldown_max

    def _enemy_ai_fire_cooldown_max(self, enemy):
        unit_type = str(getattr(enemy, "unit_type", "tank")).lower()
        if unit_type in self.enemy_ai_fire_cooldown_by_type:
            return float(self.enemy_ai_fire_cooldown_by_type[unit_type])
        if self.enemy_ai_fire_cooldown_max is not None:
            return float(self.enemy_ai_fire_cooldown_max)
        return None

    def _cooldown_ratio(self, unit, cooldown):
        denom = max(1e-6, self._agent_fire_cooldown_max(unit))
        return float(cooldown) / denom

    def _apply_bullet_overrides(self, shooter, bullet):
        if bullet is None:
            return
        unit_type = str(getattr(shooter, "unit_type", "tank")).lower()
        overrides = self.bullet_overrides_by_unit_type.get(unit_type, {})
        if not overrides:
            return
        ammo_name = str(getattr(shooter, "current_ammunition", "")).lower()
        ammo_overrides = overrides.get(ammo_name, overrides.get("default", overrides))
        if not isinstance(ammo_overrides, dict):
            return
        for key, value in ammo_overrides.items():
            if key in ("speed_rate", "speed"):
                continue
            setattr(bullet, key, value)
        if "damage_rate" in ammo_overrides:
            bullet.base_damage = BULLET_DAMAGE * float(bullet.damage_rate)

    def _attach_unit_bullet_overrides(self, unit):
        unit_type = str(getattr(unit, "unit_type", "tank")).lower()
        overrides = self.bullet_overrides_by_unit_type.get(unit_type, {})
        if overrides:
            unit.bullet_overrides = overrides

    def _fire_agent_weapon(self, agent_id, agent):
        bullet = agent.fire()
        if not bullet:
            return None
        self._apply_bullet_overrides(agent, bullet)
        self.bullet_manager.add_bullet(bullet)
        cooldown = self._agent_fire_cooldown_max(agent)
        agent.fire_cooldown = cooldown
        self.agent_fire_cooldowns[agent_id] = cooldown
        return bullet

    def _unit_weapon_range(self, unit):
        ammo_name = str(getattr(unit, "current_ammunition", "")).lower()
        if ammo_name == "rocket_shell":
            return float(BULLET_SPEED) * 1.2 * 1.5
        if ammo_name == "heavy_shell":
            return float(BULLET_SPEED) * 0.8 * 1.8
        return self.normal_shell_range

    def reset(self):
        self.steps = 0
        self.has_enemy_kill = False
        self.episode_reward_so_far = 0.0
        self.agent_fire_cooldowns = {i: 0.0 for i in range(self.n_agents)}
        self.game_map = self._create_game_map()
        self.bullet_manager = BulletManager()
        self.unit_manager = UnitManager()  # 实例化新的 UnitManager
        
        self.agents = []
        # 创建玩家: RL网络控制，不使用内置AI (usingAI=False)
        for i in range(self.n_agents):
            pos = self.ally_positions[i] if i < len(self.ally_positions) else (220, 220 + i * 100)
            pos = self._jitter_position(pos)
            unit_type = self.ally_unit_types[i]
            player = self._create_unit_by_type(unit_type, i + 1, Team.PLAYER, pos, using_ai=False)
            player.usingAI = False
            self._apply_unit_sight_range(player)
            self._apply_unit_scales(player, self.ally_unit_scales)
            self._apply_unit_scales(player, self._unit_type_scale_cfg(self.ally_unit_type_scales, unit_type))
            self._attach_unit_bullet_overrides(player)
            initial_heading = self._initial_heading(self.ally_initial_headings, i)
            if initial_heading is not None:
                self._set_unit_heading(player, initial_heading)
            self._apply_initial_heading_jitter(player)
            self.unit_manager.add_unit(player, self.bullet_manager, self.game_map)
            self.agents.append(player)
        
        self.enemies = []
        for i in range(self.n_enemies):
            # 安全读取坐标，防止配置坐标数量不足
            pos = self.enemy_positions[i] if i < len(self.enemy_positions) else (740, 220 + i * 100)
            pos = self._jitter_position(pos)

            # 创建敌人: 开启内置AI控制
            unit_type = self.enemy_unit_types[i]
            enemy = self._create_unit_by_type(unit_type, 100 + i, Team.ENEMY, pos, using_ai=self.enemy_use_ai)
            enemy.usingAI = self.enemy_use_ai
            self._apply_unit_sight_range(enemy)
            self._apply_unit_scales(enemy, self.enemy_unit_scales)
            self._apply_unit_scales(enemy, self._unit_type_scale_cfg(self.enemy_unit_type_scales, unit_type))
            self._attach_unit_bullet_overrides(enemy)
            initial_heading = self._initial_heading(self.enemy_initial_headings, i)
            if initial_heading is not None:
                self._set_unit_heading(enemy, initial_heading)
            self._apply_initial_heading_jitter(enemy)

            enemy_ai_fire_cooldown = self._enemy_ai_fire_cooldown_max(enemy)
            if enemy_ai_fire_cooldown is not None:
                enemy.ai_fire_cooldown_max = float(enemy_ai_fire_cooldown)
            if self.enemy_ai_fire_angle_tolerance is not None:
                enemy.ai_fire_angle_tolerance = float(self.enemy_ai_fire_angle_tolerance)

            self.unit_manager.add_unit(enemy, self.bullet_manager, self.game_map)
            self.enemies.append(enemy)

        self._refresh_all_vision()
            
        if self.use_video:
            if self.video_writer is not None:
                self.video_writer.release()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(self.video_dir, f"episode_{timestamp}.mp4")
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            fps = int(1.0 / self.delta_time)
            self.video_writer = cv2.VideoWriter(video_path, fourcc, fps, (self.screen_width, self.screen_height))
            self._render_to_video()
            
        return self.get_obs(), self.get_state()

    def _refresh_all_vision(self):
        """刷新所有单位的初始/当前视野，不推进物理和 AI。"""
        for unit in self.unit_manager.units:
            if unit is not None and unit.is_alive:
                unit._update_vision(self.unit_manager, self.bullet_manager, self.game_map)

    def _obs_dim(self):
        return (
            (10 + self.unit_type_dim)
            + (self.n_agents - 1) * (11 + self.unit_type_dim)
            + self.n_enemies * (13 + self.unit_type_dim)
            + self.max_obs_bullets * 9
            + self._obs_map_dim()
            + 1
        )

    def _obs_map_dim(self):
        if not self.include_obs_map_features:
            return 0
        return self.obs_map_grid_size * self.obs_map_grid_size * self.map_feature_dim

    def _state_map_dim(self):
        if not self.include_state_map_features:
            return 0
        return self.state_map_grid_size * self.state_map_grid_size * self.map_feature_dim

    def _terrain_features_at(self, x, y):
        if self.game_map is None:
            return [0.0] * self.map_feature_dim
        col = int(float(x) // self.game_map.tile_size)
        row = int(float(y) // self.game_map.tile_size)
        if row < 0 or row >= self.game_map.height or col < 0 or col >= self.game_map.width:
            return [0.0, 0.0, 0.0, 0.0]

        tile = self.game_map.tiles[row][col]
        blocks_unit = 1.0 if getattr(tile, "blocks_unit", False) else 0.0
        blocks_bullet = 1.0 if getattr(tile, "blocks_bullet", False) else 0.0
        water = 1.0 if getattr(tile, "letter", "") == "w" else 0.0
        return [1.0, blocks_unit, blocks_bullet, water]

    def _local_map_features(self, agent):
        if not self.include_obs_map_features:
            return []

        grid = self.obs_map_grid_size
        half = grid // 2
        cell = self.obs_map_cell_size
        features = []
        base_x, base_y = agent.position
        for gy in range(grid):
            for gx in range(grid):
                sample_x = base_x + (gx - half) * cell
                sample_y = base_y + (gy - half) * cell
                features.extend(self._terrain_features_at(sample_x, sample_y))
        return features

    def _global_map_features(self):
        if not self.include_state_map_features:
            return []

        grid = self.state_map_grid_size
        features = []
        width = max(1.0, float(self.screen_width))
        height = max(1.0, float(self.screen_height))
        for gy in range(grid):
            sample_y = (gy + 0.5) * height / grid
            for gx in range(grid):
                sample_x = (gx + 0.5) * width / grid
                features.extend(self._terrain_features_at(sample_x, sample_y))
        return features

    def _unit_speed_norm(self, unit):
        return float(unit.speed) / max(1.0, float(unit.max_speed))

    def _unit_angular_norm(self, unit):
        return float(unit.angular_speed) / max(1.0, float(unit.max_angular_speed))

    def _unit_accel_norm(self, unit):
        denom = max(1.0, abs(float(unit.max_acceleration)), abs(float(unit.min_acceleration)))
        return float(unit.acceleration) / denom

    def _distance_between(self, src, dst):
        return math.hypot(dst.position[0] - src.position[0], dst.position[1] - src.position[1])

    def _sorted_observable_units(self, observer, units):
        def sort_key(unit):
            alive = bool(getattr(unit, "is_alive", False))
            visible = alive and self.is_visible_to_agent(observer, unit)
            if visible:
                group = 0
            elif alive:
                group = 1
            else:
                group = 2
            dist = self._distance_between(observer, unit) if alive else float("inf")
            return (group, dist, getattr(unit, "id", 0))

        return sorted(units, key=sort_key)

    def _has_auto_aim_fire_target(self, agent):
        fire_angle_tolerance = self.auto_aim_fire_angle_tolerance
        for enemy in self.enemies:
            if not enemy.is_alive or not self.is_visible_to_agent(agent, enemy):
                continue
            dist = self._distance_between(agent, enemy)
            line_of_fire, in_range, _ = self._target_geometry_features(agent, enemy, dist)
            if line_of_fire > 0.0 and in_range > 0.0:
                if fire_angle_tolerance is not None:
                    dx = enemy.position[0] - agent.position[0]
                    dy = enemy.position[1] - agent.position[1]
                    target_angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
                    angle_diff = abs(agent.get_angle_difference(agent.turret_direction_angle, target_angle))
                    if angle_diff > float(fire_angle_tolerance):
                        continue
                return True
        return False

    def _target_geometry_features(self, observer, target, dist):
        dx = target.position[0] - observer.position[0]
        dy = target.position[1] - observer.position[1]
        target_angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
        turret_diff = abs(observer.get_angle_difference(observer.turret_direction_angle, target_angle))
        turret_alignment = 1.0 - min(turret_diff, 180.0) / 180.0
        line_of_fire = 1.0 if self.check_raycast_unblocked(observer, target) else 0.0
        in_range = 1.0 if dist <= self._unit_weapon_range(observer) else 0.0
        return line_of_fire, in_range, turret_alignment

    def _bullet_time_to_impact(self, agent, bullet):
        hostile = hasattr(bullet, "shooter_team") and bullet.shooter_team != agent.team
        if not hostile:
            return 1.0

        rel_x = agent.position[0] - bullet.position[0]
        rel_y = agent.position[1] - bullet.position[1]
        vel_x, vel_y = getattr(bullet, "velocity", (0.0, 0.0))
        speed_sq = vel_x * vel_x + vel_y * vel_y
        if speed_sq <= 1e-6:
            return 1.0

        closing = rel_x * vel_x + rel_y * vel_y
        if closing <= 0.0:
            return 1.0

        t_closest = closing / speed_sq
        closest_x = bullet.position[0] + vel_x * t_closest
        closest_y = bullet.position[1] + vel_y * t_closest
        miss_dist = math.hypot(agent.position[0] - closest_x, agent.position[1] - closest_y)
        unit_radius = math.hypot(agent.size[0], agent.size[1]) / 2.0
        if miss_dist > unit_radius * 1.5:
            return 1.0
        return float(np.clip(t_closest / 2.0, 0.0, 1.0))

    @property
    def n_actions(self):
        """
        动作空间动态调整:
        - auto_aim=True : 10 维 (9种机身走位 + 1种停步开火 炮塔自动追踪)
        - auto_aim=False: 28 维 (9种机身 * 3种炮塔旋转 + 1种停步开火)
        """
        return 10 if self.auto_aim else 28
    
    def get_avail_agent_actions(self, agent_id):
        avail_actions = [0] * self.n_actions
        agent = self.agents[agent_id]
        if not agent.is_alive:
            avail_actions[0] = 1 
            return avail_actions
        if self.auto_aim:
            # auto_aim: 0~8 为机动，9 为开火
            avail_actions[0:9] = [1] * 9
            if self.agent_fire_cooldowns[agent_id] <= 0 and self._has_auto_aim_fire_target(agent):
                avail_actions[9] = 1
        else:
            # manual_aim: 0~26 为机动+炮塔，27 为开火
            avail_actions[0:28] = [1] * 28
        return avail_actions
        
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def step(self, actions):
        self.steps += 1
        
        for i in range(self.n_agents):
            if self.agent_fire_cooldowns[i] > 0:
                self.agent_fire_cooldowns[i] -= self.delta_time

       # --- 1. 动作解析与执行 ---
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive: continue
                
            action = actions[agent_id]
            
            # 每次解析前先重置机身指令
            agent.set_movement(forward=False, backward=False)
            agent.set_turning(left=False, right=False)
            
            # 根据当前的瞄准模式，将动作分发给对应的成员函数处理
            if self.auto_aim:
                self._parse_action_auto_aim(agent_id, agent, action)
            else:
                self._parse_action_manual(agent_id, agent, action)

        # ==========================================
        # 1. 物理步进前：采集环境快照
        pre_stats = self._get_battle_stats()
        # ==========================================

        # --- 2. 物理更新 (完全适配新的 UnitManager 架构) ---
        # 现在所有单位更新和AI结算全部由 UnitManager 自动在内部循环完成
        self.unit_manager.update(self.delta_time, self.unit_manager, self.bullet_manager, self.game_map)
        self.bullet_manager.update(self.delta_time, self.unit_manager, self.game_map)
        self._refresh_all_vision()
        
        # --- 3. 视频录制 ---
        if self.use_video:
            self._render_to_video()

        # ==========================================
        # 2. 物理步进后：采集新快照并结算奖励
        post_stats = self._get_battle_stats()
        reward, info = self._calculate_reward(pre_stats, post_stats, actions)
        done = self._check_done()
        self.episode_reward_so_far += float(reward)
        # ==========================================
        
        return self.get_obs(), self.get_state(), reward, done, info
    
    def _check_done(self):
        if self.steps >= self.max_steps: return True
        if all(not agent.is_alive for agent in self.agents): return True
        if all(not enemy.is_alive for enemy in self.enemies): return True
        return False

    def check_raycast_unblocked(self, observer, target):
        """仅做纯粹的物理射线遮挡检测 (Raycasting)"""
        line_start = observer.position
        line_end = target.position
        for obstacle_rect in self.game_map.bullet_obstacles:
            if obstacle_rect.clipline(line_start, line_end):
                return False
        return True

    def is_visible_to_agent(self, agent, target):
        """
        结合底层水滴视野与环境层物理遮挡的综合判定
        """
        if not agent.is_alive or not target.is_alive:
            return False
        # 1. 底层查表：检查目标是否在原作者实现的可见列表中（包含水滴视野和距离限制）
        in_underlying_vision = any(u.id == target.id for u in agent.visible_units.units)
        if not in_underlying_vision:
            return False
        # 2. 环境层过滤：如果底层认为可见，叠加一次严格的物理防透视遮挡检测
        return self.check_raycast_unblocked(agent, target)

    def _get_battle_stats(self):
        """获取当前战局的统计信息快照"""
        enemy_alive = sum([1 for e in self.enemies if e.is_alive])
        agent_alive = sum([1 for a in self.agents if a.is_alive])

        alive_agents = [a for a in self.agents if a.is_alive]
        alive_enemies = [e for e in self.enemies if e.is_alive]
        if alive_agents and alive_enemies:
            nearest_sum = 0.0
            for agent in alive_agents:
                nearest = min(
                    math.hypot(enemy.position[0] - agent.position[0], enemy.position[1] - agent.position[1])
                    for enemy in alive_enemies
                )
                nearest_sum += nearest

            avg_nearest_enemy_dist = nearest_sum / len(alive_agents)
            norm_scale = max(1.0, math.hypot(self.screen_width, self.screen_height))
            avg_nearest_enemy_dist /= norm_scale
        else:
            avg_nearest_enemy_dist = 0.0

        alive_advantage = (agent_alive / max(1, self.n_agents)) - (enemy_alive / max(1, self.n_enemies))

        return {
            "enemy_health": sum([e.health for e in self.enemies]),
            "agent_health": sum([a.health for a in self.agents]),
            "enemy_alive": enemy_alive,
            "agent_alive": agent_alive,
            "avg_nearest_enemy_dist": avg_nearest_enemy_dist,
            "alive_advantage": alive_advantage,
        }

    def _timeout_return_penalty(self, current_step_reward, cfg):
        scale = float(cfg.get("timeout_return_penalty_scale", 0.0))
        if scale <= 0.0:
            return 0.0

        projected_return = self.episode_reward_so_far + float(current_step_reward)
        return scale * max(0.0, projected_return)

    def _fast_win_bonus(self, cfg):
        bonus = float(cfg.get("fast_win_bonus", 0.0))
        if bonus <= 0.0:
            return 0.0

        reference_steps = int(cfg.get("fast_win_reference_steps", 0))
        if reference_steps <= 0:
            reference_steps = self.max_steps
        return bonus * max(0.0, 1.0 - (self.steps / max(1, reference_steps)))

    def _calculate_reward(self, pre_stats, post_stats, actions):
        """
        根据当前瞄准模式自动切换奖励函数：
        - auto_aim=True  -> 偏向交战结果
        - auto_aim=False -> 增加手动瞄准过程奖励
        """
        if self.auto_aim:
            return self._calculate_reward_auto_aim(pre_stats, post_stats)
        return self._calculate_reward_manual_aim(pre_stats, post_stats, actions)

    def _calculate_reward_auto_aim(self, pre_stats, post_stats):
        cfg = self.reward_config["auto_aim"]
        enemy_limit_dealt = pre_stats["enemy_health"] - post_stats["enemy_health"]
        agent_limit_received = pre_stats["agent_health"] - post_stats["agent_health"]

        enemies_killed = pre_stats["enemy_alive"] - post_stats["enemy_alive"]
        agents_killed = pre_stats["agent_alive"] - post_stats["agent_alive"]
        if enemies_killed > 0:
            self.has_enemy_kill = True
        approach_progress = pre_stats.get("avg_nearest_enemy_dist", 0.0) - post_stats.get("avg_nearest_enemy_dist", 0.0)
        alive_adv_delta = post_stats.get("alive_advantage", 0.0) - pre_stats.get("alive_advantage", 0.0)

        reward = 0.0
        reward += (enemy_limit_dealt * float(cfg.get("enemy_limit_scale", 0.1)))
        reward -= (agent_limit_received * float(cfg.get("agent_limit_scale", 0.1)))
        reward += (enemies_killed * float(cfg.get("enemy_kill_bonus", 20.0)))
        reward -= (agents_killed * float(cfg.get("agent_killed_penalty", 10.0)))
        reward += (approach_progress * float(cfg.get("approach_scale", 0.0)))
        reward += (alive_adv_delta * float(cfg.get("alive_adv_scale", 0.0)))
        reward -= float(cfg.get("step_penalty", 0.0))

        info = {
            "battle_won": False,
            "reward_mode": "auto_aim",
            "approach_progress": round(float(approach_progress), 5),
            "alive_adv_delta": round(float(alive_adv_delta), 5),
            "no_kill_timeout": False,
            "episode_limit": False,
        }

        is_terminal = False
        if post_stats["enemy_alive"] == 0:
            reward += float(cfg.get("win_bonus", 30.0))
            reward += self._fast_win_bonus(cfg)
            info["battle_won"] = True
            is_terminal = True
        elif post_stats["agent_alive"] == 0:
            reward -= float(cfg.get("lose_penalty", 30.0))
            is_terminal = True
        elif self.steps >= self.max_steps:
            reward -= float(cfg.get("timeout_penalty", 30.0))
            if not self.has_enemy_kill:
                reward -= float(cfg.get("no_kill_timeout_penalty", 0.0))
                info["no_kill_timeout"] = True
            timeout_return_penalty = self._timeout_return_penalty(reward, cfg)
            if timeout_return_penalty > 0.0:
                reward -= timeout_return_penalty
                info["timeout_return_penalty"] = round(float(timeout_return_penalty), 5)
            info["episode_limit"] = True
            is_terminal = True

        if not is_terminal:
            clip_abs = float(cfg.get("reward_clip_abs", 0.0))
            if clip_abs > 0.0:
                reward = float(np.clip(reward, -clip_abs, clip_abs))

        return reward, info

    def _calculate_reward_manual_aim(self, pre_stats, post_stats, actions):
        cfg = self.reward_config["manual_aim"]
        enemy_limit_dealt = pre_stats["enemy_health"] - post_stats["enemy_health"]
        agent_limit_received = pre_stats["agent_health"] - post_stats["agent_health"]

        enemies_killed = pre_stats["enemy_alive"] - post_stats["enemy_alive"]
        agents_killed = pre_stats["agent_alive"] - post_stats["agent_alive"]
        if enemies_killed > 0:
            self.has_enemy_kill = True

        reward = 0.0
        reward += (enemy_limit_dealt * cfg["enemy_limit_scale"])
        reward -= (agent_limit_received * cfg["agent_limit_scale"])
        reward += (enemies_killed * cfg["enemy_kill_bonus"])
        reward -= (agents_killed * cfg["agent_killed_penalty"])

        aim_shaping = 0.0
        fire_shaping = 0.0
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive:
                continue

            visible_enemies = [
                enemy for enemy in self.enemies
                if enemy.is_alive and self.is_visible_to_agent(agent, enemy)
            ]
            if not visible_enemies:
                continue

            closest_enemy = min(
                visible_enemies,
                key=lambda enemy: math.hypot(enemy.position[0] - agent.position[0], enemy.position[1] - agent.position[1])
            )

            dx = closest_enemy.position[0] - agent.position[0]
            dy = closest_enemy.position[1] - agent.position[1]
            target_angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
            angle_diff = abs(agent.get_angle_difference(agent.turret_direction_angle, target_angle))

            if angle_diff <= cfg["aim_good_angle"]:
                aim_shaping += cfg["aim_good_reward"]
            elif angle_diff <= cfg["aim_ok_angle"]:
                aim_shaping += cfg["aim_ok_reward"]
            else:
                aim_shaping -= cfg["aim_bad_penalty"]

            if agent_id < len(actions) and actions[agent_id] == cfg["fire_action_id"]:
                if angle_diff <= cfg["fire_good_angle"]:
                    fire_shaping += cfg["fire_good_reward"]
                else:
                    fire_shaping -= cfg["fire_bad_penalty"]

        reward += aim_shaping
        reward += fire_shaping
        reward -= cfg["step_penalty"]

        info = {
            "battle_won": False,
            "reward_mode": "manual_aim",
            "aim_shaping": round(float(aim_shaping), 4),
            "fire_shaping": round(float(fire_shaping), 4),
            "no_kill_timeout": False,
            "episode_limit": False,
        }
        if post_stats["enemy_alive"] == 0:
            reward += cfg["win_bonus"]
            reward += self._fast_win_bonus(cfg)
            info["battle_won"] = True
        elif post_stats["agent_alive"] == 0:
            reward -= cfg["lose_penalty"]
        elif self.steps >= self.max_steps:
            reward -= cfg["timeout_penalty"]
            if not self.has_enemy_kill:
                reward -= cfg.get("no_kill_timeout_penalty", 0.0)
                info["no_kill_timeout"] = True
            timeout_return_penalty = self._timeout_return_penalty(reward, cfg)
            if timeout_return_penalty > 0.0:
                reward -= timeout_return_penalty
                info["timeout_return_penalty"] = round(float(timeout_return_penalty), 5)
            info["episode_limit"] = True

        return reward, info

    def get_obs(self):
        """
        获取所有智能体的局部观测 (Observation)
        包含自身、友军、敌军、视野内最近若干子弹以及时间进度。
        """
        obs_list = []
        sight_range = self.obs_sight_range
        time_ratio = self.steps / self.max_steps
        
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive:
                obs_list.append(np.zeros(self._obs_dim(), dtype=np.float32))
                continue
                
            obs_features = []
            # --- 1. 自身特征 (10维 + 可选单位类型one-hot) ---
            norm_x = agent.position[0] / self.screen_width
            norm_y = agent.position[1] / self.screen_height
            hp_ratio = agent.health / agent.max_health
            cos_dir = math.cos(math.radians(agent.direction_angle))
            sin_dir = math.sin(math.radians(agent.direction_angle))
            cos_turret = math.cos(math.radians(agent.turret_direction_angle))
            sin_turret = math.sin(math.radians(agent.turret_direction_angle))
            cooldown_ratio = self._cooldown_ratio(agent, self.agent_fire_cooldowns[agent_id])
            speed_norm = self._unit_speed_norm(agent)
            angular_norm = self._unit_angular_norm(agent)
            
            obs_features.extend([
                norm_x,
                norm_y,
                hp_ratio,
                cos_dir,
                sin_dir,
                cos_turret,
                sin_turret,
                cooldown_ratio,
                speed_norm,
                angular_norm,
            ])
            obs_features.extend(self._unit_type_onehot(agent))
            
            # --- 2. 友军特征 (每个友军11维 + 可选单位类型one-hot) ---
            ally_units = [other for other_id, other in enumerate(self.agents) if other_id != agent_id]
            for other_agent in ally_units:
                other_id = self.agents.index(other_agent)
                other_alive = bool(other_agent.is_alive)
                other_visible = other_alive and self.is_visible_to_agent(agent, other_agent)
                if other_visible:
                    rel_x = other_agent.position[0] - agent.position[0]
                    rel_y = other_agent.position[1] - agent.position[1]
                    dist = math.hypot(rel_x, rel_y) / sight_range
                    other_hp = other_agent.health / other_agent.max_health
                    other_cooldown = self._cooldown_ratio(other_agent, self.agent_fire_cooldowns[other_id])
                    obs_features.extend([
                        1.0,
                        1.0,
                        rel_x / sight_range,
                        rel_y / sight_range,
                        dist,
                        other_hp,
                        math.cos(math.radians(other_agent.direction_angle)),
                        math.sin(math.radians(other_agent.direction_angle)),
                        self._unit_speed_norm(other_agent),
                        other_cooldown,
                        1.0 if self.check_raycast_unblocked(agent, other_agent) else 0.0,
                    ])
                    obs_features.extend(self._unit_type_onehot(other_agent))
                else:
                    obs_features.extend([0.0, 1.0 if other_alive else 0.0] + [0.0] * 9)
                    obs_features.extend(self._unit_type_onehot(other_agent))
                    
            # --- 3. 敌军特征 (每个敌军13维 + 可选单位类型one-hot) ---
            enemy_units = self.enemies
            for enemy in enemy_units:
                enemy_alive = bool(enemy.is_alive)
                enemy_visible = enemy_alive and self.is_visible_to_agent(agent, enemy)
                if enemy_visible:
                    rel_x = enemy.position[0] - agent.position[0]
                    rel_y = enemy.position[1] - agent.position[1]
                    raw_dist = math.hypot(rel_x, rel_y)
                    dist = raw_dist / sight_range
                    enemy_hp = enemy.health / enemy.max_health
                    line_of_fire, in_range, turret_alignment = self._target_geometry_features(agent, enemy, raw_dist)
                    obs_features.extend([
                        1.0,
                        1.0,
                        rel_x / sight_range,
                        rel_y / sight_range,
                        dist,
                        enemy_hp,
                        math.cos(math.radians(enemy.direction_angle)),
                        math.sin(math.radians(enemy.direction_angle)),
                        math.cos(math.radians(enemy.turret_direction_angle)),
                        math.sin(math.radians(enemy.turret_direction_angle)),
                        line_of_fire,
                        in_range,
                        turret_alignment,
                    ])
                    obs_features.extend(self._unit_type_onehot(enemy))
                else:
                    obs_features.extend([0.0, 1.0 if enemy_alive else 0.0] + [0.0] * 11)
                    obs_features.extend(self._unit_type_onehot(enemy))
                    
            # --- 4. 视野内最近的子弹特征 (最多K颗，每颗9维: visible, active, local features...) ---
            visible_bullets = []
            # 遍历底层引擎提供的可见子弹列表
            for bullet in agent.visible_bullets.bullets:
                # 环境层二次过滤：物理防透视遮挡检测（隔着墙飞行的子弹看不到）
                if not self.check_raycast_unblocked(agent, bullet):
                    continue
                    
                dx = bullet.position[0] - agent.position[0]
                dy = bullet.position[1] - agent.position[1]
                dist = math.hypot(dx, dy)
                hostile = hasattr(bullet, "shooter_team") and bullet.shooter_team != agent.team
                visible_bullets.append((0 if hostile else 1, self._bullet_time_to_impact(agent, bullet), dist, bullet))

            visible_bullets.sort(key=lambda item: (item[0], item[1]))
            for slot in range(self.max_obs_bullets):
                if slot < len(visible_bullets):
                    _, impact_time, raw_dist, bullet = visible_bullets[slot]
                    rel_x = bullet.position[0] - agent.position[0]
                    rel_y = bullet.position[1] - agent.position[1]
                    vel_x, vel_y = getattr(bullet, "velocity", (0.0, 0.0))
                    hostile_flag = 1.0 if hasattr(bullet, "shooter_team") and bullet.shooter_team != agent.team else 0.0
                    obs_features.extend([
                        1.0,
                        1.0 if getattr(bullet, "is_active", True) else 0.0,
                        rel_x / sight_range,
                        rel_y / sight_range,
                        vel_x / self.bullet_norm_speed,
                        vel_y / self.bullet_norm_speed,
                        raw_dist / sight_range,
                        hostile_flag,
                        impact_time,
                    ])
                else:
                    obs_features.extend([0.0] * 9)

            # --- 5. 局部地图特征 (可选: agent-centered grid * terrain features) ---
            obs_features.extend(self._local_map_features(agent))

            # --- 6. 时间进度特征 (1维) ---
            obs_features.append(time_ratio)
                    
            obs_list.append(np.array(obs_features, dtype=np.float32))
        return obs_list

    def get_state(self):
        """
        获取全局绝对状态 (Global State)
        """
        state_features = []
        
        # 1. 玩家智能体绝对状态 (每个11维 + 可选单位类型one-hot)
        for agent_id, agent in enumerate(self.agents):
            if agent.is_alive:
                norm_x = agent.position[0] / self.screen_width
                norm_y = agent.position[1] / self.screen_height
                hp_ratio = agent.health / agent.max_health
                cos_dir = math.cos(math.radians(agent.direction_angle))
                sin_dir = math.sin(math.radians(agent.direction_angle))
                cos_turret = math.cos(math.radians(agent.turret_direction_angle))
                sin_turret = math.sin(math.radians(agent.turret_direction_angle))
                cooldown_ratio = self._cooldown_ratio(agent, self.agent_fire_cooldowns[agent_id])
                
                state_features.extend([
                    1.0,
                    norm_x,
                    norm_y,
                    hp_ratio,
                    cos_dir,
                    sin_dir,
                    cos_turret,
                    sin_turret,
                    cooldown_ratio,
                    self._unit_speed_norm(agent),
                    self._unit_angular_norm(agent),
                ])
                state_features.extend(self._unit_type_onehot(agent))
            else:
                state_features.extend([0.0] * 11)
                state_features.extend(self._unit_type_onehot(agent))
                
        # 2. 敌方单位绝对状态 (每个10维 + 可选单位类型one-hot)
        for enemy in self.enemies:
            if enemy.is_alive:
                norm_x = enemy.position[0] / self.screen_width
                norm_y = enemy.position[1] / self.screen_height
                hp_ratio = enemy.health / enemy.max_health
                cos_dir = math.cos(math.radians(enemy.direction_angle))
                sin_dir = math.sin(math.radians(enemy.direction_angle))
                cos_turret = math.cos(math.radians(enemy.turret_direction_angle))
                sin_turret = math.sin(math.radians(enemy.turret_direction_angle))
                
                state_features.extend([
                    1.0,
                    norm_x,
                    norm_y,
                    hp_ratio,
                    cos_dir,
                    sin_dir,
                    cos_turret,
                    sin_turret,
                    self._unit_speed_norm(enemy),
                    self._unit_angular_norm(enemy),
                ])
                state_features.extend(self._unit_type_onehot(enemy))
            else:
                state_features.extend([0.0] * 10)
                state_features.extend(self._unit_type_onehot(enemy))
                
        # 3. 场上动态子弹特征 (每颗6维)
        bullets = self.bullet_manager.bullets
        
        for i in range(self.max_state_bullets):
            if i < len(bullets):
                b = bullets[i]
                norm_x = b.position[0] / self.screen_width
                norm_y = b.position[1] / self.screen_height
                team_flag = 1.0 if hasattr(b, 'shooter_team') and b.shooter_team.name == 'PLAYER' else -1.0
                vel_x, vel_y = getattr(b, "velocity", (0.0, 0.0))
                active_flag = 1.0 if getattr(b, "is_active", True) else 0.0
                state_features.extend([
                    active_flag,
                    norm_x,
                    norm_y,
                    vel_x / self.bullet_norm_speed,
                    vel_y / self.bullet_norm_speed,
                    team_flag,
                ])
            else:
                state_features.extend([0.0] * 6)

        # 4. 全局地图特征 (可选: fixed 10x10 terrain grid by default)
        state_features.extend(self._global_map_features())

        time_ratio = self.steps / self.max_steps
        state_features.append(time_ratio)
        return np.array(state_features, dtype=np.float32)

    def _render_to_video(self):
        if self.video_writer is None:
            return
        self.screen.fill((50, 50, 70))
        camera_offset = [0.0, 0.0]
        
        # 渲染逻辑适配：交由 UnitManager 统一绘制
        self.game_map.draw(self.screen, camera_offset)
        self.unit_manager.draw(self.screen, camera_offset)
        self.bullet_manager.draw(self.screen, camera_offset)
        
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)

    def _parse_action_auto_aim(self, agent_id, agent, action):
        """
        模式 A: 开启辅助瞄准时的动作解析 (10 维)
        """
        # 1. 走位与开火判定
        if action < 9:
            chassis_action = action
            if chassis_action == 1: agent.set_movement(forward=True, backward=False)
            elif chassis_action == 2: agent.set_movement(forward=False, backward=True)
            elif chassis_action == 3: agent.set_turning(left=True, right=False)
            elif chassis_action == 4: agent.set_turning(left=False, right=True)
            elif chassis_action == 5:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 6:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=False, right=True)
            elif chassis_action == 7:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 8:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=False, right=True)
                
        elif action == 9:
            # 停步开火
            if self.agent_fire_cooldowns[agent_id] <= 0 and self._has_auto_aim_fire_target(agent):
                self._fire_agent_weapon(agent_id, agent)

        # 2. 环境层接管炮塔的“辅助瞄准”
        closest_enemy = None
        min_dist = float('inf')
        for enemy in self.enemies:
            if enemy.is_alive and self.is_visible_to_agent(agent, enemy):
                dx = enemy.position[0] - agent.position[0]
                dy = enemy.position[1] - agent.position[1]
                dist = math.hypot(dx, dy)
                if dist < min_dist:
                    min_dist = dist
                    closest_enemy = enemy
        
        if closest_enemy is not None:
            dx = closest_enemy.position[0] - agent.position[0]
            dy = closest_enemy.position[1] - agent.position[1]
            target_angle = math.degrees(math.atan2(dy, dx)) + 90
            target_angle = target_angle % 360
            if target_angle < 0: target_angle += 360
            agent.turret_target_angle = target_angle
        else:
            agent.turret_target_angle = agent.direction_angle

        if (
            self.auto_aim_auto_fire_when_ready
            and action < 9
            and self.agent_fire_cooldowns[agent_id] <= 0
            and self._has_auto_aim_fire_target(agent)
        ):
            self._fire_agent_weapon(agent_id, agent)

    def _parse_action_manual(self, agent_id, agent, action):
        """
        模式 B: 关闭辅助瞄准，完全手动操作时的动作解析 (28 维)
        """
        if action < 27:
            chassis_action = action % 9
            turret_action = action // 9
            
            if chassis_action == 1: agent.set_movement(forward=True, backward=False)
            elif chassis_action == 2: agent.set_movement(forward=False, backward=True)
            elif chassis_action == 3: agent.set_turning(left=True, right=False)
            elif chassis_action == 4: agent.set_turning(left=False, right=True)
            elif chassis_action == 5:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 6:
                agent.set_movement(forward=True, backward=False)
                agent.set_turning(left=False, right=True)
            elif chassis_action == 7:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=True, right=False)
            elif chassis_action == 8:
                agent.set_movement(forward=False, backward=True)
                agent.set_turning(left=False, right=True)
                
            # 炮塔手动旋转
            if turret_action == 1:
                agent.turret_target_angle = agent.turret_direction_angle - 15.0
            elif turret_action == 2:
                agent.turret_target_angle = agent.turret_direction_angle + 15.0
            else:
                agent.turret_target_angle = agent.turret_direction_angle
                
        elif action == 27:
            # 停步锁定并开火
            agent.turret_target_angle = agent.turret_direction_angle
            if self.agent_fire_cooldowns[agent_id] <= 0:
                self._fire_agent_weapon(agent_id, agent)

    def close(self):
        if self.use_video and self.video_writer is not None:
            self.video_writer.release()
        pygame.quit()
