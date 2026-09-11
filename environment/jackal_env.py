import pygame
import os
import random
import numpy as np
import math
import datetime
from typing import Callable, Optional

from game.Parameter import BULLET_SPEED, Team, UNIT_MIN_SIGHT_RATIO
from game.Unit.Tank.Tank import create_tank, create_enemy_tank
from game.Unit.Archie.Archie import create_archie, create_enemy_archie
from game.Map.GameMap import (
    GameMap,
    create_border_map,
    create_corridor_map,
    create_dual_corridor_map,
    create_empty_map,
    create_four_blocks_map,
    create_maze_map,
    create_random_map,
    create_river_map,
    create_spindle_map,
    create_square_ring_map,
    create_valley_map,
    create_map_from_file,
    create_map_from_strings,
)
from game.Bullet.BulletManager import BulletManager
from environment.observation import ObservationManager
from environment.rendering import create_video_writer, rgb_to_bgr
from environment.reward import RewardManager, default_reward_config, merge_reward_config

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
        self.reward_config = default_reward_config()
        if reward_config:
            merge_reward_config(self.reward_config, reward_config)
        
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
        self.max_obs_bullets = 3
        self.max_state_bullets = 10
        self.obs_sight_range = self.unit_sight_range
        self.bullet_norm_speed = max(1.0, float(BULLET_SPEED))

        self.observation_manager = ObservationManager(self)
        self.reward_manager = RewardManager(self, self.reward_config)
        
        if self.use_video:
            os.makedirs(self.video_dir, exist_ok=True)

    def _create_game_map(self) -> GameMap:
        if self.map_data:
            return create_map_from_strings(self.map_data, tile_size=self.map_tile_size)
        if self.map_file:
            game_map = create_map_from_file(self.map_file, tile_size=self.map_tile_size)
            if game_map is None:
                raise ValueError(f"Failed to load map_file: {self.map_file}")
            return game_map

        map_factories: dict[str, Callable[[], Optional[GameMap]]] = {
            "border": create_border_map,
            "empty": create_empty_map,
            "maze": create_maze_map,
            "random": create_random_map,
            "valley": create_valley_map,
            "valley_map": create_valley_map,
            "river": create_river_map,
            "river_map": create_river_map,
            "spindle": create_spindle_map,
            "spindle_map": create_spindle_map,
            "corridor": create_corridor_map,
            "corridor_map": create_corridor_map,
            "dual_corridor": create_dual_corridor_map,
            "dual_corridor_map": create_dual_corridor_map,
            "square_ring": create_square_ring_map,
            "square_ring_map": create_square_ring_map,
            "four_blocks": create_four_blocks_map,
            "four_blocks_map": create_four_blocks_map,
        }
        factory = map_factories.get(self.map_name)
        if factory is None:
            raise ValueError(f"Unknown map_name: {self.map_name}")

        game_map = factory()
        if game_map is None:
            raise ValueError(f"Failed to load bundled map: {self.map_name}")
        return game_map

    def set_reward_config(self, reward_config):
        if reward_config:
            merge_reward_config(self.reward_config, reward_config)

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

    def _configure_unit_weapon(self, unit, *, is_enemy):
        unit_type = str(getattr(unit, "unit_type", "tank")).lower()
        overrides = self.bullet_overrides_by_unit_type.get(unit_type, {})
        cooldown = (
            self._enemy_ai_fire_cooldown_max(unit)
            if is_enemy
            else self._agent_fire_cooldown_max(unit)
        )
        unit.configure_weapon(
            fire_cooldown=cooldown,
            projectile_overrides=overrides,
        )

    def _fire_agent_weapon(self, agent):
        bullet = agent.fire()
        if not bullet:
            return None
        self.bullet_manager.add_bullet(bullet)
        return bullet

    def reset(self):
        self.steps = 0
        self.has_enemy_kill = False
        self.episode_reward_so_far = 0.0
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
            self._configure_unit_weapon(player, is_enemy=False)
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
            self._configure_unit_weapon(enemy, is_enemy=True)
            initial_heading = self._initial_heading(self.enemy_initial_headings, i)
            if initial_heading is not None:
                self._set_unit_heading(enemy, initial_heading)
            self._apply_initial_heading_jitter(enemy)

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
            fps = int(1.0 / self.delta_time)
            self.video_writer = create_video_writer(
                video_path,
                fps,
                (self.screen_width, self.screen_height),
            )
            self._render_to_video()
            
        return (
            self.observation_manager.get_observations(),
            self.observation_manager.get_state(),
        )

    def _refresh_all_vision(self):
        """刷新所有单位的初始/当前视野，不推进物理和 AI。"""
        for unit in self.unit_manager.units:
            if unit is not None and unit.is_alive:
                unit._update_vision(self.unit_manager, self.bullet_manager, self.game_map)

    def _distance_between(self, src, dst):
        return math.hypot(dst.position[0] - src.position[0], dst.position[1] - src.position[1])

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
        in_range = 1.0 if dist <= observer.weapon_range() else 0.0
        return line_of_fire, in_range, turret_alignment

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
            if agent.can_fire() and self._has_auto_aim_fire_target(agent):
                avail_actions[9] = 1
        else:
            # manual_aim: 0~26 为机动+炮塔，27 为开火
            avail_actions[0:28] = [1] * 28
        return avail_actions
        
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def step(self, actions):
        self.steps += 1
        
       # --- 1. 动作解析与执行 ---
        for agent_id, agent in enumerate(self.agents):
            if not agent.is_alive: continue
                
            action = actions[agent_id]
            
            # 每次解析前先重置机身指令
            agent.set_movement(forward=False, backward=False)
            agent.set_turning(left=False, right=False)
            
            # 根据当前的瞄准模式，将动作分发给对应的成员函数处理
            if self.auto_aim:
                self._parse_action_auto_aim(agent, action)
            else:
                self._parse_action_manual(agent, action)

        # ==========================================
        # 1. 物理步进前：采集环境快照
        pre_stats = self.reward_manager.battle_stats()
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
        post_stats = self.reward_manager.battle_stats()
        reward, info = self.reward_manager.calculate(pre_stats, post_stats, actions)
        done = self._check_done()
        self.episode_reward_so_far += float(reward)
        # ==========================================
        
        return (
            self.observation_manager.get_observations(),
            self.observation_manager.get_state(),
            reward,
            done,
            info,
        )
    
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

    def get_obs(self):
        """Return local observations through the stable environment API."""
        return self.observation_manager.get_observations()

    def get_state(self):
        """Return centralized state through the stable environment API."""
        return self.observation_manager.get_state()

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
        frame = rgb_to_bgr(frame)
        self.video_writer.write(frame)

    def _parse_action_auto_aim(self, agent, action):
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
            if agent.can_fire() and self._has_auto_aim_fire_target(agent):
                self._fire_agent_weapon(agent)

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
            and agent.can_fire()
            and self._has_auto_aim_fire_target(agent)
        ):
            self._fire_agent_weapon(agent)

    def _parse_action_manual(self, agent, action):
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
            if agent.can_fire():
                self._fire_agent_weapon(agent)

    def close(self):
        if self.use_video and self.video_writer is not None:
            self.video_writer.release()
        pygame.quit()
