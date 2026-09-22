'''
    描述战斗单位的基类
'''

import pygame
import math
import json
import os
from itertools import count
from game.Parameter import *
from game.utils import *
from typing import Any, List, Mapping, Optional, Tuple

from game.Bullet.weapon_specs import PROJECTILE_SPECS, ProjectileSpec, get_projectile_spec

_PROJECTILE_SEQUENCE = count()

class BaseUnit:
    def __init__(self, unit_id, unit_team, usingAI, unit_type, body_image_path, turret_image_path,
                 size=(1.0, 1.0),
                 visible=True,
                 max_speed_rate=1.0, 
                 max_acceleration_rate=INF, 
                 min_acceleration_rate=-INF, 
                 max_angular_speed_rate=INF, 
                 turret_angular_speed_rate=INF, 
                 max_health_rate=1.0, 
                 sight_range=INF,
                 communication_range=INF,
                 armor_type=ArmorType.NONE, 
                 ammunition_types=None,
                 ammo_switch_time=UNIT_AMMO_SWITCH_TIME,
                 collision_size=None,
                 ai_intelligence_level=DEFAULT_AI_INTELLIGENCE_LEVEL):
        
        # 基本信息
        self.id: int = unit_id
        self.team: Team = unit_team
        self.unit_type: str = unit_type
        self.body_image_path: str = body_image_path
        self.turret_image_path: str = turret_image_path
        self.size = (float(size[0]), float(size[1]))
        base_collision_size = self.size if collision_size is None else collision_size
        self.base_collision_size = (
            float(base_collision_size[0]),
            float(base_collision_size[1]),
        )
        if self.base_collision_size[0] <= 0.0 or self.base_collision_size[1] <= 0.0:
            raise ValueError("collision_size dimensions must be > 0")
        self.collision_scale = 1.0
        self.collision_size = self.base_collision_size
        self.usingAI: bool = bool(usingAI)
        self.ai_intelligence_level = int(ai_intelligence_level)
        if not 1 <= self.ai_intelligence_level <= 9:
            raise ValueError("ai_intelligence_level must be between 1 and 9")
        self.visible = visible
        
        # 基本属性
        self.max_speed_rate: float = max_speed_rate
        self.max_acceleration_rate: float = max_acceleration_rate
        self.min_acceleration_rate: float = min_acceleration_rate
        self.max_angular_speed_rate: float = max_angular_speed_rate
        self.turret_angular_speed_rate: float = turret_angular_speed_rate
        self.max_health_rate: float = max_health_rate
        self.armor_type: ArmorType = armor_type                                                        # 护甲类型
        self.ammunition_types: List[str] = list(ammunition_types or [])                                # 单位拥有弹种
        self.ammo_switch_time: float =  ammo_switch_time                                               # 单位切换弹种时间
        
        self.max_speed = UNIT_SPEED * self.max_speed_rate                                       # 最大速度  
        self.max_acceleration = UNIT_ACC * self.max_acceleration_rate                           # 最大加速度
        self.min_acceleration = UNIT_ACC * self.min_acceleration_rate                           # 最小加速度
        self.max_angular_speed = UNIT_ANGULAR_SPEED * self.max_angular_speed_rate               # 最大角速度
        self.turret_angular_speed = UNIT_TURRET_ANGULAR_SPEED * self.turret_angular_speed_rate  # 炮塔转动角速度
        self.max_health = UNIT_HEALTH * self.max_health_rate                                    # 最大生命值
        self.sight_range = sight_range                                                          # 视野范围
        self.min_sight_range = UNIT_MIN_SIGHT_RATIO * self.sight_range                          # 应用水滴形视野时，最小视野范围（即向正后方的视野范围）
        self.communication_range = communication_range                                          # 通信范围, 此值应小于等于视野范围
        
        # 视野
        from game.Map.GameMap import GameMap
        from game.Unit.UnitManager import UnitManager
        from game.Bullet.BulletManager import BulletManager
        self.visible_map = GameMap()
        self.visible_units = UnitManager()
        self.visible_bullets = BulletManager()
        
        # 实时属性
        self.position = (0.0, 0.0)
        self.speed = 0.0
        self.direction_angle = 0.0              # 单位朝向角度
        self.turret_direction_angle = 0.0       # 单位炮塔朝向角度
        self.acceleration = 0.0
        self.angular_speed = 0.0
        self.health: float = self.max_health
        # bounding_box is the projectile hit box. collision_box is the independent
        # physical footprint used for terrain and optional unit-unit collisions.
        # Both boxes are valid Rect instances for the full lifetime of a unit.
        self.bounding_box: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self.collision_box: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self.velocity: Tuple[float, float] = self.cal_velocity()     # 速度向量
        self.current_ammunition: str = ""            # 单位当前选中弹种
        self.fire_cooldown: float = 0.0              # 剩余开火冷却时间
        self.fire_cooldown_override: Optional[float] = None
        self.projectile_overrides: Mapping[str, Any] = {}
        # 敌方 AI 可由环境覆盖瞄准容差；冷却统一由单位武器状态管理。
        self.ai_fire_angle_tolerance: float = 10.0
        
        self.is_alive = True
        self.reload_timer = 0.0         # 切换弹种剩余时间计时器
        self.target_ammunition: str = ""
        self.turret_target_angle = 0.0          # 炮塔目标角度
        self.is_switching_ammo = False          # 是否正在切换弹药
        
        if self.ammunition_types:
            self.current_ammunition = self.ammunition_types[0]
        
        # 记录
        self.destroy_enemy_count = 0        # 击杀数
        self.damage_dealt = 0               # 伤害输出
        self.assist_destroy_count = 0       # 协助击杀（为击杀者提供视野造成的击杀）
        self.assist_damage_dealt = 0        # 协助伤害（为击杀者提供视野造成的伤害）
        self.damage_received = 0            # 伤害承受
        self.potential_damage = 0.0         # 潜在伤害
        self.killed_by = None               # 击杀者
        self.living_time = 0.0              # 存活时间
        self.reward = 0.0
        self.blocked_by_unit = False
        self.unit_collision_count = 0

        # 地块效果会在每一帧开始时重置并重新计算。
        self.speed_slow_multiplier = 1.0
        self.conceal = False
        
        # 初始化碰撞箱
        self._update_bounding_box()
        self._update_collision_box()
        
    def cal_velocity(self):
        adjusted_angle = self.direction_angle - 90
        angle_rad = math.radians(adjusted_angle)
        return (
            self.speed * math.cos(angle_rad),
            self.speed * math.sin(angle_rad)
        )

    def update(self, delta_time, unit_manager = None, bullet_manager = None, game_map = None):
        from game.Unit.UnitManager import UnitManager
        from game.Bullet.BulletManager import BulletManager
        from game.Map.GameMap import GameMap
        if unit_manager is None:
            unit_manager = UnitManager()
        if bullet_manager is None:
            bullet_manager = BulletManager()
        if game_map is None:
            game_map = GameMap()
            
        old_position = self.position
        if not self.is_alive:
            return False
        
        if self.health <= 0:
            self.is_alive = False
            return False
        
        self.living_time += delta_time
        self._frame_init()                            # 重置上一帧地块效果
        self._update_tile_buff(game_map, unit_manager)  # 应用当前位置的地块效果
        self._update_ammo_switch(delta_time)         # 更新弹种切换计时器
        self._update_fire_cooldown(delta_time)       # 更新开火冷却时间
        self._update_speed(delta_time)               # 更新速度
        self._update_direction(delta_time)           # 更新朝向
        self._update_turret_direction(delta_time)    # 更新炮塔朝向
        self._update_position(delta_time)            # 更新位置
        self._update_bounding_box()                  # 更新受击箱
        self._update_collision_box()                 # 更新物理碰撞箱
        self.velocity = self.cal_velocity()          # 更新速度向量
        
        if self.is_switching_ammo and self.reload_timer <= 0:    # 完成弹种切换
            self._complete_ammo_switch()

        # 地图边界与不可通行地块共用 GameMap 的放置查询。
        if not game_map.can_place_unit(self.position, self.collision_size):
            self._rollback_blocked_movement(old_position)
            return True

        # 单位碰撞是可选规则。仅做邻近 broad-phase 查询和矩形精确检测；
        # 命中时回退移动者，不引入推挤、质量或弹性求解。
        if (
            self.collision_box
            and unit_manager.enable_unit_collision
        ):
            other = unit_manager.find_unit_collision(
                self.collision_box,
                exclude_unit=self,
            )
            if other is not None:
                self._rollback_blocked_movement(old_position)
                unit_manager.record_unit_collision(self, other)
                return True
        
        return True

    def _frame_init(self) -> None:
        """重置仅在当前帧生效的地块效果。"""
        self.speed_slow_multiplier = 1.0
        self.conceal = False
        self.blocked_by_unit = False

    def _rollback_blocked_movement(
        self,
        old_position: Tuple[float, float],
    ) -> None:
        """回退本次移动，不引入推挤或滑动求解。"""

        self.position = old_position
        self._update_bounding_box()
        self._update_collision_box()
        self.speed = 0

    def configure_collision(self, scale: float = 1.0) -> None:
        """Configure the physical footprint without changing render or hit size."""

        scale = float(scale)
        if scale <= 0.0:
            raise ValueError("collision_scale must be > 0")
        self.collision_scale = scale
        self.collision_size = (
            self.base_collision_size[0] * scale,
            self.base_collision_size[1] * scale,
        )
        self._update_collision_box()

    def _update_tile_buff(self, game_map, unit_manager) -> None:
        """应用单位当前位置的地块效果。"""
        tile = game_map.get_tile_at_position(self.position[0], self.position[1])
        if tile is not None:
            tile.apply_buff(self, unit_manager)
    
    def _update_ammo_switch(self, delta_time) -> None:
        """更新弹药切换状态"""
        if self.is_switching_ammo and self.reload_timer > 0:
            self.reload_timer -= delta_time
            if self.reload_timer <= 0:
                self.reload_timer = 0
                self.is_switching_ammo = False
                self.current_ammunition = self.target_ammunition
                self.target_ammunition = ""
    
    def _update_fire_cooldown(self, delta_time) -> None:
        """更新开火冷却时间"""
        if self.fire_cooldown > 0:
            self.fire_cooldown -= delta_time
            if self.fire_cooldown < 0:
                self.fire_cooldown = 0

    def configure_weapon(
        self,
        *,
        fire_cooldown: Optional[float] = None,
        projectile_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Configure effective weapon parameters for this unit.

        The unit remains the single owner of its firing cooldown. Environment
        configuration is converted into an override here instead of creating a
        second timer outside the game engine.
        """

        if fire_cooldown is not None and float(fire_cooldown) < 0:
            raise ValueError("fire_cooldown must be >= 0")
        self.fire_cooldown_override = (
            None if fire_cooldown is None else float(fire_cooldown)
        )
        self.projectile_overrides = dict(projectile_overrides or {})
        # Validate every available ammunition eagerly so configuration errors fail
        # during reset rather than in the middle of a training episode.
        for ammunition in self.ammunition_types:
            self.get_weapon_spec(ammunition)

    def _current_projectile_overrides(self, ammunition: str) -> Mapping[str, Any]:
        overrides = self.projectile_overrides
        if ammunition in overrides:
            selected = overrides[ammunition]
        elif "default" in overrides:
            selected = overrides["default"]
        elif any(name in PROJECTILE_SPECS for name in overrides):
            # This is an ammunition-keyed mapping with no entry for the
            # requested ammunition.
            return {}
        else:
            # A flat mapping applies to all ammunition for compatibility with
            # the original environment configuration format.
            selected = overrides
        return selected if isinstance(selected, Mapping) else {}

    def get_weapon_spec(self, ammunition: Optional[str] = None) -> ProjectileSpec:
        """Return the effective, validated spec for this unit's ammunition."""

        ammo_name = str(ammunition or self.current_ammunition).lower()
        spec = get_projectile_spec(
            ammo_name,
            self._current_projectile_overrides(ammo_name),
        )
        if self.fire_cooldown_override is not None:
            spec = spec.with_overrides({"cooldown": self.fire_cooldown_override})
        return spec

    def weapon_range(self) -> float:
        return self.get_weapon_spec().max_range if self.current_ammunition else 0.0

    def fire_cooldown_duration(self) -> float:
        return self.get_weapon_spec().cooldown if self.current_ammunition else 0.0

    def fire_cooldown_ratio(self) -> float:
        duration = max(1e-6, self.fire_cooldown_duration())
        return max(0.0, min(1.0, self.fire_cooldown / duration))

    def can_fire(self) -> bool:
        return bool(
            self.is_alive
            and self.current_ammunition
            and not self.is_switching_ammo
            and self.fire_cooldown <= 0.0
        )
    
    def _update_speed(self, delta_time) -> None:
        """更新速度"""
        effective_max_speed = self.max_speed * self.speed_slow_multiplier
        real_acceleration = self.acceleration

        # 已超过地块限制时逐步减速，防止速度在新上限附近震荡。
        if self.speed > effective_max_speed:
            real_acceleration = -self.max_acceleration
        elif self.speed < -effective_max_speed:
            real_acceleration = self.max_acceleration

        # 应用加速度
        self.speed += real_acceleration * delta_time
        
        # 穿过有效速度上限时直接钳制，避免反复越界。
        if real_acceleration >= 0 and self.speed > effective_max_speed:
            self.speed = effective_max_speed
        elif real_acceleration <= 0 and self.speed < -effective_max_speed:
            self.speed = -effective_max_speed
    
    def _update_direction(self, delta_time) -> None:
        """更新单位朝向角度"""
        if self.angular_speed != 0:
            self.direction_angle += self.angular_speed * delta_time
            self.direction_angle = self.normalize_angle(self.direction_angle)
            if abs(self.angular_speed) > self.max_angular_speed:
                self.angular_speed = self.max_angular_speed if self.angular_speed > 0 else -self.max_angular_speed
    
    def _update_turret_direction(self, delta_time) -> None:
        """更新炮塔朝向角度"""
        angle_diff = self.get_angle_difference(self.turret_direction_angle, self.turret_target_angle)
        
        if abs(angle_diff) > 0.1:
            rotation_speed = self.turret_angular_speed * delta_time
            
            if angle_diff > 0:
                if rotation_speed > angle_diff:
                    rotation_speed = angle_diff
                self.turret_direction_angle += rotation_speed
            else:
                if rotation_speed > -angle_diff:
                    rotation_speed = -angle_diff
                self.turret_direction_angle -= rotation_speed
            
            # 规范化角度到0-360度范围
            self.turret_direction_angle = self.normalize_angle(self.turret_direction_angle)
    
    def _update_position(self, delta_time) -> None:
        """根据速度更新位置"""
        self.position = self.candidate_position(delta_time)

    def candidate_position(self, delta_time: float) -> Tuple[float, float]:
        """返回按当前速度前进一个物理步的候选位置。"""

        return (
            self.position[0] + self.velocity[0] * float(delta_time),
            self.position[1] + self.velocity[1] * float(delta_time),
        )
    
    def _update_bounding_box(self) -> None:
        if self.size[0] > 0 and self.size[1] > 0:
            self.bounding_box = centered_rect(self.position, self.size)

    def _update_collision_box(self) -> None:
        self.collision_box = centered_rect(self.position, self.collision_size)

    def is_in_sight(self, target, use_tear_drop_vision: bool) -> bool:
        """
        判断目标（单位或子弹）是否在视野内。
        由 UnitManager 传入本场战斗的视野形状配置。
        """
        dx = target.position[0] - self.position[0]
        dy = target.position[1] - self.position[1]
        distance = math.hypot(dx, dy)

        if not use_tear_drop_vision:
            return distance <= self.sight_range

        # 水滴形视野
        target_angle = math.atan2(dy, dx)               # 返回 [-π, π]
        forward_rad = math.radians(self.direction_angle - 90)
        diff_angle = target_angle - forward_rad
        diff_angle = (diff_angle + math.pi) % (2 * math.pi) - math.pi

        a = (self.sight_range + self.min_sight_range) / 2
        b = (self.sight_range - self.min_sight_range) / 2
        max_dist_at_angle = a + b * math.cos(diff_angle)

        return distance <= max_dist_at_angle

    def normalize_angle(self, angle) -> float:
        """将角度规范化到0-360度范围内"""
        angle %= 360
        if angle < 0:
            angle += 360
        return angle
    
    def get_angle_difference(self, angle1, angle2) -> float:
        """
        计算两个角度之间的最小差值(考虑360度循环)
        """
        diff = angle2 - angle1
        diff = (diff + 180) % 360 - 180
        return diff
    
    def fire(self, bullet_class = None):
        registered_bullet_class = get_class_from_str(self.current_ammunition)
        if bullet_class is None:
            bullet_class = registered_bullet_class
        if not bullet_class:
            return None

        if not self.can_fire():
            return None
        
        turret_angle_rad = math.radians(self.turret_direction_angle - 90)
        
        bullet_start_x = self.position[0] + math.cos(turret_angle_rad) * (self.size[0] / 2 + 5)
        bullet_start_y = self.position[1] + math.sin(turret_angle_rad) * (self.size[1] / 2 + 5)
        
        bullet_direction = (math.cos(turret_angle_rad), math.sin(turret_angle_rad))
        
        try:
            bullet_kwargs: dict[str, Any] = {
                "projectile_id": f"bullet_{self.id}_{next(_PROJECTILE_SEQUENCE)}",
                "shooter": self,
                "shooter_team": self.team,
                "position": (bullet_start_x, bullet_start_y),
                "velocity_direction": bullet_direction,
            }
            if bullet_class is registered_bullet_class:
                bullet_kwargs["spec"] = self.get_weapon_spec()
            bullet = bullet_class(**bullet_kwargs)
            self.fire_cooldown = bullet.cooldown        # 设置开火冷却时间
                
            return bullet
            
        except Exception as e:
            print(f"创建子弹时出错: {e}")
            return None 
        
    def switch_ammunition(self, ammo_type = None) -> bool:
        if ammo_type == None:
            idx = self.ammunition_types.index(self.current_ammunition)
            next_idx = (idx + 1) % len(self.ammunition_types)
            ammo_type = self.ammunition_types[next_idx]
            
        if ammo_type not in self.ammunition_types:
            print(f"坦克 {self.id} 没有 {ammo_type} 类型弹药")
            return False
        
        if ammo_type == self.current_ammunition:
            return True
        
        self.is_switching_ammo = True
        self.reload_timer = self.ammo_switch_time
        
        self.target_ammunition = ammo_type
        
        return True     
    
    def _complete_ammo_switch(self) -> None:
        if hasattr(self, 'target_ammunition'):
            self.current_ammunition = self.target_ammunition
            delattr(self, 'target_ammunition')
        self.is_switching_ammo = False
        
    def set_movement(self, forward=False, backward=False) -> bool:
        """
        设置坦克前进或后退
        """
        if forward and not backward:
            self.acceleration = self.max_acceleration
        elif backward and not forward:
            self.acceleration = self.min_acceleration
        else:
            self.acceleration = 0
        return True
    
    def set_turning(self, left=False, right=False) -> bool:
        """
        设置坦克转向
        """
        if left and not right:
            self.angular_speed = -self.max_angular_speed
        elif right and not left:
            self.angular_speed = self.max_angular_speed
        else:
            self.angular_speed = 0
        return True
    
    def set_turret_target_to_mouse(self, mouse_pos, camera_offset) -> bool:
        """
        设置炮塔目标指向鼠标位置
        """
        # 计算鼠标的世界坐标
        world_mouse_x = mouse_pos[0] + camera_offset[0]
        world_mouse_y = mouse_pos[1] + camera_offset[1]
        
        dx = world_mouse_x - self.position[0]
        dy = world_mouse_y - self.position[1]
        
        # 计算角度（度）
        target_angle = math.degrees(math.atan2(dy, dx))
        target_angle += 90
        
        # 规范化角度到0-360度
        target_angle %= 360
        if target_angle < 0:
            target_angle += 360
        
        self.turret_target_angle = target_angle
        return True

    def _merge_visible_info_from(self, source_unit):
        """
        将源单位的可见单位和子弹合并到自己的可见信息中。
        返回 (unit_added, bullet_added, map_added, any_added)
        """
        unit_added = False
        bullet_added = False

        # 合并可见单位
        for other_unit in source_unit.visible_units.units:
            if other_unit.id == self.id:
                continue
            if not any(u.id == other_unit.id for u in self.visible_units.units):
                self.visible_units.units.append(other_unit)
                unit_added = True

        # 合并可见子弹
        for bullet in source_unit.visible_bullets.bullets:
            if not any(b.id == bullet.id for b in self.visible_bullets.bullets):
                self.visible_bullets.bullets.append(bullet)
                bullet_added = True

        # 地图信息（目前所有单位共享同一地图对象，因此永远不会有新增）
        map_added = False
        any_added = unit_added or bullet_added or map_added
        return unit_added, bullet_added, map_added, any_added

    def communicate_to(self, target_unit):
        """
        向指定的友方单位单向发送自己的可见信息。
        返回四个布尔值，表示目标单位是否因此获得了新的信息：(unit_added, bullet_added, map_added, any_added)
        """
        # 检查目标是否存活且可见且为友方
        if not target_unit.is_alive or not target_unit.visible:
            return False, False, False, False
        if target_unit.team != self.team:
            return False, False, False, False
        # 检查距离是否在通讯范围内
        dist = count_distance(self, target_unit)
        if dist > self.communication_range:
            return False, False, False, False

        # 将自身信息合并到目标单位
        return target_unit._merge_visible_info_from(self)

    def broadcast(self, unit_manager):
        """
        向通讯范围内所有可见的友方单位广播自己的可见信息。
        返回四个布尔值，表示本次广播是否有任何单位获得了新信息：(any_unit_added, any_bullet_added, any_map_added, any_added)
        """
        any_unit_added = False
        any_bullet_added = False
        any_map_added = False

        for other in unit_manager.units:
            if not other.is_alive or not other.visible:
                continue
            if other.team != self.team or other.id == self.id:
                continue
            dist = count_distance(self, other)
            if dist <= self.communication_range:
                unit_added, bullet_added, map_added, _ = other._merge_visible_info_from(self)
                if unit_added:
                    any_unit_added = True
                if bullet_added:
                    any_bullet_added = True
                if map_added:
                    any_map_added = True

        any_added = any_unit_added or any_bullet_added or any_map_added
        return any_unit_added, any_bullet_added, any_map_added, any_added

    def receive_from(self, source_unit):
        """
        从指定的友方单位接收信息（要求源单位在通讯范围内）。
        返回四个布尔值，表示自己是否因此获得了新的信息：(unit_added, bullet_added, map_added, any_added)
        """
        if not source_unit.is_alive or not source_unit.visible:
            return False, False, False, False
        if source_unit.team != self.team:
            return False, False, False, False
        dist = count_distance(self, source_unit)
        if dist > source_unit.communication_range:  # 使用源单位的通讯范围
            return False, False, False, False

        return self._merge_visible_info_from(source_unit)

    def broadcast_receive(self, unit_manager):
        """
        检查所有可见的友方单位，如果自己在对方的通讯范围内，则从对方接收信息。
        返回四个布尔值，表示自己是否因此获得了新的信息：(unit_added, bullet_added, map_added, any_added)
        """
        unit_added = False
        bullet_added = False
        map_added = False

        for other in unit_manager.units:
            if not other.is_alive or not other.visible:
                continue
            if other.team != self.team or other.id == self.id:
                continue
            dist = count_distance(self, other)
            if dist <= other.communication_range:  # 检查自己是否在其他单位的通讯范围内
                u_added, b_added, m_added, _ = self._merge_visible_info_from(other)
                if u_added:
                    unit_added = True
                if b_added:
                    bullet_added = True
                if m_added:
                    map_added = True

        any_added = unit_added or bullet_added or map_added
        return unit_added, bullet_added, map_added, any_added

    def get_visible_unit_ids(self) -> List[int]:
        """
        获取当前可见的所有单位ID列表。
        """
        return [unit.id for unit in self.visible_units.units]

    def get_info(self) -> dict:
        return {
            "id": self.id,
            "team": self.team,
            "type": self.unit_type,
            "position": self.position,
            "health": self.health,
            "max_health": self.max_health,
            "speed": self.speed,
            "max_speed": self.max_speed,
            "direction": self.direction_angle,
            "turret_direction": self.turret_direction_angle,
            "current_ammo": self.current_ammunition,
            "is_alive": self.is_alive,
            "is_switching_ammo": self.is_switching_ammo,
            "reload_timer": self.reload_timer
        }
    
    def take_damage(self, unit_manager, damage_source, damage_amount):
        """
        坦克承受伤害
        """
        actual_damage = min(self.health, max(0.0, float(damage_amount)))
        self.health -= damage_amount
        self.damage_received += damage_amount
        damage_source.damage_dealt += damage_amount
        destroyed = False
        
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            destroyed = True
            #print(f"坦克 {self.id} 被摧毁")
            damage_source.destroy_enemy_count += 1
            self.killed_by = damage_source.id
        self._handle_assistance(unit_manager, damage_source, destroyed, damage_amount)
        unit_manager.record_damage(damage_source, self, actual_damage, destroyed)
        return destroyed, damage_amount

    def take_damage_from_tile(self, damage_amount, unit_manager=None):
        """承受地形伤害；地形击杀使用 -1 作为来源标记。"""
        actual_damage = min(self.health, max(0.0, float(damage_amount)))
        self.health -= damage_amount
        destroyed = False
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            self.killed_by = -1
            destroyed = True
        if unit_manager is not None:
            unit_manager.record_damage(None, self, actual_damage, destroyed)

    def _handle_assistance(self, unit_manager, damage_source, destroy:bool , damage_amount:float) -> None:
        # 当自身受到伤害时，处理伤害来源的协助信息
        for unit in unit_manager.units:
            if unit.id == damage_source.id:
                continue
            if unit.team != damage_source.team:
                continue
            if unit.is_alive == False:
                continue
            if unit.is_in_sight(self, unit_manager.use_tear_drop_vision):
                unit.assist_damage_dealt += damage_amount
                if destroy:
                    unit.assist_destroy_count += 1
        return None

    def get_record(self) -> dict:
        return {
            "id": self.id,
            "destroy_enemy_count": self.destroy_enemy_count,
            "damage_dealt": self.damage_dealt,
            "assist_destroy_count": self.assist_destroy_count,
            "assist_damage_dealt": self.assist_damage_dealt,
            "damage_received": self.damage_received,
            "potential_damage": self.potential_damage,
            "killed_by": self.killed_by,
            "living_time": self.living_time,
            "reward": self.reward
        }

    def save_to_file(self, file_name="default_unit.json"):
        # 保存单位信息到文件, file_name包含后缀，但不包含路径
        # 当前BaseUnit类中的属性没有确定下来，该方法为TODO
        save_dir = DEFAULT_UNIT_PATH
        filepath = os.path.join(save_dir, file_name)
        
        pass
        return True
        
    def save(self, file_name = None):
        # 提供一种直接的保存方法
        if file_name is None:
            save_path = get_next_filename(DEFAULT_UNIT_PATH, 'default_unit', '.json')
            return self.save_to_file(save_path)
        return self.save_to_file(file_name)
    
    @classmethod
    def load_from_file(cls, file_name="default_unit.json"):
        # 从JSON文件加载兵种单位信息并创建实例
        # 当前BaseUnit类中的属性没有确定下来，该方法为TODO
        filepath = os.path.join(DEFAULT_UNIT_PATH, file_name)
        
        pass
