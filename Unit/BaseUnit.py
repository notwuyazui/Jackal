'''
    描述战斗单位的基类
'''

import pygame
import math
import json
import os
from Parameter import *
from utils import *
from GameMode import *
from typing import List, Tuple

class BaseUnit:
    def __init__(self, unit_id, unit_team, usingAI, unit_type, body_image_path, turret_image_path,
                 visible=True, 
                 max_speed_rate=1.0, 
                 max_acceleration_rate=INF, 
                 min_acceleration_rate=-INF, 
                 max_angular_speed_rate=INF, 
                 turret_angular_speed_rate=INF, 
                 max_health_rate=1.0, 
                 sight_range=INF,
                 armor_type=ArmorType.NONE, 
                 ammunition_types=[], 
                 ammo_switch_time=UNIT_AMMO_SWITCH_TIME):
        
        # 基本信息
        self.id: int = unit_id
        self.team: Team = unit_team
        self.unit_type: str = unit_type
        self.body_image_path: str = body_image_path
        self.turret_image_path: str = turret_image_path
        self.body_image = load_image(self.body_image_path) if self.body_image_path else None
        self.turret_image = load_image(self.turret_image_path) if self.turret_image_path else None
        self.size = self.body_image.get_size()
        self.usingAI = usingAI
        self.visible = visible
        
        # 基本属性
        self.max_speed_rate: float = max_speed_rate
        self.max_acceleration_rate: float = max_acceleration_rate
        self.min_acceleration_rate: float = min_acceleration_rate
        self.max_angular_speed_rate: float = max_angular_speed_rate
        self.turret_angular_speed_rate: float = turret_angular_speed_rate
        self.max_health_rate: float = max_health_rate
        self.armor_type: ArmorType = armor_type                                                        # 护甲类型
        self.ammunition_types: List[str] = ammunition_types                                            # 单位拥有弹种
        self.ammo_switch_time: float =  ammo_switch_time                                               # 单位切换弹种时间
        
        self.max_speed = UNIT_SPEED * self.max_speed_rate                                       # 最大速度  
        self.max_acceleration = UNIT_ACC * self.max_acceleration_rate                           # 最大加速度
        self.min_acceleration = UNIT_ACC * self.min_acceleration_rate                           # 最小加速度
        self.max_angular_speed = UNIT_ANGULAR_SPEED * self.max_angular_speed_rate               # 最大角速度
        self.turret_angular_speed = UNIT_TURRET_ANGULAR_SPEED * self.turret_angular_speed_rate  # 炮塔转动角速度
        self.max_health = UNIT_HEALTH * self.max_health_rate                                    # 最大生命值
        self.sight_range = sight_range                                                          # 视野范围
        
        # 视野
        from Map.Map import GameMap
        from Unit.UnitManager import UnitManager
        from Bullet.BulletManager import BulletManager
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
        self.bounding_box: pygame.Rect = None                # 碰撞箱，pygame.Rect对象
        self.velocity: Tuple[float, float] = self.cal_velocity()     # 速度向量
        self.current_ammunition: str = ""            # 单位当前选中弹种
        self.fire_cooldown: float = 0.0              # 剩余开火冷却时间
        
        self.is_alive = True
        self.reload_timer = 0.0         # 切换弹种剩余时间计时器
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
        self.killed_by = None               # 击杀者
        self.living_time = 0.0              # 存活时间
        self.reward = 0.0
        
        # 初始化碰撞箱
        self._update_bounding_box()
        
    def cal_velocity(self):
        adjusted_angle = self.direction_angle - 90
        angle_rad = math.radians(adjusted_angle)
        return (
            self.speed * math.cos(angle_rad),
            self.speed * math.sin(angle_rad)
        )

    def update(self, delta_time, unit_manager = None, bullet_manager = None, game_map = None):
        from Unit.UnitManager import UnitManager
        from Bullet.BulletManager import BulletManager
        from Map.Map import GameMap
        if unit_manager is None:
            unit_manager = UnitManager()
        if bullet_manager is None:
            bullet_manager = BulletManager()
        if game_map is None:
            game_map = GameMap()
            
        old_position = self.position
        old_bounding_box = self.bounding_box.copy() if self.bounding_box else None
        
        if not self.is_alive:
            return False
        
        if self.health <= 0:
            self.is_alive = False
            return False
        
        self.living_time += delta_time
        self._update_vision(unit_manager, bullet_manager, game_map)              # 更新视野
        self._update_ammo_switch(delta_time)         # 更新弹种切换计时器
        self._update_fire_cooldown(delta_time)       # 更新开火冷却时间
        self._update_speed(delta_time)               # 更新速度
        self._update_direction(delta_time)           # 更新朝向
        self._update_turret_direction(delta_time)    # 更新炮塔朝向
        self._update_position(delta_time)            # 更新位置
        self._update_bounding_box()                  # 更新碰撞箱
        self.velocity = self.cal_velocity()          # 更新速度向量
        
        if self.is_switching_ammo and self.reload_timer <= 0:    # 完成弹种切换
            self._complete_ammo_switch()

        # 检查与障碍物的碰撞
        if self.bounding_box:
            for obstacle in game_map.unit_obstacles: 
                if self.bounding_box.colliderect(obstacle):
                    # 发生碰撞，恢复到之前的位置
                    self.position = old_position
                    self._update_bounding_box()
                    self.speed = 0  # 停止移动
                    return True
        
        return True
    
    def _update_vision(self, unit_manager, bullet_manager, game_map) -> None:
        """更新视野"""
        self.visible_map = game_map
        self.visible_units.clear()
        self.visible_bullets.clear()
        for unit in unit_manager.units:
            if count_distance(self,unit) <= self.sight_range:
                self.visible_units.add_unit(unit, bullet_manager, game_map)
        for bullet in bullet_manager.bullets:
            if count_distance(self,bullet) <= self.sight_range:
                self.visible_bullets.add_bullet(bullet)
    
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
    
    def _update_speed(self, delta_time) -> None:
        """更新速度"""
        # 应用加速度
        self.speed += self.acceleration * delta_time
        
        # 限制速度范围
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < -self.max_speed:
            self.speed = -self.max_speed
    
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
        dx = self.velocity[0] * delta_time
        dy = self.velocity[1] * delta_time
        x, y = self.position
        self.position = (x + dx, y + dy)
    
    def _update_bounding_box(self) -> None:
        if self.size[0] > 0 and self.size[1] > 0:
            x, y = self.position
            width, height = self.size
            self.bounding_box = pygame.Rect(x - width / 2, y - height / 2, width, height)
            
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
        if bullet_class is None:
            bullet_class = get_class_from_str(self.current_ammunition)
        
        if self.is_switching_ammo:
            # 坦克正在切换弹药
            return None
        
        if not self.current_ammunition:
            # 当前弹药类型没有弹药
            return None
        
        if self.fire_cooldown > 0:
            # 坦克开火冷却中
            return None
        
        turret_angle_rad = math.radians(self.turret_direction_angle - 90)
        
        bullet_start_x = self.position[0] + math.cos(turret_angle_rad) * (self.size[0] / 2 + 5)
        bullet_start_y = self.position[1] + math.sin(turret_angle_rad) * (self.size[1] / 2 + 5)
        
        bullet_direction = (math.cos(turret_angle_rad), math.sin(turret_angle_rad))
        
        try:
            bullet = bullet_class(
                projectile_id=f"bullet_{self.id}_{pygame.time.get_ticks()}",  # 使用时间戳确保唯一性
                shooter=self,
                shooter_team=self.team,
                position=(bullet_start_x, bullet_start_y),
                velocity_direction=bullet_direction
            )
            
            # 根据当前弹药类型设置子弹属性
            self._configure_bullet_for_ammo(bullet)
            self.fire_cooldown = bullet.cooldown        # 设置开火冷却时间
                
            return bullet
            
        except Exception as e:
            print(f"创建子弹时出错: {e}")
            return None 
        
    def _configure_bullet_for_ammo(self, bullet) -> None:
        """
        根据当前弹药类型配置子弹属性
        """
        if self.current_ammunition == "normal_shell":
            from Bullet.NormalShell.NormalShell import NormalShell
            if isinstance(bullet, NormalShell):
                # 可以在这里调整特定属性
                pass
        elif self.current_ammunition == "rocket_shell":
            from Bullet.RocketShell.RocketShell import RocketShell
            if isinstance(bullet, RocketShell):
                # 可以在这里调整特定属性
                pass
        elif self.current_ammunition == "bullet":
            bullet.damage_rate = 1.0
            bullet.penetration = [1.0, 1.0, 1.0]
            bullet.speed_rate = 1.0
            bullet.is_explosive = False
            
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
        
    def set_movement(self, forward=False, backward=False) -> None:
        """
        设置坦克前进或后退
        """
        if forward and not backward:
            self.acceleration = self.max_acceleration
        elif backward and not forward:
            self.acceleration = self.min_acceleration
        else:
            self.acceleration = 0
    
    def set_turning(self, left=False, right=False) -> None:
        """
        设置坦克转向
        """
        if left and not right:
            self.angular_speed = -self.max_angular_speed
        elif right and not left:
            self.angular_speed = self.max_angular_speed
        else:
            self.angular_speed = 0
    
    def set_turret_target_to_mouse(self, mouse_pos, camera_offset) -> None:
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
    
    def draw(self, surface, camera_offset=(0, 0), mouse_pos=None) -> None:
        if not self.is_alive:
            return
        if not self.visible:
            return

        
        screen_x = self.position[0] - camera_offset[0]
        screen_y = self.position[1] - camera_offset[1]
        
        # 绘制车身
        if self.body_image:
            rotated_body = pygame.transform.rotate(self.body_image, -self.direction_angle)
            body_rect = rotated_body.get_rect(center=(screen_x, screen_y))
            surface.blit(rotated_body, body_rect)
        
        # 绘制炮塔
        if self.turret_image:
            rotated_turret = pygame.transform.rotate(self.turret_image, -self.turret_direction_angle)
            turret_rect = rotated_turret.get_rect(center=(screen_x, screen_y))
            surface.blit(rotated_turret, turret_rect)
        
        # 绘制生命条
        if DRAW_HEALTH_BAR or DEBUG_MODE:
            self._draw_health_bar(surface, screen_x, screen_y)
            
        # 绘制视野范围
        if DRAW_SIGHT_RANGE or DEBUG_MODE:
            self._draw_sight_range(surface, camera_offset)
            
        # 绘制从坦克到鼠标位置的线段
        if mouse_pos is not None and (DRAW_MOUSE_TARGET_LINE or DEBUG_MODE) :
            self._draw_mouse_target_line(surface, camera_offset, mouse_pos)
    
    def _draw_health_bar(self, surface, x, y) -> None:
        """
        绘制生命条并在血条中间显示生命值
        """
        bar_width = 40
        bar_height = 8  # 稍微增加高度以容纳文字
        
        bar_x = x - bar_width / 2
        bar_y = y - self.size[1] / 2 - 15  # 稍微上调血条位置
        
        # 绘制背景（红色）
        pygame.draw.rect(surface, (255, 0, 0), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # 绘制生命值（绿色）
        health_percentage = self.health / self.max_health
        current_width = bar_width * health_percentage
        pygame.draw.rect(surface, (0, 255, 0), 
                        (bar_x, bar_y, current_width, bar_height))
        
        # 绘制边框
        pygame.draw.rect(surface, (255, 255, 255), 
                        (bar_x, bar_y, bar_width, bar_height), 1)
        
        # 在血条中间显示生命值文本
        health_text = f"{int(self.health)}/{int(self.max_health)}"
        
        # 使用小号字体
        font = pygame.font.Font(None, 12)
        text_surface = font.render(health_text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(x, bar_y + bar_height / 2))
        
        surface.blit(text_surface, text_rect)
    
    def _draw_sight_range(self, surface, camera_offset):
        """绘制视野范围"""
        if not self.is_alive:
            return
        screen_x = self.position[0] - camera_offset[0]
        screen_y = self.position[1] - camera_offset[1]
        pygame.draw.circle(surface, (0, 0, 0), 
                          (int(screen_x), int(screen_y)), 
                          int(self.sight_range), 1)
    
    def _draw_mouse_target_line(self, surface, camera_offset, mouse_pos):
        """绘制从坦克到鼠标位置的线段"""
        if not self.is_alive:
            return
        screen_x = self.position[0] - camera_offset[0]
        screen_y = self.position[1] - camera_offset[1]
        pygame.draw.line(surface, (255, 0, 255), (screen_x, screen_y), (mouse_pos[0], mouse_pos[1]), 1)
        pygame.draw.circle(surface, (255, 0, 255), (int(mouse_pos[0]), int(mouse_pos[1])), 3)
                         
    
    def take_damage(self, unit_manager, damage_source, damage_amount) -> float:
        """
        坦克承受伤害
        """
        self.health -= damage_amount
        self.damage_received += damage_amount
        damage_source.damage_dealt += damage_amount
        destroyed = False
        
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            destroyed = True
            print(f"坦克 {self.id} 被摧毁")
            damage_source.destroy_enemy_count += 1
            self.killed_by = damage_source.id
        self._handle_assistance(unit_manager, damage_source, destroyed, damage_amount)
        return destroyed, damage_amount

    def _handle_assistance(self, unit_manager, damage_source, destroy:bool , damage_amount:float) -> None:
        # 当自身受到伤害时，处理伤害来源的协助信息
        for unit in unit_manager.units:
            if unit.id == damage_source.id:
                continue
            if unit.team != damage_source.team:
                continue
            if unit.is_alive == False:
                continue
            if count_distance(self, unit) <= unit.sight_range:
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
            "killed_by": self.killed_by,
            "living_time": self.living_time,
            "reward": self.reward
        }

    def save_to_file(self, file_name="default_unit.json") -> bool:
        # 保存单位信息到文件, file_name包含后缀，但不包含路径
        # 当前BaseUnit类中的属性没有确定下来，该方法为TODO
        save_dir = DEFAULT_UNIT_PATH
        filepath = os.path.join(save_dir, file_name)
        
        pass
        
    def save(self, file_name = None) -> bool:
        # 提供一种直接的保存方法
        if file_name is None:
            save_path = get_next_filename(DEFAULT_UNIT_PATH, 'default_unit', '.json')
            return self.save_to_file(save_path)
        return self.save_to_file(file_name)
    
    @classmethod
    def load_from_file(cls, file_name="default_unit.json") -> 'BaseUnit':
        # 从JSON文件加载兵种单位信息并创建实例
        # 当前BaseUnit类中的属性没有确定下来，该方法为TODO
        filepath = os.path.join(DEFAULT_UNIT_PATH, file_name)
        
        pass