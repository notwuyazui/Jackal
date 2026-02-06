'''
    描述弹药的基类
'''

import pygame
import math
from typing import List, Tuple, Optional, Dict, Any
from Parameter import *
from utils import *
from GameMode import *
import json
import os

class BaseBullet:
    def __init__(self, 
                 projectile_id: str, 
                 shooter_id: int, 
                 shooter_team: Team,
                 position: Tuple[float, float] = (0.0, 0.0), 
                 velocity_direction: Tuple[float, float] = (1.0, 0.0),
                 bullet_image_path: Optional[str] = None,
                 size: Tuple[float, float] = (0.0, 0.0),
                 bounding_box: Optional[pygame.Rect] = None,
                 lifetime: float = 3.0,
                 speed_rate: float = 1.0,
                 damage_rate: float = 1.0,
                 penetration: List[float] = None,
                 is_explosive: bool = False,
                 explosion_radius: float = 0.0,
                 explosion_damage_rate: float = 1.0,
                 explosion_image_path: Optional[str] = None):
        
        # 基本信息
        self.id: str = projectile_id
        self.shooter_id: int = shooter_id
        self.shooter_team: Team = shooter_team
        
        # 图像和渲染
        self.image_path: Optional[str] = bullet_image_path
        self.image: Optional[pygame.Surface] = load_image(bullet_image_path) if bullet_image_path else None
        
        if size == (0.0, 0.0) and self.image:
            self.size: Tuple[float, float] = self.image.get_size()
        else:
            self.size = size
            
        # 弹道属性
        self.max_lifetime: float = lifetime                                                 # 射程
        self.speed_rate: float = speed_rate
        self.damage_rate: float = damage_rate
        self.penetration: List[float] = penetration if penetration else [1.0, 1.0, 1.0]     # 对不同护甲的伤害
        
        self.speed: float = BULLET_SPEED * self.speed_rate                                  # 速度
        self.base_damage: float = BULLET_DAMAGE * self.damage_rate                          # 基础伤害
        
        # 爆炸
        self.is_explosive: bool = is_explosive                                              # 是否为爆炸弹
        self.explosion_radius: float = explosion_radius                                     # 爆炸半径
        self.explosion_damage_rate: float = explosion_damage_rate                           # 爆炸伤害
        self.explosion_image_path: Optional[str] = explosion_image_path                     # 爆炸效果图像路径
        self.explosion_image: Optional[pygame.Surface] = None
        if explosion_image_path:
            self.explosion_image = load_image(explosion_image_path)
        
        # 实时属性
        self.lifetime: float = lifetime
        self.position: Tuple[float, float] = position
        self.velocity_direction: Tuple[float, float] = velocity_direction
        self.velocity: Tuple[float, float] = self._calculate_velocity()
        self.bounding_box: pygame.Rect = bounding_box if bounding_box else self._update_bounding_box()
        self.rotation_angle: float = math.degrees(math.atan2(velocity_direction[1], velocity_direction[0]))
        
        # 状态标志
        self.is_active: bool = True
        self.has_collided: bool = False
        self.has_exploded: bool = False
        self.explosion_timer: float = 0.0
        self.max_explosion_display_time: float = 0.3  # 爆炸效果显示时间
        self.collided_with: Optional[str] = None  # 'unit', 'obstacle', 'friendly'
        self.collided_objects: List[Any] = []  # 碰撞到的对象列表
        self.distance_traveled: float = 0.0  # 已飞行距离
        
        # 初始化碰撞箱
        self._update_bounding_box()
        
    def _calculate_velocity(self) -> Tuple[float, float]:
        """计算速度向量"""
        dir_len = math.sqrt(self.velocity_direction[0]**2 + self.velocity_direction[1]**2)
        if dir_len > 0:
            normalized_dir = (
                self.velocity_direction[0] / dir_len,
                self.velocity_direction[1] / dir_len
            )
        else:
            normalized_dir = (1.0, 0.0)
            
        return (
            normalized_dir[0] * self.speed,
            normalized_dir[1] * self.speed
        )
    
    def _update_bounding_box(self) -> pygame.Rect:
        """更新碰撞箱"""
        x, y = self.position
        width, height = self.size
        self.bounding_box = pygame.Rect(
            x - width / 2,
            y - height / 2,
            width,
            height
        )
        return self.bounding_box
    
    def update(self, delta_time: float, units: List[Any], obstacles: List[pygame.Rect]) -> bool:
        """
        更新子弹状态
        """
        if not self.is_active:
            return False
            
        # 如果已经爆炸，处理爆炸效果
        if self.has_exploded:
            return self._update_explosion(delta_time)
        
        # 更新生命期
        self.lifetime -= delta_time
        if self.lifetime <= 0:
            self.is_active = False
            return False
        
        old_position = self.position
        
        dx = self.velocity[0] * delta_time
        dy = self.velocity[1] * delta_time
        self.position = (self.position[0] + dx, self.position[1] + dy)
        
        self._update_bounding_box()
        
        # 更新已飞行距离
        self.distance_traveled += math.sqrt(dx**2 + dy**2)
        
        # 检查与障碍物的碰撞
        obstacle_collision = self._check_obstacle_collision(obstacles)
        if obstacle_collision:
            self._handle_obstacle_collision()
            return self.is_active
        
        # 检查与单位的碰撞
        unit_collision = self._check_unit_collision(units)
        if unit_collision:
            unit = unit_collision
            self._handle_unit_collision(unit)
            return self.is_active
        
        return True
    
    def _update_explosion(self, delta_time: float) -> bool:
        """更新爆炸效果"""
        self.explosion_timer += delta_time
        if self.explosion_timer >= self.max_explosion_display_time:
            self.is_active = False
        return True  # 在爆炸期间仍然返回True，以便绘制爆炸效果
    
    def _check_obstacle_collision(self, obstacles: List[pygame.Rect]) -> Optional[pygame.Rect]:
        """检查与障碍物的碰撞"""
        for obstacle in obstacles:
            if self.bounding_box.colliderect(obstacle):
                return obstacle
        return None
    
    def _check_unit_collision(self, units: List[Any]) -> Optional[Any]:
        """检查与单位的碰撞"""
        for unit in units:
            # 跳过无效单位
            if not hasattr(unit, 'is_alive') or not unit.is_alive:
                continue
                
            # 检查碰撞
            if (hasattr(unit, 'bounding_box') and unit.bounding_box and 
                self.bounding_box.colliderect(unit.bounding_box)):
                return unit
        return None
    
    def _handle_obstacle_collision(self):
        """处理与障碍物的碰撞"""
        self.has_collided = True
        self.collided_with = 'obstacle'
        
        if self.is_explosive:
            self._trigger_explosion()
        else:
            self.is_active = False
    
    def _handle_unit_collision(self, unit):
        """处理与单位的碰撞"""
        self.has_collided = True
        self.collided_objects.append(unit)
        
        # 检查是否是友军
        if hasattr(unit, 'team') and unit.team == self.shooter_team:
            self.collided_with = 'friendly'
            # 友军不受伤害，子弹继续飞行
            return
        
        # 是敌军
        self.collided_with = 'unit'
        
        # 计算伤害
        damage = self._calculate_damage(unit)
        
        # 应用伤害
        if hasattr(unit, 'take_damage'):
            unit.take_damage(damage)
        
        # 如果子弹会爆炸，触发爆炸
        if self.is_explosive:
            self._trigger_explosion()
        else:
            self.is_active = False
    
    def _calculate_damage(self, unit) -> float:
        """根据护甲类型计算伤害"""
        base_damage = self.base_damage
        
        if hasattr(unit, 'armor_type'):
            armor_type = unit.armor_type
            # 根据护甲类型应用穿透系数
            if armor_type == ArmorType.NONE:
                penetration_multiplier = 1.0
            elif armor_type == ArmorType.LIGHT:
                penetration_multiplier = self.penetration[0] if len(self.penetration) > 0 else 1.0
            elif armor_type == ArmorType.MEDIUM:
                penetration_multiplier = self.penetration[1] if len(self.penetration) > 1 else 1.0
            elif armor_type == ArmorType.HEAVY:
                penetration_multiplier = self.penetration[2] if len(self.penetration) > 2 else 1.0
            else:
                penetration_multiplier = 1.0
            
            return base_damage * penetration_multiplier
        
        return base_damage
    
    def _trigger_explosion(self):
        """触发爆炸"""
        self.has_exploded = True
        self.explosion_timer = 0.0
        self.is_active = True  # 保持活跃以显示爆炸效果
        
        # 如果有爆炸图像，使用子弹位置作为爆炸中心
        if self.explosion_image:
            # 将子弹图像替换为爆炸图像
            self.image = self.explosion_image
            
            # 调整大小以匹配爆炸半径
            if self.explosion_radius > 0:
                scaled_size = (int(self.explosion_radius * 2), int(self.explosion_radius * 2))
                self.image = pygame.transform.scale(self.image, scaled_size)
                self.size = scaled_size
                self._update_bounding_box()
    
    def apply_explosion_damage(self, units: List[Any]) -> Dict[int, float]:
        """
        应用爆炸伤害到范围内的单位
        """
        if not self.has_exploded or not self.is_explosive:
            return {}
        
        damage_map = {}
        explosion_center = self.position
        
        for unit in units:
            if not unit.is_alive:
                continue
                
            if unit.team == self.shooter_team:
                continue
            
            # 检查单位是否在爆炸范围内
            dx = unit.position[0] - explosion_center[0]
            dy = unit.position[1] - explosion_center[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            if distance <= self.explosion_radius:
                distance_factor = 1.0 - (distance / self.explosion_radius)
                damage = self.base_damage * self.explosion_damage_rate * distance_factor
                
                actual_damage = unit.take_damage(damage)
                damage_map[unit.id] = actual_damage
        
        return damage_map
    
    def draw(self, surface: pygame.Surface, camera_offset: Tuple[float, float] = (0, 0)) -> None:
        """绘制子弹"""
        if not self.is_active:
            return
        
        screen_x = self.position[0] - camera_offset[0]
        screen_y = self.position[1] - camera_offset[1]
        
        if self.image:
            # 如果需要旋转，根据速度方向旋转图像
            if self.rotation_angle != 0:
                rotated_image = pygame.transform.rotate(self.image, -self.rotation_angle)
                image_rect = rotated_image.get_rect(center=(screen_x, screen_y))
                surface.blit(rotated_image, image_rect)
            else:
                image_rect = self.image.get_rect(center=(screen_x, screen_y))
                surface.blit(self.image, image_rect)
        
        # 调试绘制：爆炸范围
        if self.has_exploded and self.is_explosive and (DEBUG_MODE or DRAW_BULLET_EXPLOSION_RANGE):
            pygame.draw.circle(
                surface, 
                (255, 100, 100, 128),  # 半透明红色
                (int(screen_x), int(screen_y)),
                int(self.explosion_radius),
                2  # 线宽
            )
        
        # 调试绘制：碰撞箱
        if (DEBUG_MODE or DRAW_BULLET_BOUNDING_BOX) and self.bounding_box:
            debug_rect = pygame.Rect(
                self.bounding_box.x - camera_offset[0],
                self.bounding_box.y - camera_offset[1],
                self.bounding_box.width,
                self.bounding_box.height
            )
            pygame.draw.rect(surface, (255, 0, 0), debug_rect, 1)
    
    def get_info(self) -> Dict[str, Any]:
        """获取子弹信息"""
        return {
            "id": self.id,
            "shooter_id": self.shooter_id,
            "shooter_team": self.shooter_team.value if hasattr(self.shooter_team, 'value') else str(self.shooter_team),
            "position": self.position,
            "speed": self.speed,
            "damage": self.base_damage,
            "is_explosive": self.is_explosive,
            "explosion_radius": self.explosion_radius,
            "lifetime": self.lifetime,
            "max_lifetime": self.max_lifetime,
            "is_active": self.is_active,
            "has_collided": self.has_collided,
            "has_exploded": self.has_exploded,
            "distance_traveled": self.distance_traveled
        }
    
    def save_to_file(self, file_name: str = "default_bullet.json") -> bool:
        """保存子弹配置到文件"""
        try:
            save_dir = DEFAULT_BULLET_PATH
            os.makedirs(save_dir, exist_ok=True)
            
            filepath = os.path.join(save_dir, file_name)
            
            config = {
                "bullet_type": self.__class__.__name__,
                "image_path": self.image_path,
                "size": self.size,
                "max_lifetime": self.max_lifetime,
                "speed_rate": self.speed_rate,
                "damage_rate": self.damage_rate,
                "penetration": self.penetration,
                "is_explosive": self.is_explosive,
                "explosion_radius": self.explosion_radius,
                "explosion_damage_rate": self.explosion_damage_rate,
                "explosion_image_path": self.explosion_image_path
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            print(f"子弹配置已保存到: {filepath}")
            return True
            
        except Exception as e:
            print(f"保存子弹配置失败: {e}")
            return False
    
    def save(self) -> bool:
        """便捷保存方法"""
        save_path = get_next_filename(DEFAULT_BULLET_PATH, 'default_bullet', '.json')
        return self.save_to_file(save_path)
    
    @classmethod
    def load_from_file(cls, file_name: str = "default_bullet.json") -> Optional[Dict[str, Any]]:
        """从文件加载子弹配置"""
        try:
            filepath = os.path.join(DEFAULT_BULLET_PATH, file_name)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"子弹配置已从 {filepath} 加载")
            return config
            
        except FileNotFoundError:
            print(f"子弹配置文件不存在: {filepath}")
            return None
        except Exception as e:
            print(f"加载子弹配置失败: {e}")
            return None