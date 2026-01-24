'''
    一个坦克类
'''

import pygame
import math
from Unit.BaseUnit import BaseUnit
from Parameter import *

class Tank(BaseUnit):
    
    def __init__(self, unit_id, unit_team):
        
        self.body_image_path = 'Unit/Tank/tank.png'
        self.turret_image_path = 'Unit/Tank/turret.png'
        
        super().__init__(unit_id, unit_team, self.body_image_path, self.turret_image_path)
        
        self.unit_type = 'tank'
        
        self.max_speed_rate = 1.0
        self.max_acceleration_rate = UNIT_ACC_INF
        self.max_angular_speed_rate = UNIT_ANGULAR_SPEED_INF
        self.turret_angular_speed_rate = UNIT_TURRET_ANGULAR_SPEED_INF
        self.max_health_rate = 1.0
        
        self.max_speed = UNIT_SPEED * self.max_speed_rate
        self.max_accerattion = UNIT_ACC * self.max_accerattion_rate
        self.min_accerattion = -self.max_accerattion
        self.max_angular_speed = UNIT_ANGULAR_SPEED * self.max_angular_speed_rate
        self.turret_angular_speed = UNIT_TURRET_ANGULAR_SPEED * self.turret_angular_speed_rate
        self.max_health = UNIT_HEALTH * self.max_health_rate
        
        self.armor_type = ArmorType.NONE
        self.ammunition_types = ['bullet']
        self.ammo_switch_time = UNIT_AMMO_SWITCH_TIME
        
        self.health = self.max_health
        
        if self.ammunition_types:
            self.current_ammunition = self.ammunition_types[0]
        
        self._update_bounding_box()
        
    def fire(self, bullet_class):
        if self.is_switching_ammo:
            # 坦克正在切换弹药
            return None
        
        if not self.current_ammunition:
            # 当前弹药类型没有弹药
            return None
        
        turret_angle_rad = math.radians(self.turret_direction_angle)
        bullet_start_x = self.position[0] + math.cos(turret_angle_rad) * (self.size[0] / 2 + 5)
        bullet_start_y = self.position[1] + math.sin(turret_angle_rad) * (self.size[1] / 2 + 5)
        
        bullet_direction = (math.cos(turret_angle_rad), math.sin(turret_angle_rad))
        
        try:
            bullet = bullet_class(
                projectile_id=f"bullet_{self.id}_{id(self)}",  # 生成唯一ID
                shooter_id=self.id,
                position=(bullet_start_x, bullet_start_y),
                velocity_direction=bullet_direction
            )
            
            # 根据当前弹药类型设置子弹属性
            self._configure_bullet_for_ammo(bullet)
            
            return bullet
            
        except Exception as e:
            print(f"创建子弹时出错: {e}")
            return None          
        
    def _configure_bullet_for_ammo(self, bullet):
        """
        根据当前弹药类型配置子弹属性
        """
        if self.current_ammunition == "bullet":
            bullet.damage_rate = 1.0
            bullet.penetration = [1.0, 1.0, 1.0]
            bullet.speed_rate = 1.0
            bullet.is_explosive = False
            
    def switch_ammunition(self, ammo_type):
        if ammo_type not in self.ammunition_types:
            print(f"坦克 {self.id} 没有 {ammo_type} 类型弹药")
            return False
        
        if ammo_type == self.current_ammunition:
            return True
        
        self.is_switching_ammo = True
        self.reload_timer = self.ammo_switch_time
        
        self.target_ammunition = ammo_type
        
        return True     
    
    def _complete_ammo_switch(self):
        if hasattr(self, 'target_ammunition'):
            self.current_ammunition = self.target_ammunition
            delattr(self, 'target_ammunition')
        self.is_switching_ammo = False
        
    def set_movement(self, forward=False, backward=False):
        """
        设置坦克前进或后退
        """
        if forward and not backward:
            self.accerattion = self.max_accerattion
        elif backward and not forward:
            self.accerattion = self.min_accerattion
        else:
            self.accerattion = 0
    
    def set_turning(self, left=False, right=False):
        """
        设置坦克转向
        """
        if left and not right:
            self.angular_speed = -self.max_angular_speed
        elif right and not left:
            self.angular_speed = self.max_angular_speed
        else:
            self.angular_speed = 0
    
    def set_turret_target_to_mouse(self, mouse_pos, camera_offset):
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
    
    def update_with_collision(self, delta_time, obstacles):
        """
        更新坦克状态并处理碰撞
        """
        old_position = self.position
        old_bounding_box = self.bounding_box.copy() if self.bounding_box else None
        
        is_alive = self.update(delta_time)
        
        if self.is_switching_ammo and self.reload_timer <= 0:
            self._complete_ammo_switch()
        
        # 检查与障碍物的碰撞
        if self.bounding_box:
            for obstacle in obstacles:
                if self.bounding_box.colliderect(obstacle):
                    # 发生碰撞，恢复到之前的位置
                    self.position = old_position
                    self._update_bounding_box()
                    self.speed = 0  # 停止移动
                    return is_alive
        
        return is_alive
            
    def update(self, delta_time):
        is_alive = super().update(delta_time)
        
        if self.is_switching_ammo and self.reload_timer <= 0:
            self._complete_ammo_switch()
        
        return is_alive
    
    def get_info(self):
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
    
    def draw(self, surface, camera_offset=(0, 0)):
        if not self.is_alive:
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
        self._draw_health_bar(surface, screen_x, screen_y)
    
    def _draw_health_bar(self, surface, x, y):
        """
        绘制生命条
        """
        bar_width = 40
        bar_height = 4
        
        bar_x = x - bar_width / 2
        bar_y = y - self.size[1] / 2 - 10
        
        pygame.draw.rect(surface, (255, 0, 0), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        health_percentage = self.health / self.max_health
        current_width = bar_width * health_percentage
        pygame.draw.rect(surface, (0, 255, 0), 
                        (bar_x, bar_y, current_width, bar_height))
    
    def take_damage(self, damage_amount):
        """
        坦克承受伤害
        """
        self.health -= damage_amount
        
        if self.health <= 0:
            self.health = 0
            self.is_alive = False
            print(f"坦克 {self.id} 被摧毁")
        
        return damage_amount


def create_tank(unit_id, unit_team, position=(0, 0)):
    tank = Tank(unit_id, unit_team)
    tank.position = position
    return tank            
