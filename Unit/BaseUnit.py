'''
    描述战斗单位的基类
'''

import pygame
import math
from Parameter import *

class BaseUnit:
    def __init__(self, unit_id, unit_team, unit_type, body_image_path, turret_image_path, 
                 max_speed_rate=1.0, 
                 max_acceleration_rate=UNIT_ACC_INF, 
                 min_acceleration_rate=-UNIT_ACC_INF, 
                 max_angular_speed_rate=UNIT_ANGULAR_SPEED_INF, 
                 turret_angular_speed_rate=UNIT_TURRET_ANGULAR_SPEED_INF, 
                 max_health_rate=1.0, 
                 armor_type=ArmorType.NONE, 
                 ammunition_types=[], 
                 ammo_switch_time=UNIT_AMMO_SWITCH_TIME):
        
        # 基本信息
        self.id = unit_id
        self.team = unit_team
        self.unit_type = unit_type
        self.body_image_path = body_image_path
        self.turret_image_path = turret_image_path
        self.body_image = self.load_image(self.body_image_path) if self.body_image_path else None
        self.turret_image = self.load_image(self.turret_image_path) if self.turret_image_path else None
        self.size = self.body_image.get_size()
        
        # 基本属性
        self.max_speed_rate = max_speed_rate
        self.max_acceleration_rate = max_acceleration_rate
        self.min_acceleration_rate = min_acceleration_rate
        self.max_angular_speed_rate = max_angular_speed_rate
        self.turret_angular_speed_rate = turret_angular_speed_rate
        self.max_health_rate = max_health_rate
        self.max_speed = UNIT_SPEED * self.max_speed_rate                                       # 最大速度  
        self.max_acceleration = UNIT_ACC * self.max_acceleration_rate                             # 最大加速度
        self.min_acceleration = UNIT_ACC * self.min_acceleration_rate                             # 最小加速度
        self.max_angular_speed = UNIT_ANGULAR_SPEED * self.max_angular_speed_rate               # 最大角速度
        self.turret_angular_speed = UNIT_TURRET_ANGULAR_SPEED * self.turret_angular_speed_rate  # 炮塔转动角速度
        self.max_health = UNIT_HEALTH * self.max_health_rate                                    # 最大生命值
        self.armor_type = armor_type                                                            # 护甲类型
        self.ammunition_types = ammunition_types                                                # 单位拥有弹种
        self.ammo_switch_time =  ammo_switch_time                                               # 单位切换弹种时间
        
        # 实时属性
        self.position = (0.0, 0.0)
        self.speed = 0.0
        self.direction_angle = 0.0              # 单位朝向角度
        self.turret_direction_angle = 0.0       # 单位炮塔朝向角度
        self.acceleration = 0.0
        self.angular_speed = 0.0
        self.health = self.max_health
        self.bounding_box = None                # 碰撞箱，pygame.Rect对象
        self.velocity = self.cal_velocity()     # 速度向量
        self.current_ammunition = ""            # 单位当前选中弹种
        
        if self.ammunition_types:
            self.current_ammunition = self.ammunition_types[0]
        
        # 预留属性
        self.is_alive = True
        self.reload_timer = 0.0         # 切换弹种剩余时间计时器
        self.reward = 0.0
        self.turret_target_angle = 0.0          # 炮塔目标角度
        self.is_switching_ammo = False          # 是否正在切换弹药
        
        # 初始化碰撞箱
        self._update_bounding_box()
        
    def cal_velocity(self):
        adjusted_angle = self.direction_angle - 90
        angle_rad = math.radians(adjusted_angle)
        return (
            self.speed * math.cos(angle_rad),
            self.speed * math.sin(angle_rad)
        )
    
    def load_image(self, image_path):
        try:
            return pygame.image.load(image_path)
        except:
            print(f"Warning: Cannot load image: {image_path}")
            return None

    def update(self, delta_time):
        if not self.is_alive:
            return False
        
        if self.health <= 0:
            self.is_alive = False
            return False
        
        self._update_ammo_switch(delta_time)         # 更新弹种切换计时器
        self._update_speed(delta_time)               # 更新速度
        self._update_direction(delta_time)           # 更新朝向
        self._update_turret_direction(delta_time)    # 更新炮塔朝向
        self._update_position(delta_time)            # 更新位置
        self._update_bounding_box()                  # 更新碰撞箱
        self.velocity = self.cal_velocity()         # 更新速度向量
        
        return True
    
    def _update_ammo_switch(self, delta_time):
        """更新弹药切换状态"""
        if self.is_switching_ammo and self.reload_timer > 0:
            self.reload_timer -= delta_time
            if self.reload_timer <= 0:
                self.reload_timer = 0
                self.is_switching_ammo = False
    
    def _update_speed(self, delta_time):
        """更新速度"""
        # 应用加速度
        self.speed += self.acceleration * delta_time
        
        # 限制速度范围
        if self.speed > self.max_speed:
            self.speed = self.max_speed
        elif self.speed < -self.max_speed:
            self.speed = -self.max_speed
    
    def _update_direction(self, delta_time):
        """更新单位朝向角度"""
        if self.angular_speed != 0:
            self.direction_angle += self.angular_speed * delta_time
            self.direction_angle = self.normalize_angle(self.direction_angle)
            if abs(self.angular_speed) > self.max_angular_speed:
                self.angular_speed = self.max_angular_speed if self.angular_speed > 0 else -self.max_angular_speed
    
    def _update_turret_direction(self, delta_time):
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
    
    def _update_position(self, delta_time):
        """根据速度更新位置"""
        dx = self.velocity[0] * delta_time
        dy = self.velocity[1] * delta_time
        x, y = self.position
        self.position = (x + dx, y + dy)
    
    def _update_bounding_box(self):
        if self.size[0] > 0 and self.size[1] > 0:
            x, y = self.position
            width, height = self.size
            self.bounding_box = pygame.Rect(x - width / 2, y - height / 2, width, height)
            
    def normalize_angle(self, angle):
        """将角度规范化到0-360度范围内"""
        angle %= 360
        if angle < 0:
            angle += 360
        return angle
    
    def get_angle_difference(self, angle1, angle2):
        """
        计算两个角度之间的最小差值(考虑360度循环)
        """
        diff = angle2 - angle1
        diff = (diff + 180) % 360 - 180
        return diff
