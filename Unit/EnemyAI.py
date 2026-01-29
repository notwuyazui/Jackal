'''
    用于控制敌方ai的行为逻辑
'''
import pygame
import math
import random
from Unit.Tank.Tank import *
from Map.Map import *
from GameMode import *
from Bullet.NormalShell.NormalShell import NormalShell

class EnemyAI:
    """敌方坦克AI控制器"""
    
    def __init__(self, enemy_tank, player_tank, bullet_manager, map_obstacles):
        self.enemy = enemy_tank
        self.player = player_tank
        self.bullet_manager = bullet_manager
        self.map_obstacles = map_obstacles
        
        # AI状态
        self.ai_state = "patrol"  # patrol, attack, retreat
        self.state_timer = 0.0
        self.change_state_time = random.uniform(3.0, 8.0)  # 状态改变时间
        
        # 移动控制
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        
        # 射击控制
        self.fire_cooldown = 0.0
        self.fire_cooldown_max = random.uniform(1.0, 2.0)  # 射击冷却时间
        self.can_see_player = False
        
        # 巡逻点
        self.patrol_points = []
        self.current_patrol_index = 0
        self.generate_patrol_points()
    
    def generate_patrol_points(self, game_map=None):
        """为敌人生成巡逻点"""
        base_x, base_y = self.enemy.position
        
        # 生成4个围绕当前位置的巡逻点
        for i in range(4):
            angle = i * 90
            distance = random.uniform(100, 200)
            angle_rad = math.radians(angle)
            point_x = base_x + math.cos(angle_rad) * distance
            point_y = base_y + math.sin(angle_rad) * distance
            
            # 确保巡逻点在地图范围内
            map_width, map_height = game_map.get_map_size() if 'game_map' in globals() else (1000, 1000)
            point_x = max(50, min(point_x, map_width - 50))
            point_y = max(50, min(point_y, map_height - 50))
            
            self.patrol_points.append((point_x, point_y))
    
    def update(self, delta_time):
        """更新AI逻辑"""
        if not self.enemy.is_alive:
            return
        
        # 更新计时器
        self.state_timer += delta_time
        if self.state_timer >= self.change_state_time:
            self.change_state()
            self.state_timer = 0.0
        
        # 检查是否能看见玩家
        self.check_line_of_sight()
        
        # 更新射击冷却
        if self.fire_cooldown > 0:
            self.fire_cooldown -= delta_time
        
        # 根据状态执行行为
        if self.ai_state == "patrol":
            self.patrol_behavior(delta_time)
        elif self.ai_state == "attack":
            self.attack_behavior(delta_time)
        elif self.ai_state == "retreat":
            self.retreat_behavior(delta_time)
        
        # 更新敌人移动
        self.enemy.set_movement(forward=self.move_forward, backward=self.move_backward)
        self.enemy.set_turning(left=self.turn_left, right=self.turn_right)
        
        # 更新敌人
        self.enemy.update(delta_time, self.map_obstacles)
        
        # 如果能看到玩家且可以射击，则开火
        if self.can_see_player and self.fire_cooldown <= 0:
            self.try_fire()
    
    def change_state(self):
        """改变AI状态"""
        states = ["patrol", "attack", "retreat"]
        
        # 如果能看到玩家，更倾向于攻击
        if self.can_see_player:
            weights = [0.2, 0.7, 0.1]  # 攻击状态权重更高
        else:
            weights = [0.7, 0.2, 0.1]  # 巡逻状态权重更高
        
        # 随机选择下一个状态
        self.ai_state = random.choices(states, weights=weights)[0]
        self.change_state_time = random.uniform(3.0, 8.0)
        
        # 重置移动控制
        self.move_forward = False
        self.move_backward = False
        self.turn_left = False
        self.turn_right = False
        
        if DEBUG_MODE or ENEMY_STATE_TEXT:
            print(f"敌人 {self.enemy.id} 改变状态为: {self.ai_state}")
    
    def check_line_of_sight(self):
        """检查是否有玩家的视线（简单版本，不考虑障碍物）"""
        if not self.player.is_alive:
            self.can_see_player = False
            return
        
        # 计算到玩家的距离
        dx = self.player.position[0] - self.enemy.position[0]
        dy = self.player.position[1] - self.enemy.position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # 如果玩家在视线范围内（距离和角度）
        max_sight_distance = 300
        self.can_see_player = distance <= max_sight_distance
    
    def patrol_behavior(self, delta_time):
        """巡逻行为"""
        if not self.patrol_points:
            return
        
        # 获取当前巡逻点
        target_x, target_y = self.patrol_points[self.current_patrol_index]
        
        # 计算到巡逻点的方向
        dx = target_x - self.enemy.position[0]
        dy = target_y - self.enemy.position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # 如果接近巡逻点，切换到下一个
        if distance < 20:
            self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
            target_x, target_y = self.patrol_points[self.current_patrol_index]
            dx = target_x - self.enemy.position[0]
            dy = target_y - self.enemy.position[1]
        
        # 计算目标角度
        target_angle = math.degrees(math.atan2(dy, dx)) + 90
        target_angle %= 360
        
        # 设置炮塔目标角度
        self.enemy.turret_target_angle = target_angle
        
        # 设置移动：朝目标前进
        self.move_forward = True
        self.move_backward = False
        
        # 转向逻辑
        angle_diff = self.get_angle_difference(self.enemy.direction_angle, target_angle)
        if abs(angle_diff) > 15:
            if angle_diff > 0:
                self.turn_right = True
                self.turn_left = False
            else:
                self.turn_left = True
                self.turn_right = False
        else:
            self.turn_left = False
            self.turn_right = False
    
    def attack_behavior(self, delta_time):
        """攻击行为"""
        if not self.player.is_alive:
            return
        
        # 计算到玩家的方向
        dx = self.player.position[0] - self.enemy.position[0]
        dy = self.player.position[1] - self.enemy.position[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # 计算目标角度
        target_angle = math.degrees(math.atan2(dy, dx)) + 90
        target_angle %= 360
        
        # 设置炮塔目标角度
        self.enemy.turret_target_angle = target_angle
        
        # 攻击行为：保持一定距离
        ideal_distance = 150
        if distance > ideal_distance + 50:
            # 太远，前进
            self.move_forward = True
            self.move_backward = False
        elif distance < ideal_distance - 50:
            # 太近，后退
            self.move_forward = False
            self.move_backward = True
        else:
            # 理想距离，停止移动
            self.move_forward = False
            self.move_backward = False
        
        # 转向以面对玩家
        angle_diff = self.get_angle_difference(self.enemy.direction_angle, target_angle)
        if abs(angle_diff) > 15:
            if angle_diff > 0:
                self.turn_right = True
                self.turn_left = False
            else:
                self.turn_left = True
                self.turn_right = False
        else:
            self.turn_left = False
            self.turn_right = False
    
    def retreat_behavior(self, delta_time):
        """撤退行为"""
        if not self.player.is_alive:
            return
        
        # 计算从玩家到敌人的方向（相反方向）
        dx = self.enemy.position[0] - self.player.position[0]
        dy = self.enemy.position[1] - self.player.position[1]
        
        # 计算撤退方向
        retreat_angle = math.degrees(math.atan2(dy, dx)) + 90
        retreat_angle %= 360
        
        # 设置炮塔目标角度（指向撤退方向）
        self.enemy.turret_target_angle = retreat_angle
        
        # 撤退行为：后退
        self.move_backward = True
        self.move_forward = False
        
        # 转向以面对撤退方向
        angle_diff = self.get_angle_difference(self.enemy.direction_angle, retreat_angle)
        if abs(angle_diff) > 15:
            if angle_diff > 0:
                self.turn_right = True
                self.turn_left = False
            else:
                self.turn_left = True
                self.turn_right = False
        else:
            self.turn_left = False
            self.turn_right = False
    
    def try_fire(self):
        """尝试开火"""
        if self.fire_cooldown <= 0 and self.can_see_player:
            # 检查炮塔是否大致指向玩家
            dx = self.player.position[0] - self.enemy.position[0]
            dy = self.player.position[1] - self.enemy.position[1]
            target_angle = math.degrees(math.atan2(dy, dx)) + 90
            target_angle %= 360
            
            angle_diff = self.get_angle_difference(self.enemy.turret_direction_angle, target_angle)
            
            # 如果炮塔指向玩家（误差在20度内），则开火
            if abs(angle_diff) < 20:
                bullet = self.enemy.fire(NormalShell)
                if bullet:
                    self.bullet_manager.add_bullet(bullet)
                    self.fire_cooldown = self.fire_cooldown_max
                    if DEBUG_MODE or ENEMY_STATE_TEXT:
                        print(f"敌人 {self.enemy.id} 开火!")
    
    def get_angle_difference(self, angle1, angle2):
        """计算两个角度之间的最小差值"""
        diff = angle2 - angle1
        diff = (diff + 180) % 360 - 180
        return diff