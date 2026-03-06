"""
    敌人 AI 控制模块（增强版）
    管理使用 AI 的敌方单位，自动决策移动、攻击、躲避子弹与障碍物
"""

import pygame
import math
import random
from typing import List, Optional, Tuple
from Parameter import *
from Unit.BaseUnit import BaseUnit


class EnemyAI:
    def __init__(self, unit: BaseUnit, unit_manager, bullet_manager, obstacles):
        """
        :param unit: 受此 AI 控制的单位
        :param unit_manager: 单位管理器，用于获取所有单位信息及障碍物
        :param bullet_manager: 子弹管理器，用于获取子弹列表并发射子弹
        """
        self.unit = unit
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        self.obstacles = obstacles

        # AI 状态
        self.ai_state = "idle"          # 可能的状态：idle, chase, attack, retreat, dodge
        self.target_unit = None         # 当前目标单位
        self.fire_cooldown = 0.0
        self.fire_cooldown_max = 0.5    # 发射间隔（秒）
        
        # 躲避参数
        self.dodge_threshold = 100      # 子弹威胁距离阈值（像素）
        self.dodge_weight = 2.0         # 躲避方向权重
        self.obstacle_avoid_distance = 50  # 障碍物检测距离
        
        # 移动行为参数
        self.desired_distance = 200      # 理想攻击距离（像素）
        self.strafe_enabled = True       # 是否允许横向移动（通过转向实现）

    def update(self, delta_time: float, unit_manager, bullet_manager, obstacles):
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        self.obstacles = obstacles
        
        """每帧更新 AI 决策"""
        # 更新开火冷却
        if self.fire_cooldown > 0:
            self.fire_cooldown -= delta_time

        # 获取敌军与友军
        enemy_units = self._get_enemy_units()
        if not enemy_units:
            self.ai_state = "idle"
            self.unit.set_movement(False, False)
            self.unit.set_turning(False, False)
            return

        # 选择最近的敌军作为目标
        closest_enemy = self._get_closest_unit(enemy_units)
        self.target_unit = closest_enemy

        # ========== 1. 感知威胁（敌方子弹）==========
        threat_direction = self._get_threat_avoidance_direction()

        # ========== 2. 感知障碍物 ==========
        avoid_obstacle_direction = self._get_obstacle_avoidance_direction()

        # ========== 3. 计算期望移动方向（融合）==========
        target_pos = closest_enemy.position
        desired_direction = self._calculate_desired_direction(
            target_pos, threat_direction, avoid_obstacle_direction
        )

        # ========== 4. 执行移动 ==========
        self._move_towards_direction(desired_direction)

        # ========== 5. 炮塔指向目标 ==========
        self._aim_at_target(closest_enemy)

        # ========== 6. 攻击决策 ==========
        self._try_fire(closest_enemy)

        # 更新状态机（简单状态，用于调试）
        if threat_direction is not None:
            self.ai_state = "dodge"
        elif avoid_obstacle_direction is not None:
            self.ai_state = "avoid_obstacle"
        else:
            self.ai_state = "chase"

    # ----------------- 感知方法 -----------------
    def _get_enemy_units(self) -> List[BaseUnit]:
        """获取所有敌军单位（与控制的单位不同队伍）"""
        return [u for u in self.unit_manager.units if u.is_alive and u.team != self.unit.team]

    def _get_friendly_units(self) -> List[BaseUnit]:
        """获取所有友军单位（同队伍且不是自己）"""
        return [u for u in self.unit_manager.units if u.is_alive and u.team == self.unit.team and u.id != self.unit.id]

    def _get_closest_unit(self, units: List[BaseUnit]) -> Optional[BaseUnit]:
        """返回距离控制单位最近的有效单位"""
        if not units:
            return None
        closest = None
        min_dist = float('inf')
        for u in units:
            dx = u.position[0] - self.unit.position[0]
            dy = u.position[1] - self.unit.position[1]
            dist = dx * dx + dy * dy
            if dist < min_dist:
                min_dist = dist
                closest = u
        return closest

    def _get_threat_avoidance_direction(self) -> Optional[Tuple[float, float]]:
        """
        检测威胁子弹，返回躲避方向向量（单位向量）
        若无威胁，返回 None
        """
        threatening_bullets = []
        my_pos = self.unit.position
        my_radius = max(self.unit.size) / 2  # 粗略半径

        for bullet in self.bullet_manager.bullets:
            # 忽略己方子弹
            if bullet.shooter_team == self.unit.team:
                continue
            # 粗略预测：计算子弹到自己的相对位置和速度
            dx = my_pos[0] - bullet.position[0]
            dy = my_pos[1] - bullet.position[1]
            dist = math.hypot(dx, dy)
            if dist > self.dodge_threshold:
                continue

            # 计算子弹相对速度方向（子弹速度向量）
            vx, vy = bullet.velocity
            # 如果子弹正在远离自己，忽略
            rel_dot = dx * vx + dy * vy
            if rel_dot > 0:  # 子弹远离
                continue

            # 简单预测：若子弹直线运动，计算是否会撞到自己
            # 这里简化：只要距离小于阈值且朝向自己，就视为威胁
            threatening_bullets.append((dx, dy, dist))

        if not threatening_bullets:
            return None

        # 计算平均躲避方向（垂直于所有威胁方向的平均）
        # 简单策略：选择最危险子弹（最近）的垂直方向
        closest_threat = min(threatening_bullets, key=lambda t: t[2])
        dx, dy, _ = closest_threat
        # 垂直方向（左或右，随机选择一个）
        if random.random() < 0.5:
            avoid_dir = (-dy, dx)   # 逆时针垂直
        else:
            avoid_dir = (dy, -dx)   # 顺时针垂直
        # 归一化
        length = math.hypot(*avoid_dir)
        if length > 0:
            return (avoid_dir[0] / length, avoid_dir[1] / length)
        return None

    def _get_obstacle_avoidance_direction(self) -> Optional[Tuple[float, float]]:
        """
        检测前方障碍物，返回避开方向向量
        若无障碍，返回 None
        """
        # 获取当前移动方向（车身朝向）
        forward_angle = math.radians(self.unit.direction_angle - 90)
        forward_dir = (math.cos(forward_angle), math.sin(forward_angle))
        # 检测点：前方一段距离
        check_distance = self.obstacle_avoid_distance + max(self.unit.size) / 2
        check_x = self.unit.position[0] + forward_dir[0] * check_distance
        check_y = self.unit.position[1] + forward_dir[1] * check_distance

        # 创建检测矩形（略小于单位大小）
        detect_rect = pygame.Rect(
            check_x - self.unit.size[0] / 2,
            check_y - self.unit.size[1] / 2,
            self.unit.size[0],
            self.unit.size[1]
        )

        # 检查与障碍物碰撞
        for obs in self.obstacles:
            if detect_rect.colliderect(obs):
                # 有障碍，计算避开方向：垂直于当前方向，并偏向于目标方向
                # 简单策略：左转或右转，取决于哪边更空旷
                # 这里先返回一个垂直于前进方向的方向
                avoid_dir = (-forward_dir[1], forward_dir[0])  # 左转方向
                return avoid_dir
        return None

    # ----------------- 决策融合 -----------------
    def _calculate_desired_direction(self, target_pos: Tuple[float, float],
                                      threat_dir: Optional[Tuple[float, float]],
                                      avoid_dir: Optional[Tuple[float, float]]) -> Tuple[float, float]:
        """
        融合目标方向、躲避子弹方向和避障方向，返回最终期望移动方向（单位向量）
        """
        # 目标方向（指向敌人）
        dx = target_pos[0] - self.unit.position[0]
        dy = target_pos[1] - self.unit.position[1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            target_dir = (dx / dist, dy / dist)
        else:
            target_dir = (1.0, 0.0)

        # 优先级：威胁 > 障碍 > 目标
        if threat_dir is not None:
            # 如果威胁方向存在，直接使用威胁方向（也可与目标方向混合，但先简单处理）
            return threat_dir
        elif avoid_dir is not None:
            # 避障方向可能与目标方向混合，此处简单混合：避障方向占主导，但仍倾向于目标
            # 加权平均后归一化
            mix_x = avoid_dir[0] * 2.0 + target_dir[0]
            mix_y = avoid_dir[1] * 2.0 + target_dir[1]
            length = math.hypot(mix_x, mix_y)
            if length > 0:
                return (mix_x / length, mix_y / length)
            else:
                return target_dir
        else:
            return target_dir

    # ----------------- 执行移动 -----------------
    def _move_towards_direction(self, desired_dir: Tuple[float, float]):
        """
        根据期望方向控制车身转向和前进
        """
        # 计算期望方向对应的角度（车身朝向角度，0°为向上）
        desired_angle = math.degrees(math.atan2(desired_dir[1], desired_dir[0])) + 90
        desired_angle %= 360

        # 计算当前方向与期望方向的差值
        angle_diff = self.unit.get_angle_difference(self.unit.direction_angle, desired_angle)

        # 转向控制
        turn_threshold = 5  # 死区
        if abs(angle_diff) > turn_threshold:
            if angle_diff > 0:
                self.unit.set_turning(left=False, right=True)   # 顺时针
            else:
                self.unit.set_turning(left=True, right=False)   # 逆时针
        else:
            self.unit.set_turning(False, False)

        # 前进/后退控制：如果角度差较大，可以减速，但这里简单保持前进
        self.unit.set_movement(forward=True, backward=False)

    def _aim_at_target(self, target: BaseUnit):
        """炮塔指向目标"""
        dx = target.position[0] - self.unit.position[0]
        dy = target.position[1] - self.unit.position[1]
        target_angle = math.degrees(math.atan2(dy, dx)) + 90
        target_angle %= 360
        self.unit.turret_target_angle = target_angle

    # ----------------- 攻击决策 -----------------
    def _try_fire(self, target: BaseUnit):
        """尝试开火，检查视线和冷却"""
        if self.fire_cooldown <= 0 and self._can_see_target(target):
            # 炮塔指向误差检查
            angle_diff = self.unit.get_angle_difference(
                self.unit.turret_direction_angle, self.unit.turret_target_angle
            )
            if abs(angle_diff) < 10:
                bullet = self.unit.fire()
                if bullet:
                    self.bullet_manager.add_bullet(bullet)
                    self.fire_cooldown = self.fire_cooldown_max

    def _can_see_target(self, target: BaseUnit) -> bool:
        """检查从控制单位到目标之间是否有障碍物阻挡视线"""
        start = self.unit.position
        end = target.position
        for obs in self.obstacles:
            if obs.clipline(start, end):
                return False
        return True