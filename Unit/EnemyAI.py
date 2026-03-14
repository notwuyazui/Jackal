"""
    敌人 AI 控制模块（增强版）
    实现智能行为：碰撞脱离、主动避障、子弹规避、目标跟踪与距离调整
"""

import pygame
import math
import random
from typing import List, Optional, Tuple
from Parameter import *
from Unit.BaseUnit import BaseUnit
from Bullet.BaseBullet import BaseBullet

# 导入子弹类用于获取射程
from Bullet.NormalShell.NormalShell import NormalShell
from Bullet.RocketShell.RocketShell import RocketShell
from Bullet.HeavyShell.HeavyShell import HeavyShell


class EnemyAI:
    # 内部参数（可根据需要调整）
    LOOK_AHEAD_TIME = 3.0           # 路径预测时间（秒）
    SAFE_DIST_FACTOR = 1.0           # 安全距离因子（相对于单位尺寸）
    BULLET_THREAT_RADIUS = 400       # 子弹威胁检测半径
    EVASION_SAMPLES = 8              # 脱离方向采样数
    MAX_SPEED_FACTOR = 1.0           # 移动速度因子（直接使用单位最大速度）

    def __init__(self, unit: BaseUnit, unit_manager, bullet_manager, game_map):
        self.unit = unit
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        self.game_map = game_map

        # 状态标识
        self.state = "idle"  # colliding, avoid_obstacle, evade_bullet, approach_target

        # 缓存当前弹药射程（每帧更新）
        self.current_ammo_range = 0.0

        # 用于记录上一次期望方向，平滑转向
        self.last_desired_dir = (1.0, 0.0)

    # ----------------- 核心更新 -----------------
    def update(self, delta_time: float, unit_manager, bullet_manager, game_map):
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        self.game_map = game_map

        # 更新弹药射程
        self.current_ammo_range = self._get_current_ammo_range()

        # 获取敌军
        enemy_units = self._get_enemy_units()
        if not enemy_units:
            self.state = "idle"
            self.unit.set_movement(False, False)
            self.unit.set_turning(False, False)
            return

        # 选择最近敌军作为目标
        target = self._get_closest_unit(enemy_units)

        # 炮塔始终指向目标
        self._aim_at_target(target)

        # 决策期望移动方向（按优先级）
        desired_dir = self._decide_desired_direction(target)

        # 执行移动
        self._move_towards_direction(desired_dir)

        # 攻击决策
        self._try_fire(target)

    # ----------------- 决策层次 -----------------
    def _decide_desired_direction(self, target: BaseUnit) -> Tuple[float, float]:
        """按优先级返回期望移动方向（单位向量）"""
        # 1. 碰撞脱离（当前已重叠）
        if self._check_collision_overlap():
            self.state = "colliding"
            return self._find_escape_direction()

        # 2. 路径障碍检测（预测即将碰撞）
        obstacle_dir = self._check_path_obstacle()
        if obstacle_dir is not None:
            self.state = "avoid_obstacle"
            return obstacle_dir

        # 3. 子弹规避
        evade_dir = self._get_threat_evasion_direction()
        if evade_dir is not None:
            self.state = "evade_bullet"
            return evade_dir

        # 4. 目标跟踪
        self.state = "approach_target"
        return self._get_target_approach_direction(target)

    # ----------------- 感知方法 -----------------
    def _get_enemy_units(self) -> List[BaseUnit]:
        """获取可见的敌军单位"""
        return [u for u in self.unit_manager.units if u.is_alive and u.visible and u.team != self.unit.team]

    def _get_closest_unit(self, units: List[BaseUnit]) -> Optional[BaseUnit]:
        """返回距离最近的有效单位"""
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

    def _aim_at_target(self, target: BaseUnit):
        """炮塔指向目标"""
        dx = target.position[0] - self.unit.position[0]
        dy = target.position[1] - self.unit.position[1]
        target_angle = math.degrees(math.atan2(dy, dx)) + 90
        target_angle %= 360
        self.unit.turret_target_angle = target_angle

    # ----------------- 碰撞脱离 -----------------
    def _check_collision_overlap(self) -> bool:
        """检查单位当前是否与任何障碍物重叠（碰撞）"""
        for obs in self.game_map.unit_obstacles:
            if self.unit.bounding_box.colliderect(obs):
                return True
        return False

    def _find_escape_direction(self) -> Tuple[float, float]:
        """
        当发生重叠时，寻找最空旷的方向作为脱离方向
        采样 8 个方向，发射射线检测障碍物距离，选择距离最大的方向
        """
        best_dir = (1.0, 0.0)
        max_dist = -1.0
        center = self.unit.position
        step = 2 * math.pi / self.EVASION_SAMPLES

        for i in range(self.EVASION_SAMPLES):
            angle = i * step
            dx = math.cos(angle)
            dy = math.sin(angle)
            # 从中心沿方向发射射线，检测与障碍物的交点距离
            ray_end = (center[0] + dx * 1000, center[1] + dy * 1000)  # 长射线
            closest_dist = 1000
            for obs in self.game_map.unit_obstacles:
                # 使用矩形与线段的碰撞检测（近似）
                # 这里简单用矩形与线段的交点，也可以使用更精确的算法
                # 为了效率，我们使用 pygame.Rect.clipline
                clipped = obs.clipline(center, ray_end)
                if clipped:
                    # 取最近的交点
                    for point in clipped:
                        dist = math.hypot(point[0] - center[0], point[1] - center[1])
                        if dist < closest_dist:
                            closest_dist = dist
            if closest_dist > max_dist:
                max_dist = closest_dist
                best_dir = (dx, dy)

        # 如果所有方向都很近（被包围），随机选一个
        if max_dist < self.unit.size[0] * 0.5:
            best_dir = (random.uniform(-1, 1), random.uniform(-1, 1))
            norm = math.hypot(*best_dir)
            if norm > 0:
                best_dir = (best_dir[0] / norm, best_dir[1] / norm)

        return best_dir

    # ----------------- 主动避障 -----------------
    def _check_path_obstacle(self) -> Optional[Tuple[float, float]]:
        """
        预测单位继续前进是否会与障碍物碰撞
        如果会，返回避障方向（单位向量）；否则返回 None
        """
        # 当前速度向量（用于预测移动）
        vel = self.unit.velocity
        speed = math.hypot(*vel)
        if speed < 1e-3:
            # 静止时，使用当前朝向作为预测方向
            forward_angle = math.radians(self.unit.direction_angle - 90)
            vel = (math.cos(forward_angle), math.sin(forward_angle))
            speed = 1.0

        # 预测距离
        predict_distance = speed * self.LOOK_AHEAD_TIME
        if predict_distance < self.unit.size[0] * 0.5:
            predict_distance = self.unit.size[0] * 0.5  # 至少半个车身

        # 预测位置
        dx = vel[0] / speed * predict_distance
        dy = vel[1] / speed * predict_distance
        future_pos = (self.unit.position[0] + dx, self.unit.position[1] + dy)

        # 创建预测矩形（大小与单位相同）
        future_rect = pygame.Rect(
            future_pos[0] - self.unit.size[0] / 2,
            future_pos[1] - self.unit.size[1] / 2,
            self.unit.size[0],
            self.unit.size[1]
        )

        # 检查预测矩形是否与障碍物碰撞
        for obs in self.game_map.unit_obstacles:
            if future_rect.colliderect(obs):
                # 有碰撞，需要计算避障方向
                return self._calculate_avoid_direction(vel)
        return None

    def _calculate_avoid_direction(self, forward_dir: Tuple[float, float]) -> Tuple[float, float]:
        """
        根据当前前进方向，计算避障方向（选择左右两侧中较空旷的一侧）
        发射左右两条射线，选择障碍物距离较远的一侧作为避障方向
        """
        # 左右向量（垂直于前进方向）
        left_dir = (-forward_dir[1], forward_dir[0])
        right_dir = (forward_dir[1], -forward_dir[0])

        # 检测左右方向的障碍物距离
        left_dist = self._ray_cast_distance(left_dir)
        right_dist = self._ray_cast_distance(right_dir)

        # 选择距离较远的一侧，并稍微偏向目标方向（可选）
        if left_dist > right_dist:
            return left_dir
        else:
            return right_dir

    def _ray_cast_distance(self, direction: Tuple[float, float], max_dist: float = 300) -> float:
        """从单位中心沿方向发射射线，返回与障碍物的最近距离（若无障碍则返回 max_dist）"""
        center = self.unit.position
        ray_end = (center[0] + direction[0] * max_dist, center[1] + direction[1] * max_dist)
        min_dist = max_dist
        for obs in self.game_map.unit_obstacles:
            clipped = obs.clipline(center, ray_end)
            if clipped:
                for point in clipped:
                    dist = math.hypot(point[0] - center[0], point[1] - center[1])
                    if dist < min_dist:
                        min_dist = dist
        return min_dist

    # ----------------- 子弹规避 -----------------
    def _get_threat_evasion_direction(self) -> Optional[Tuple[float, float]]:
        """
        检测威胁子弹，返回规避方向（单位向量）
        若无威胁，返回 None
        """
        threatening = []
        my_pos = self.unit.position
        my_rect = self.unit.bounding_box

        for bullet in self.bullet_manager.bullets:
            # 忽略己方子弹
            if bullet.shooter_team == self.unit.team:
                continue
            # 忽略距离过远的子弹
            dx = my_pos[0] - bullet.position[0]
            dy = my_pos[1] - bullet.position[1]
            dist = math.hypot(dx, dy)
            if dist > self.BULLET_THREAT_RADIUS:
                continue

            # 预测子弹是否会撞到自己（简化：直线运动）
            # 计算子弹到自己的相对速度方向
            vx, vy = bullet.velocity
            rel_dot = dx * vx + dy * vy
            if rel_dot > 0:  # 子弹远离
                continue

            # 粗略估计：如果子弹直线运动，计算最近距离
            # 这里简单处理：只要在威胁半径内且朝向自己，就视为威胁
            # 计算垂直于子弹方向的向量作为躲避方向
            # 取子弹位置到自己的向量，旋转90度
            if dist < 1e-3:
                continue
            nx = -dy / dist   # 垂直方向（顺时针90度）
            ny = dx / dist
            threatening.append((nx, ny))

        if not threatening:
            return None

        # 平均所有威胁方向的垂直方向作为规避方向
        sum_x = sum(v[0] for v in threatening)
        sum_y = sum(v[1] for v in threatening)
        norm = math.hypot(sum_x, sum_y)
        if norm > 0:
            return (sum_x / norm, sum_y / norm)
        else:
            return None

    # ----------------- 目标跟踪与距离调整 -----------------
    def _get_target_approach_direction(self, target: BaseUnit) -> Tuple[float, float]:
        """
        返回指向目标的单位向量，并根据射程调整
        如果目标距离大于射程，则直接指向目标；否则，保持当前方向或随机徘徊（可选）
        """
        dx = target.position[0] - self.unit.position[0]
        dy = target.position[1] - self.unit.position[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return self.last_desired_dir

        target_dir = (dx / dist, dy / dist)

        # 射程调整：如果目标超出射程，则靠近；否则可以保持或稍微调整
        if dist > self.current_ammo_range * 0.9:  # 留10%余量
            # 需要靠近
            return target_dir
        else:
            # 在射程内，可以适当横向移动以增加命中率，这里简单返回目标方向
            # 也可随机偏转一个角度，但保持整体向目标
            # 为增加战术性，可以加入小幅随机扰动
            angle = math.atan2(target_dir[1], target_dir[0])
            angle += random.uniform(-0.2, 0.2)  # 小幅扰动
            new_dir = (math.cos(angle), math.sin(angle))
            return new_dir

    # ----------------- 移动控制 -----------------
    def _move_towards_direction(self, desired_dir: Tuple[float, float]):
        """
        根据期望方向控制转向和前进
        计算当前朝向与期望方向的夹角，控制左右转向，始终前进
        """
        # 计算期望方向对应的角度（车身朝向，0°向上）
        desired_angle = math.degrees(math.atan2(desired_dir[1], desired_dir[0])) + 90
        desired_angle %= 360

        # 角度差
        angle_diff = self.unit.get_angle_difference(self.unit.direction_angle, desired_angle)

        # 转向控制（带死区）
        turn_threshold = 5
        if abs(angle_diff) > turn_threshold:
            if angle_diff > 0:
                self.unit.set_turning(left=False, right=True)   # 顺时针
            else:
                self.unit.set_turning(left=True, right=False)   # 逆时针
        else:
            self.unit.set_turning(False, False)

        # 始终前进（除非特殊情况，如碰撞脱离可能需要后退，但这里简化）
        self.unit.set_movement(forward=True, backward=False)

        # 更新上一次方向
        self.last_desired_dir = desired_dir

    # ----------------- 攻击决策 -----------------
    def _try_fire(self, target: BaseUnit):
        """尝试开火，检查视线和冷却"""
        # 冷却检查（使用单位自身的 fire_cooldown，但 AI 维护自己的冷却更准确？这里使用单位冷却）
        if self.unit.fire_cooldown > 0:
            return

        if not self._can_see_target(target):
            return

        # 炮塔指向误差检查
        angle_diff = self.unit.get_angle_difference(
            self.unit.turret_direction_angle, self.unit.turret_target_angle
        )
        if abs(angle_diff) < 10:
            bullet = self.unit.fire()
            if bullet:
                self.bullet_manager.add_bullet(bullet)

    def _can_see_target(self, target: BaseUnit) -> bool:
        """检查从单位到目标是否有障碍物阻挡（仅实心障碍物）"""
        start = self.unit.position
        end = target.position
        for obs in self.game_map.bullet_obstacles:
            if obs.clipline(start, end):
                return False
        return True

    # ----------------- 辅助方法 -----------------
    def _get_current_ammo_range(self) -> float:
        """获取当前弹药的最大射程"""
        ammo_type = self.unit.current_ammunition
        bullet_class = None
        if ammo_type == "normal_shell":
            bullet_class = NormalShell
        elif ammo_type == "rocket_shell":
            bullet_class = RocketShell
        elif ammo_type == "heavy_shell":
            bullet_class = HeavyShell
        else:
            return 300.0  # 默认值

        if bullet_class:
            # 临时实例化以获取属性
            temp_bullet = bullet_class(
                projectile_id="temp",
                shooter=self.unit,
                shooter_team=self.unit.team
            )
            return temp_bullet.max_lifetime * temp_bullet.speed
        return 300.0