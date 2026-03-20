"""
    敌人 AI 控制模块（智能版）
    实现：
    1. 碰撞脱离：与障碍物重叠时立即寻找可脱离方向
    2. 主动避障：预测未来路径，避免撞上障碍物
    3. 子弹规避：预测子弹威胁，主动躲避
    4. 朝向目标移动：在所有行动中尽量靠近目标
    优先级：碰撞脱离 > 主动避障 > 子弹规避 > 默认朝向目标
"""

import pygame
import math
import random
from typing import List, Optional, Tuple, Dict, Any
from Parameter import *
from Unit.BaseUnit import BaseUnit


class EnemyAI:
    # 内部参数（不依赖外部 Parameter）
    # 碰撞脱离参数
    ESCAPE_STEP_DISTANCE = 5.0          # 检测脱离时尝试移动的距离（像素）
    ESCAPE_ANGLE_STEP = 15.0             # 检测脱离时尝试的旋转角度步长（度）
    
    # 主动避障参数
    OBSTACLE_LOOKAHEAD_TIME = 1.0        # 预测前方碰撞的时间（秒）
    OBSTACLE_CHECK_STEPS = 5              # 预测时间内采样点数
    SAFE_MARGIN = 2.0                     # 安全范围额外增加的像素（超出单位碰撞箱对角线一半）
    AVOID_ANGLE_RANGE = 60.0               # 主动避障时考虑的转向角度范围（度）
    AVOID_ANGLE_STEP = 15.0                # 角度采样步长
    
    # 子弹规避参数
    BULLET_LOOKAHEAD_TIME = 2.0            # 子弹预测时间（秒）
    BULLET_THREAT_DISTANCE = 150.0          # 子弹威胁检测距离（像素）
    BULLET_MIN_TIME_TO_IMPACT = 0.5         # 最小撞击时间，小于此值才考虑规避
    BULLET_EVADE_ANGLE_STEP = 30.0          # 子弹规避方向采样步长
    
    # 目标朝向权重
    TARGET_DIRECTION_WEIGHT = 0.7           # 方向选择时目标方向的权重
    SAFETY_WEIGHT = 0.3                      # 安全性的权重（避障时）
    
    def __init__(self, unit: BaseUnit, unit_manager, bullet_manager, game_map):
        """
        :param unit: 受此 AI 控制的单位
        :param unit_manager: 单位管理器
        :param bullet_manager: 子弹管理器
        :param game_map: 游戏地图（需提供 unit_obstacles 和 bullet_obstacles）
        """
        self.unit = unit
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        self.game_map = game_map

        # 状态标记（仅用于调试）
        self.ai_state = "idle"

        # 缓存单位碰撞箱对角线一半（用于安全范围计算）
        self._update_unit_radius()
        
        # 开火冷却（与外部一致）
        self.fire_cooldown = 0.0
        self.fire_cooldown_max = 0.5

    def _update_unit_radius(self):
        """计算单位碰撞箱对角线的一半（即中心到最远角的距离）"""
        w, h = self.unit.size
        self.unit_radius = math.hypot(w, h) / 2.0
        self.safe_radius = self.unit_radius + self.SAFE_MARGIN

    # ----------------- 公开更新接口 -----------------
    def update(self, delta_time: float, unit_manager, bullet_manager, game_map):
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        self.game_map = game_map
        self._update_unit_radius()  # 单位大小可能变化，重新计算

        # 更新开火冷却
        if self.fire_cooldown > 0:
            self.fire_cooldown -= delta_time

        # 获取目标（最近的敌方单位）
        enemy_units = self._get_enemy_units()
        if not enemy_units:
            self.ai_state = "idle"
            self.unit.set_movement(False, False)
            self.unit.set_turning(False, False)
            return
        target = self._get_closest_unit(enemy_units)
        if target is None:
            self.ai_state = "idle"
            self.unit.set_movement(False, False)
            self.unit.set_turning(False, False)
            return
            
        self.target_unit = target

        # 获取目标方向向量
        target_dir = self._vector_to_target(target)

        # 决策步骤 1：检查是否已经与障碍物碰撞
        if self._is_colliding():
            best_dir = self._get_collision_escape_direction(target_dir)
            if best_dir is not None:
                self.ai_state = "collision_escape"
                self._move_towards_direction(best_dir)
                # 碰撞脱离后仍然尝试瞄准和开火
                self._aim_at_target(target)
                self._try_fire(target)
                return

        # 决策步骤 2：主动避障（预测未来碰撞）
        avoid_dir = self._get_obstacle_avoidance_direction(target_dir)
        if avoid_dir is not None:
            self.ai_state = "avoid_obstacle"
            self._move_towards_direction(avoid_dir)
            self._aim_at_target(target)
            self._try_fire(target)
            return

        # 决策步骤 3：子弹规避
        evade_dir = self._get_bullet_evasion_direction(target_dir)
        if evade_dir is not None:
            self.ai_state = "evade_bullet"
            self._move_towards_direction(evade_dir)
            self._aim_at_target(target)
            self._try_fire(target)
            return

        # 默认行为：直接朝向目标移动
        self.ai_state = "chase"
        self._move_towards_direction(target_dir)
        self._aim_at_target(target)
        self._try_fire(target)

    # ----------------- 辅助方法 -----------------
    def _get_enemy_units(self) -> List[BaseUnit]:
        return [u for u in self.unit_manager.units if u.is_alive and u.visible and u.team != self.unit.team]

    def _get_closest_unit(self, units: List[BaseUnit]) -> Optional[BaseUnit]:
        if not units:
            return None
        closest = None
        min_dist_sq = float('inf')
        for u in units:
            dx = u.position[0] - self.unit.position[0]
            dy = u.position[1] - self.unit.position[1]
            dist_sq = dx*dx + dy*dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest = u
        return closest

    def _vector_to_target(self, target: BaseUnit) -> Tuple[float, float]:
        dx = target.position[0] - self.unit.position[0]
        dy = target.position[1] - self.unit.position[1]
        dist = math.hypot(dx, dy)
        if dist > 0:
            return (dx / dist, dy / dist)
        return (1.0, 0.0)

    def _is_colliding(self) -> bool:
        """检测单位当前是否与任何单位障碍物碰撞"""
        if not self.unit.bounding_box:
            return False
        for obs in self.game_map.unit_obstacles:
            if self.unit.bounding_box.colliderect(obs):
                return True
        return False

    def _is_position_safe(self, pos: Tuple[float, float]) -> bool:
        """
        检查给定位置是否安全（即单位放置在该位置时，其安全范围矩形不与任何障碍物碰撞）
        安全范围 = 中心点 + safe_radius 构成的矩形（即单位碰撞箱外扩 SAFE_MARGIN）
        """
        x, y = pos
        # 构建安全范围矩形（以中心点，宽高为 unit.size + 2*SAFE_MARGIN）
        safe_rect = pygame.Rect(
            x - self.safe_radius,
            y - self.safe_radius,
            self.safe_radius * 2,
            self.safe_radius * 2
        )
        for obs in self.game_map.unit_obstacles:
            if safe_rect.colliderect(obs):
                return False
        return True

    def _is_future_position_safe(self, direction: Tuple[float, float], duration: float) -> bool:
        """
        预测沿 direction 移动 duration 秒后的位置是否安全
        （简单线性预测，不考虑转向）
        """
        # 预计移动距离
        move_dist = self.unit.max_speed * duration
        dx = direction[0] * move_dist
        dy = direction[1] * move_dist
        future_pos = (self.unit.position[0] + dx, self.unit.position[1] + dy)
        return self._is_position_safe(future_pos)

    # ----------------- 碰撞脱离 -----------------
    def _get_collision_escape_direction(self, target_dir: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        当单位已经与障碍物碰撞时，寻找一个可脱离的方向。
        尝试以下方向：
        - 旋转车身，然后向前移动一小步
        - 直接后退
        返回最佳方向向量（单位向量）
        """
        # 候选方向列表：当前方向、左前、右前、左、右、后
        current_angle = self.unit.direction_angle
        candidate_angles = []
        # 生成一系列角度（步长 ESCAPE_ANGLE_STEP，覆盖360度）
        for offset in range(-180, 181, int(self.ESCAPE_ANGLE_STEP)):
            candidate_angles.append((current_angle + offset) % 360)

        best_score = -float('inf')
        best_dir = None

        # 目标方向角度（用于评分）
        target_angle = math.degrees(math.atan2(target_dir[1], target_dir[0])) + 90
        target_angle %= 360

        for angle in candidate_angles:
            # 计算该角度对应的方向向量（单位前进方向）
            rad = math.radians(angle - 90)
            dir_vec = (math.cos(rad), math.sin(rad))

            # 尝试向前移动一小步
            test_pos = (
                self.unit.position[0] + dir_vec[0] * self.ESCAPE_STEP_DISTANCE,
                self.unit.position[1] + dir_vec[1] * self.ESCAPE_STEP_DISTANCE
            )
            if self._is_position_safe(test_pos):
                # 这个方向可行，计算评分（越接近目标方向分越高）
                # 角度差
                angle_diff = abs((angle - target_angle + 180) % 360 - 180)
                score = 180 - angle_diff  # 越大越好
                if score > best_score:
                    best_score = score
                    best_dir = dir_vec

        # 如果向前都不行，尝试后退
        if best_dir is None:
            # 后退方向（当前角度+180）
            back_angle = (current_angle + 180) % 360
            rad = math.radians(back_angle - 90)
            back_dir = (math.cos(rad), math.sin(rad))
            test_pos = (
                self.unit.position[0] + back_dir[0] * self.ESCAPE_STEP_DISTANCE,
                self.unit.position[1] + back_dir[1] * self.ESCAPE_STEP_DISTANCE
            )
            if self._is_position_safe(test_pos):
                best_dir = back_dir

        return best_dir

    # ----------------- 主动避障 -----------------
    def _get_obstacle_avoidance_direction(self, target_dir: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        预测未来是否会与障碍物碰撞，返回一个能够避开障碍且尽量靠近目标的方向。
        如果没有碰撞风险，返回 None。
        """
        # 如果当前已经碰撞，由碰撞脱离处理，这里不处理
        if self._is_colliding():
            return None

        current_angle = self.unit.direction_angle
        target_angle = math.degrees(math.atan2(target_dir[1], target_dir[0])) + 90
        target_angle %= 360

        # 生成候选角度（围绕当前方向 ± AVOID_ANGLE_RANGE）
        start_angle = current_angle - self.AVOID_ANGLE_RANGE
        end_angle = current_angle + self.AVOID_ANGLE_RANGE
        candidate_angles = []
        angle = start_angle
        while angle <= end_angle + 1e-6:
            candidate_angles.append(angle % 360)
            angle += self.AVOID_ANGLE_STEP

        # 如果没有候选，直接返回None
        if not candidate_angles:
            return None

        # 评估每个方向：安全性（预测时间内是否安全） + 朝向目标得分
        best_score = -float('inf')
        best_dir = None

        for angle in candidate_angles:
            rad = math.radians(angle - 90)
            dir_vec = (math.cos(rad), math.sin(rad))

            # 预测未来 OBSTACLE_LOOKAHEAD_TIME 内是否安全
            safe = True
            # 采样多个时间点
            for t in range(1, self.OBSTACLE_CHECK_STEPS + 1):
                dt = t * self.OBSTACLE_LOOKAHEAD_TIME / self.OBSTACLE_CHECK_STEPS
                if not self._is_future_position_safe(dir_vec, dt):
                    safe = False
                    break
            if not safe:
                continue

            # 计算朝向目标得分
            angle_diff = abs((angle - target_angle + 180) % 360 - 180)
            target_score = 180 - angle_diff  # 范围0~180

            # 计算安全性得分（这里安全即为满分，不安全已排除）
            safety_score = 180  # 最大值

            # 综合得分
            score = self.TARGET_DIRECTION_WEIGHT * target_score + self.SAFETY_WEIGHT * safety_score

            if score > best_score:
                best_score = score
                best_dir = dir_vec

        return best_dir

    # ----------------- 子弹规避 -----------------
    def _get_bullet_evasion_direction(self, target_dir: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        检测子弹威胁，返回规避方向。
        若无威胁或无法有效规避，返回 None。
        """
        # 收集有威胁的子弹
        threats = self._detect_threatening_bullets()
        if not threats:
            return None

        # 对每个威胁，计算规避方向候选
        # 规避方向：垂直于子弹速度方向（左右两个方向），选取远离子弹的方向
        # 然后评估哪个方向能真正避免碰撞（预测移动后是否仍会被击中）
        best_evade_dir = None
        best_score = -float('inf')
        target_angle = math.degrees(math.atan2(target_dir[1], target_dir[0])) + 90

        for bullet in threats:
            # 子弹速度方向
            vx, vy = bullet.velocity
            v_len = math.hypot(vx, vy)
            if v_len == 0:
                continue
            v_norm = (vx / v_len, vy / v_len)
            # 垂直方向（左、右）
            left_perp = (-v_norm[1], v_norm[0])
            right_perp = (v_norm[1], -v_norm[0])

            # 确定哪个方向是远离子弹的
            # 计算子弹指向自己的向量
            to_bullet = (bullet.position[0] - self.unit.position[0],
                         bullet.position[1] - self.unit.position[1])
            # 点积：左垂直方向与 to_bullet 的夹角判断
            dot_left = left_perp[0]*to_bullet[0] + left_perp[1]*to_bullet[1]
            dot_right = right_perp[0]*to_bullet[0] + right_perp[1]*to_bullet[1]
            # 选择使点积为正的方向（远离子弹的方向）
            evade_dir = left_perp if dot_left > 0 else right_perp

            # 现在预测如果我们沿 evade_dir 移动，是否仍然会被击中
            if self._will_be_hit_if_move(evade_dir, bullet):
                # 仍然会被击中，放弃这个子弹的规避方向
                continue

            # 计算该方向的目标朝向得分
            evade_angle = math.degrees(math.atan2(evade_dir[1], evade_dir[0])) + 90
            evade_angle %= 360
            angle_diff = abs((evade_angle - target_angle + 180) % 360 - 180)
            target_score = 180 - angle_diff

            # 安全性得分（这里简单给满分）
            safety_score = 180

            score = self.TARGET_DIRECTION_WEIGHT * target_score + self.SAFETY_WEIGHT * safety_score

            if score > best_score:
                best_score = score
                best_evade_dir = evade_dir

        return best_evade_dir

    def _detect_threatening_bullets(self) -> List:
        """
        检测所有可能威胁到自己的子弹，返回子弹列表
        判断标准：
        - 子弹距离小于 BULLET_THREAT_DISTANCE
        - 子弹朝向自己（相对速度点积为负）
        - 预测撞击时间小于 BULLET_MIN_TIME_TO_IMPACT
        """
        threats = []
        my_pos = self.unit.position
        my_radius = self.unit_radius  # 单位近似半径

        for bullet in self.bullet_manager.bullets:
            if bullet.shooter_team == self.unit.team:
                continue  # 忽略友军子弹
            if not bullet.is_active:
                continue

            # 计算相对位置和速度
            dx = my_pos[0] - bullet.position[0]
            dy = my_pos[1] - bullet.position[1]
            dist = math.hypot(dx, dy)
            if dist > self.BULLET_THREAT_DISTANCE:
                continue

            vx, vy = bullet.velocity
            # 相对速度（单位静止，子弹相对速度即为 -v）
            # 判断子弹是否朝向自己：dx*vx + dy*vy > 0 表示子弹靠近
            if dx * vx + dy * vy <= 0:
                continue  # 正在远离或相切

            # 求解二次方程预测撞击时间（假设单位静止）
            a = vx*vx + vy*vy
            if a == 0:
                continue
            b = 2 * (dx * vx + dy * vy)
            c = dx*dx + dy*dy - my_radius*my_radius
            discriminant = b*b - 4*a*c
            if discriminant < 0:
                continue
            t1 = (-b - math.sqrt(discriminant)) / (2*a)
            t2 = (-b + math.sqrt(discriminant)) / (2*a)
            # 收集正根
            positive_times = [t for t in (t1, t2) if t > 0]
            if not positive_times:
                continue
            t_impact = min(positive_times)
            if t_impact <= self.BULLET_MIN_TIME_TO_IMPACT:
                threats.append(bullet)
        return threats

    def _will_be_hit_if_move(self, move_dir: Tuple[float, float], bullet) -> bool:
        """
        预测如果单位沿 move_dir 移动，是否仍会被指定子弹击中
        简化：假设子弹和单位都做匀速直线运动，求解最短距离时间
        """
        # 单位初始位置
        u0x, u0y = self.unit.position
        # 子弹初始位置
        b0x, b0y = bullet.position
        # 单位速度
        u_speed = self.unit.max_speed
        u_vx = move_dir[0] * u_speed
        u_vy = move_dir[1] * u_speed
        # 子弹速度
        b_vx, b_vy = bullet.velocity

        # 相对速度
        rel_vx = b_vx - u_vx
        rel_vy = b_vy - u_vy
        # 初始相对位置
        rel_x = b0x - u0x
        rel_y = b0y - u0y

        a = rel_vx*rel_vx + rel_vy*rel_vy
        if a == 0:
            # 相对静止，检查初始距离
            return math.hypot(rel_x, rel_y) <= self.unit_radius
        b = 2 * (rel_x*rel_vx + rel_y*rel_vy)
        c = rel_x*rel_x + rel_y*rel_y - self.unit_radius*self.unit_radius

        discriminant = b*b - 4*a*c
        if discriminant < 0:
            return False
        t1 = (-b - math.sqrt(discriminant)) / (2*a)
        t2 = (-b + math.sqrt(discriminant)) / (2*a)
        # 存在正的时间使得距离小于半径，且在预测时间内
        for t in (t1, t2):
            if t > 0 and t <= self.BULLET_LOOKAHEAD_TIME:
                return True
        return False

    # ----------------- 移动控制 -----------------
    def _move_towards_direction(self, desired_dir: Tuple[float, float]):
        """
        根据期望方向控制车身转向和前进
        """
        desired_angle = math.degrees(math.atan2(desired_dir[1], desired_dir[0])) + 90
        desired_angle %= 360
        angle_diff = self.unit.get_angle_difference(self.unit.direction_angle, desired_angle)

        turn_threshold = 5.0
        if abs(angle_diff) > turn_threshold:
            if angle_diff > 0:
                self.unit.set_turning(left=False, right=True)   # 顺时针
            else:
                self.unit.set_turning(left=True, right=False)   # 逆时针
        else:
            self.unit.set_turning(False, False)

        # 前进/后退：始终前进（如果角度差太大，可能应该减速，但这里简化）
        self.unit.set_movement(forward=True, backward=False)

    def _aim_at_target(self, target: BaseUnit):
        dx = target.position[0] - self.unit.position[0]
        dy = target.position[1] - self.unit.position[1]
        target_angle = math.degrees(math.atan2(dy, dx)) + 90
        target_angle %= 360
        self.unit.turret_target_angle = target_angle

    def _try_fire(self, target: BaseUnit):
        """尝试开火，检查视线和冷却"""
        if self.fire_cooldown <= 0 and self._can_see_target(target):
            angle_diff = self.unit.get_angle_difference(
                self.unit.turret_direction_angle, self.unit.turret_target_angle
            )
            if abs(angle_diff) < 10:
                bullet = self.unit.fire()
                if bullet:
                    self.bullet_manager.add_bullet(bullet)
                    self.fire_cooldown = self.fire_cooldown_max

    def _can_see_target(self, target: BaseUnit) -> bool:
        """视线检测：使用 bullet_obstacles（只被实心障碍阻挡，水不阻挡）"""
        start = self.unit.position
        end = target.position
        for obs in self.game_map.bullet_obstacles:
            if obs.clipline(start, end):
                return False
        return True