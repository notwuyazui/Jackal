"""
    单位管理器，负责管理所有单位的更新和绘制，并自动管理 AI 单位
"""
from Unit.Tank.Tank import *
from Map.Map import *
from GameMode import *

class UnitManager:
    def __init__(self):
        self.units = []               # 所有单位列表
        self.enemy_ais = []            # AI 控制器列表
        self.to_remove = []            # 待移除单位暂存

    def add_unit(self, unit, bullet_manager, obstacles):
        """添加单位，若为 AI 单位则自动创建对应的 EnemyAI"""
        from Unit.EnemyAI import EnemyAI
        if unit:
            self.units.append(unit)
            if unit.usingAI and bullet_manager is not None and obstacles is not None:
                ai = EnemyAI(unit, self, bullet_manager, obstacles)
                self.enemy_ais.append(ai)

    def update(self, delta_time, unit_manager, bullet_manager, game_map):
        obstacles = game_map.obstacles
        # 1. 更新所有 AI 决策（设置运动参数）
        for ai in self.enemy_ais:
            ai.update(delta_time, unit_manager, bullet_manager, obstacles)

        # 2. 更新所有单位（包括 AI 单位，执行物理移动、碰撞等）
        for unit in self.units:
            unit.update(delta_time, unit_manager, bullet_manager, game_map)

        # 3. 移除死亡单位及其关联的 AI
        self.to_remove.clear()
        for unit in self.units:
            if not unit.is_alive:
                self.to_remove.append(unit)

        for unit in self.to_remove:
            # 同时移除对应的 AI
            self.enemy_ais = [ai for ai in self.enemy_ais if ai.unit != unit]

    def draw(self, surface, camera_offset):
        for unit in self.units:
            unit.draw(surface, camera_offset)
            
    def get_unit_by_id(self, unit_id):
        for unit in self.units:
            if unit.id == unit_id:
                return unit
        return None

    def get_active_count(self):
        return len([u for u in self.units if u.is_alive])

    def is_in(self, unit_id):
        return any(unit.id == unit_id for unit in self.units)
    
    def clear(self):
        self.units.clear()
        self.enemy_ais.clear()
        
    def save(self):
        return [unit.save() for unit in self.units]