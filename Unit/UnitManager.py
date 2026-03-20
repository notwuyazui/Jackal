"""
    单位管理器，负责管理所有单位的更新和绘制，并自动管理 AI 单位
"""
from Unit.Tank.Tank import *
from Map.GameMap import *
from GameMode import *

class UnitManager:
    def __init__(self):
        self.units = []               # 所有单位列表
        self.enemy_ais = []            # AI 控制器列表
        self.to_remove = []            # 待移除单位暂存

    def add_unit(self, unit, bullet_manager, game_map):
        """添加单位，若为 AI 单位则自动创建对应的 EnemyAI"""
        from Unit.EnemyAI import EnemyAI
        if unit:
            self.units.append(unit)
            if unit.usingAI and bullet_manager is not None and game_map is not None:
                ai = EnemyAI(unit, self, bullet_manager, game_map)
                self.enemy_ais.append(ai)

    def update(self, delta_time, unit_manager, bullet_manager, game_map):
        # 更新所有AI决策
        for ai in self.enemy_ais:
            ai.update(delta_time, unit_manager, bullet_manager, game_map)

        # 更新所有单位
        for unit in self.units:
            unit.update(delta_time, unit_manager, bullet_manager, game_map)
            
        # 自动通信
        if AUTO_COMMUNICATE:
            self.auto_communicate()

        self.to_remove.clear()
        for unit in self.units:
            if not unit.is_alive:
                self.to_remove.append(unit)

        for unit in self.to_remove:
            # 移除对应的 AI
            self.enemy_ais = [ai for ai in self.enemy_ais if ai.unit != unit]

    def auto_communicate(self):
        """
        自动通信：反复让所有存活单位广播自己的可见信息，直到一轮中没有任何单位获得新信息为止。
        """
        changed = True
        while changed:
            changed = False
            # 收集当前所有存活单位（避免在迭代过程中列表变化）
            alive_units = [u for u in self.units if u.is_alive]
            for unit in alive_units:
                _, _, _, any_added = unit.broadcast(self)
                if any_added:
                    changed = True

    def draw(self, surface, camera_offset, mouse_pos = None):
        for unit in self.units:
            unit.draw(surface, camera_offset, mouse_pos)
            
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