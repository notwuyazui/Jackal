"""单位管理器，负责管理所有单位的更新和绘制"""
from Unit.Tank.Tank import *
from Map.Map import *
from GameMode import *
from Bullet.NormalShell.NormalShell import *
from Unit.EnemyAI import *

class UnitManager:

    def __init__(self):
        self.units = []
        self.to_remove = []
        
    def add_unit(self, unit):
        if unit:
            self.units.append(unit)
    
    def update(self, delta_time, obstacles):
        self.to_remove.clear()
        
        for unit in self.units:
            # 更新单位状态
            if not unit.update(delta_time, obstacles):
                # 单位死亡，标记为待移除
                if unit in self.units and unit not in self.to_remove:
                    self.to_remove.append(unit)
        
        # 移除死亡单位
        for unit in self.to_remove:
            if unit in self.units:
                self.units.remove(unit)
    
    def draw(self, surface, camera_offset):
        for unit in self.units:
            unit.draw(surface, camera_offset)
    
    def get_active_count(self):
        return len([b for b in self.bullets if b.is_active])
    
    def clear(self):
        self.units.clear()

