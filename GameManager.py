'''
    用于游戏实体的总体控制，包括子弹、单位、地图的更新和绘制
'''

from Map.Map import *
from Unit.UnitManager import *
from Bullet.BulletManager import *
from Unit.Tank.Tank import *

class GameManager:
    def __init__ (self, game_map:GameMap = create_empty_map(), unit_manager = UnitManager(), bullet_manager = BulletManager()):
        self.game_map = game_map
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        
        self.add_player_tank(position=(100,500), unit_id=0, usingAI=False)      # 默认添加一个玩家坦克

    def update(self, delta_time):
        self.unit_manager.update(delta_time, self.unit_manager, self.bullet_manager, self.game_map.obstacles)
        self.bullet_manager.update(delta_time, self.unit_manager.units, self.game_map.obstacles)

    def draw(self, screen, camera_offset):
        self.game_map.draw(screen, camera_offset)
        self.unit_manager.draw(screen, camera_offset)
        self.bullet_manager.draw(screen, camera_offset)
        
    def set_game_map(self, game_map:GameMap):
        self.game_map = game_map
        
    def set_border_map(self):
        self.game_map = create_border_map()
    
    def set_empty_map(self):
        self.game_map = create_empty_map()
    
    def set_random_map(self):
        self.game_map = create_random_map()
        
    def set_test_map(self):
        self.game_map = create_test_map()
    
    def set_unit_manager(self, unit_manager:UnitManager):
        self.unit_manager = unit_manager
    
    def set_bullet_manager(self, bullet_manager:BulletManager):
        self.bullet_manager = bullet_manager
        
    def add_unit(self, unit):
        self.unit_manager.add_unit(unit, self.bullet_manager, self.game_map.obstacles)
        
    def add_player_tank(self, position=(0,0), unit_id = None, usingAI = False):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        tank = create_player_tank(unit_id, position, usingAI)
        self.add_unit(tank)
        return tank
        
    def add_enemy_tank(self, position=(0,0), unit_id = None, usingAI = False):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        tank = create_enemy_tank(unit_id, position, usingAI)
        self.add_unit(tank)
        return tank
    
    def add_bullet(self, bullet):
        self.bullet_manager.add_bullet(bullet)
    
    def get_active_units_counts(self):
        return self.unit_manager.get_active_count()

    def get_active_bullets_counts(self):
        return self.bullet_manager.get_active_count()
    
    def clear_bullets(self):
        self.bullet_manager.clear()
        
    def clear_units(self):
        self.unit_manager.clear()
    
    def save_map(self, file_name = None):
        self.game_map.save(file_name)
    
    def save_unit(self, file_name = None):
        self.unit_manager.save(file_name)
        
    def save_bullet(self, file_name = None):
        self.bullet_manager.save(file_name)
    
    def save(self):
        self.save_map()
        self.save_unit()
        self.save_bullet()
