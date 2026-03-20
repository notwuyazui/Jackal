'''
    用于游戏实体的总体控制，包括子弹、单位、地图的更新和绘制
'''

from Map.GameMap import *
from Unit.UnitManager import *
from Bullet.BulletManager import *
from Unit.Tank.Tank import *
from Unit.Archie.Archie import *
from Unit.Plane.Plane import *
from GameMode import *

class GameManager:
    def __init__ (self, game_map:GameMap = create_empty_map(), unit_manager = UnitManager(), bullet_manager = BulletManager()):
        self.game_map = game_map
        self.unit_manager = unit_manager
        self.bullet_manager = bullet_manager
        
        self.camera_offset = [0.0, 0.0]
        self.camera_speed = 5
        
        self.time = 0.0        # 游戏时间
        self.print_record_timer = 0.0

    def update(self, delta_time):
        self.time += delta_time
        self.print_record_timer += delta_time
        self.game_map.update(delta_time) if self.game_map != None else None
        self.unit_manager.update(delta_time, self.unit_manager, self.bullet_manager, self.game_map)
        self.bullet_manager.update(delta_time, self.unit_manager, self.game_map)

    def draw(self, screen, camera_offset = None):
        if camera_offset == None:
            camera_offset = self.camera_offset
        if camera_offset == None:
            return
        self.game_map.draw(screen, camera_offset) if self.game_map != None else None
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
        
    def set_camera_offset_move(self, direction):
        if direction == Direction.UP:
            self.camera_offset[1] -= self.camera_speed
        elif direction == Direction.DOWN:
            self.camera_offset[1] += self.camera_speed
        elif direction == Direction.LEFT:
            self.camera_offset[0] -= self.camera_speed
        elif direction == Direction.RIGHT:
            self.camera_offset[0] += self.camera_speed
        
    def set_unit_movement(self, unit_id, forward=False, backward=False):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        return unit.set_movement(forward, backward)
    
    def set_unit_turning(self, unit_id, left=False, right=False):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        return unit.set_turning(left, right)
    
    def set_unit_turret_target_to_mouse(self, unit_id, mouse_pos, camera_offset = None):
        if camera_offset is None:
            camera_offset = self.camera_offset
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        return unit.set_turret_target_to_mouse(mouse_pos, camera_offset)
    
    def set_unit_fire(self, unit_id):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return None
        bullet = unit.fire()
        self.add_bullet(bullet)
        return bullet
        
    def set_unit_switch_ammo(self, unit_id):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        return unit.switch_ammunition()
        
    def set_unit_action(self, unit_id, action: Action):
        # 控制单位行为的七个基本动作：
        # 前进，后退，左转，右转，鼠标瞄准，开火，切换弹药
        self.set_unit_movement(unit_id, action.forward, action.backward)
        self.set_unit_turning(unit_id, action.left, action.right)
        self.set_unit_turret_target_to_mouse(unit_id, action.mouse_pos, self.camera_offset)
        if action.fire: self.set_unit_fire(unit_id)
        if action.switch_ammo: self.set_unit_switch_ammo(unit_id)
        
    def set_unit_communicate_to(self, unit_id, target_unit_id):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        unit.set_communicate_to(target_unit_id)
        
    def set_unit_broadcast(self, unit_id):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        unit.broadcast()
    
    def set_unit_receive_from(self, unit_id, source_unit_id):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        unit.receive_from(source_unit_id)
        
    def set_unit_broadcast_receive(self, unit_id):
        unit = self.unit_manager.get_unit_by_id(unit_id)
        if unit is None:
            return False
        unit.broadcast_receive()
    
    def add_unit(self, unit):
        self.unit_manager.add_unit(unit, self.bullet_manager, self.game_map)
        
    def add_player_tank(self, position=(0,0), unit_id = None, usingAI = False, visible = True):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        tank = create_player_tank(unit_id, position, usingAI, visible)
        self.add_unit(tank)
        return tank
        
    def add_enemy_tank(self, position=(0,0), unit_id = None, usingAI = False, visible = True):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        tank = create_enemy_tank(unit_id, position, usingAI, visible)
        self.add_unit(tank)
        return tank
    
    def add_player_archie(self, position=(0,0), unit_id = None, usingAI = False, visible = True):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        archie = create_player_archie(unit_id, position, usingAI, visible)
        self.add_unit(archie)
        return archie
    
    def add_enemy_archie(self, position=(0,0), unit_id = None, usingAI = False, visible = True):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        archie = create_enemy_archie(unit_id, position, usingAI, visible)
        self.add_unit(archie)
        return archie
    
    def add_player_plane(self, position=(0,0), unit_id = None, usingAI = False, visible = True):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        plane = create_player_plane(unit_id, position, usingAI, visible)
        self.add_unit(plane)
        return plane
    
    def add_enemy_plane(self, position=(0,0), unit_id = None, usingAI = False, visible = True):
        if unit_id is None:
            unit_id = len(self.unit_manager.units)
        plane = create_enemy_plane(unit_id, position, usingAI, visible)
        self.add_unit(plane)
        return plane
    
    def add_bullet(self, bullet):
        self.bullet_manager.add_bullet(bullet)
        
    def get_unit(self, unit_id):
        # 通过id定位unit
        return self.unit_manager.get_unit_by_id(unit_id)
    
    def get_active_units_counts(self):
        return self.unit_manager.get_active_count()

    def get_active_bullets_counts(self):
        return self.bullet_manager.get_active_count()
    
    def print_record(self):
        if self.print_record_timer > 5:
            self.print_record_timer = 0
            print("time: " + str(int(self.time)))
            if UNIT_RECORD_TEXT or DEBUG_MODE:
                for unit in self.unit_manager.units:
                    if unit is not None:
                        print(unit.get_record())
            if PRINT_VISIBLE_UNIT or DEBUG_MODE:
                unit0 = self.unit_manager.get_unit_by_id(0)
                if unit0 is not None:
                    ids = unit0.get_visible_unit_ids()
                    print(f"Unit 0 可见的单位ID: {ids}")
            
    def draw_mouse_target(self, unit_id, surface, mouse_pos):
        if self.unit_manager.is_in(unit_id):
            unit = self.unit_manager.get_unit_by_id(unit_id)
            if unit is not None:
                unit._draw_mouse_target_line(surface, self.camera_offset, mouse_pos)
                        
    def clear_bullets(self):
        self.bullet_manager.clear()
        
    def clear_units(self):
        self.unit_manager.clear()
    
    def save_map(self):
        if self.game_map is not None:
            self.game_map.save()
    
    def save_unit(self):
        self.unit_manager.save()
        
    def save_bullet(self):
        self.bullet_manager.save()
    
    def save(self):
        self.save_map()
        self.save_unit()
        self.save_bullet()
