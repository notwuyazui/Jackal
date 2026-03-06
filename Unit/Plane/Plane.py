'''
    一个飞机类
    机动性为0，生命值较高，装备重炮弹
'''

import pygame
import math
from Unit.BaseUnit import BaseUnit
from Parameter import *

class Plane(BaseUnit):
    
    def __init__(self, unit_id: int, unit_team: Team, usingAI = False):
        
        self.unit_type = 'plane'
        self.body_image_path = 'Unit/Plane/plane.png'
        self.turret_image_path = 'Unit/Plane/turret.png'
        if unit_team == Team.ENEMY:
            self.body_image_path = 'Unit/Plane/enemy_plane.png'
            self.turret_image_path = 'Unit/Plane/enemy_turret.png'
        self.max_speed_rate = 0.0
        self.max_acceleration_rate = 0.0
        self.min_acceleration_rate = 0.0
        self.max_angular_speed_rate = 0.0
        self.turret_angular_speed_rate = 1.0
        self.max_health_rate = 5.0
        self.armor_type = ArmorType.NONE                                                        # 护甲类型
        self.ammunition_types = ['heavy_shell']                                                      # 单位拥有弹种
        self.ammo_switch_time =  UNIT_AMMO_SWITCH_TIME                                          # 单位切换弹种时间
        
        
        super().__init__(unit_id, unit_team, usingAI,
                         unit_type=self.unit_type,
                         body_image_path=self.body_image_path, 
                         turret_image_path=self.turret_image_path, 
                         max_speed_rate=self.max_speed_rate, 
                         max_acceleration_rate=self.max_acceleration_rate, 
                         min_acceleration_rate=self.min_acceleration_rate, 
                         max_angular_speed_rate=self.max_angular_speed_rate, 
                         turret_angular_speed_rate=self.turret_angular_speed_rate, 
                         max_health_rate=self.max_health_rate, 
                         armor_type=self.armor_type, 
                         ammunition_types=self.ammunition_types, 
                         ammo_switch_time=self.ammo_switch_time)

def create_plane(unit_id, unit_team=Team.PLAYER, position=(0, 0)):
    # 示例：plane = create_plane(1, Team.PLAYER, position=(400, 300))
    plane = Plane(unit_id, unit_team)
    plane.position = position
    return plane            

def create_enemy_plane(unit_id, position=(0, 0), usingAI = False):
    enemy = Plane(unit_id, Team.ENEMY, usingAI)
    enemy.position = position
    return enemy

def create_player_plane(unit_id, position=(0, 0), usingAI = False):
    player = Plane(unit_id, Team.PLAYER, usingAI)
    player.position = position
    return player