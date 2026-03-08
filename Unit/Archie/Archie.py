'''
    一个火箭炮类
    较低机动，较高生命，较低侦察范围，仅装备火箭弹
'''

import pygame
import math
from Unit.BaseUnit import BaseUnit
from Parameter import *

class Archie(BaseUnit):
    
    def __init__(self, unit_id: int, unit_team: Team, usingAI = False, visible = True):
        
        self.unit_type = 'archie'
        self.body_image_path = 'Unit/Archie/archie.png'
        self.turret_image_path = 'Unit/Archie/turret.png'
        if unit_team == Team.ENEMY:
            self.body_image_path = 'Unit/Archie/enemy_archie.png'
            self.turret_image_path = 'Unit/Archie/enemy_turret.png'
        self.visible = visible
        self.max_speed_rate = 0.6
        self.max_acceleration_rate = 0.5
        self.min_acceleration_rate = -0.5
        self.max_angular_speed_rate = 0.3
        self.turret_angular_speed_rate = 0.5
        self.max_health_rate = 2.0
        self.sight_range = 100
        self.armor_type = ArmorType.MEDIUM                                                        # 护甲类型
        self.ammunition_types = ['rocket_shell']                                                # 单位拥有弹种
        self.ammo_switch_time =  UNIT_AMMO_SWITCH_TIME                                          # 单位切换弹种时间
        
        
        super().__init__(unit_id, unit_team, usingAI,
                         unit_type=self.unit_type,
                         body_image_path=self.body_image_path, 
                         turret_image_path=self.turret_image_path, 
                         visible=self.visible,
                         max_speed_rate=self.max_speed_rate, 
                         max_acceleration_rate=self.max_acceleration_rate, 
                         min_acceleration_rate=self.min_acceleration_rate, 
                         max_angular_speed_rate=self.max_angular_speed_rate, 
                         turret_angular_speed_rate=self.turret_angular_speed_rate, 
                         max_health_rate=self.max_health_rate, 
                         sight_range=self.sight_range,
                         armor_type=self.armor_type, 
                         ammunition_types=self.ammunition_types, 
                         ammo_switch_time=self.ammo_switch_time)

def create_archie(unit_id, unit_team, position=(0, 0), usingAI = False, visible = True):
    # 示例：archie = create_archie(1, Team.PLAYER, position=(400, 300))
    archie = Archie(unit_id, unit_team, usingAI, visible)
    archie.position = position
    return archie            

def create_enemy_archie(unit_id, position=(0, 0), usingAI = False, visible = True):
    enemy = Archie(unit_id, Team.ENEMY, usingAI, visible)
    enemy.position = position
    return enemy

def create_player_archie(unit_id, position=(0, 0), usingAI = False, visible = True):
    player = Archie(unit_id, Team.PLAYER, usingAI, visible)
    player.position = position
    return player