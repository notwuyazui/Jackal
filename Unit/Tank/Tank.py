'''
    一个坦克类
    标准机动性，标准生存性，较高侦察范围，装备子弹、火箭弹和重炮弹
'''

import pygame
import math
from Unit.BaseUnit import BaseUnit
from Parameter import *

class Tank(BaseUnit):
    
    def __init__(self, unit_id: int, unit_team: Team, usingAI = False, visible = True):
        
        self.unit_type = 'tank'
        self.body_image_path = 'Unit/Tank/tank.png'
        if unit_team == Team.ENEMY:
            self.body_image_path = 'Unit/Tank/enemy_tank.png'
        self.turret_image_path = 'Unit/Tank/turret.png'
        self.visible = visible
        self.max_speed_rate = 1.0
        self.max_acceleration_rate = 1.0
        self.min_acceleration_rate = -1.0
        self.max_angular_speed_rate = 1.0
        self.turret_angular_speed_rate = 1.0
        self.max_health_rate = 1.0
        self.sight_range = 200
        self.armor_type = ArmorType.LIGHT                                                        # 护甲类型
        self.ammunition_types = ['normal_shell','rocket_shell','heavy_shell']                                                      # 单位拥有弹种
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

def create_tank(unit_id, unit_team, position=(0, 0), usingAI = False, visible = True):
    # 示例：tank = create_tank(1, Team.PLAYER, position=(400, 300))
    tank = Tank(unit_id, unit_team, usingAI, visible)
    tank.position = position
    return tank            

def create_enemy_tank(unit_id, position=(0, 0), usingAI = False, visible = True):
    enemy = Tank(unit_id, Team.ENEMY, usingAI, visible)
    enemy.position = position
    return enemy

def create_player_tank(unit_id, position=(0, 0), usingAI = False, visible = True):
    player = Tank(unit_id, Team.PLAYER, usingAI, visible)
    player.position = position
    return player