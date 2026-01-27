'''
    一个坦克类
'''

import pygame
import math
from Unit.BaseUnit import BaseUnit
from Parameter import *

class Tank(BaseUnit):
    
    def __init__(self, unit_id, unit_team):
        
        self.unit_type = 'tank'
        self.body_image_path = 'Unit/Tank/tank.png'
        self.turret_image_path = 'Unit/Tank/turret.png'
        self.max_speed_rate = 1.0
        self.max_acceleration_rate = 1.0
        self.min_acceleration_rate = -1.0
        self.max_angular_speed_rate = 1.0
        self.turret_angular_speed_rate = 1.0
        self.max_health_rate = 1.0
        self.armor_type = ArmorType.NONE                                                        # 护甲类型
        self.ammunition_types = ['bullet']                                                      # 单位拥有弹种
        self.ammo_switch_time =  UNIT_AMMO_SWITCH_TIME                                          # 单位切换弹种时间
        
        
        super().__init__(unit_id, unit_team, 
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

def create_tank(unit_id, unit_team, position=(0, 0)):
    tank = Tank(unit_id, unit_team)
    tank.position = position
    return tank            
