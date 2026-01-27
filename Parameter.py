'''
    此处定义一些基本属性
'''
import numpy as np
import enum

FPS = 60
DELTA_TIME = 1.0 / FPS

UNIT_ACC_INF = 10000                # 当需要无视加速度时，加速度取此值
UNIT_ANGULAR_SPEED_INF = 10000
UNIT_TURRET_ANGULAR_SPEED_INF = 10000

UNIT_SPEED = 50.0                       # 单位的基础速度
UNIT_ACC = 100.0                        # 单位的基础加速度
UNIT_HEALTH = 100                       # 单位的基础血量
UNIT_ANGULAR_SPEED = 100.0              # 单位的基础旋转速度
UNIT_TURRET_ANGULAR_SPEED = 100.0       # 单位炮塔的基础旋转速度
UNIT_AMMO_SWITCH_TIME = 0.5             # 单位切换弹药的时间

BULLET_SPEED = 10.0     # 子弹的基础速度
BULLET_DAMAGE = 10      # 子弹的基础伤害

UNIT_DIAGONAL_SPEED = UNIT_SPEED / np.sqrt(2)       # 单位对角线速度
BULLET_DIAGONAL_SPEED = BULLET_SPEED / np.sqrt(2)   # 子弹对角线速度


class ArmorType(enum.Enum):     # 单位护甲类型
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3