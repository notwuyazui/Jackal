'''
    此处定义一些基本属性
'''
import numpy as np
import enum

FPS = 60
DELTA_TIME = 1.0 / FPS

INF = 10000                # 当需要无视加速度/速度时，取此值

UNIT_SPEED = 50.0                       # 单位的基础速度
UNIT_ACC = 100.0                        # 单位的基础加速度
UNIT_HEALTH = 100                       # 单位的基础血量
UNIT_ANGULAR_SPEED = 100.0              # 单位的基础旋转速度
UNIT_TURRET_ANGULAR_SPEED = 100.0       # 单位炮塔的基础旋转速度
UNIT_AMMO_SWITCH_TIME = 0.5             # 单位切换弹药的时间

BULLET_SPEED = 500.0        # 子弹的基础速度
BULLET_DAMAGE = 10          # 子弹的基础伤害
BULLET_EXPLOSION_DAMAGE_APPLY_DISTANT_FACTOR = False   # 子弹爆炸造成的伤害是否与距离有关
EXPLOSION_IMAGE_ADAPT_TO_RANGE = False               # 爆炸图片是否根据爆炸范围调整大小

# UNIT_DIAGONAL_SPEED = UNIT_SPEED / np.sqrt(2)       # 单位对角线速度
# BULLET_DIAGONAL_SPEED = BULLET_SPEED / np.sqrt(2)   # 子弹对角线速度


class ArmorType(enum.Enum):     # 单位护甲类型
    NONE = 0
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3
class Team(enum.Enum):
    PLAYER = 0
    ENEMY = 1
    NEUTRAL = 2
    
DEFAULT_MAP_PATH = 'Map\saved'
DEFAULT_UNIT_PATH = 'Unit\saved'
DEFAULT_BULLET_PATH = 'Bullet\saved'
