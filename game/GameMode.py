'''
    在此处设定和更改游戏模式的全局默认值。
    UnitManager 允许为每场战斗覆盖通信、视野和单位碰撞模式。
'''

DEBUG_MODE = False                  # 一键开启调试模式

DRAW_HEALTH_BAR = True              # 绘制单位血条
DRAW_SIGHT_RANGE = False            # 绘制单位视野范围
DRAW_ATTACK_RANGE = True           # 绘制单位当前弹药的攻击范围
DRAW_MOUSE_TARGET_LINE = False      # 绘制 unit_id=0 的鼠标瞄准线
DRAW_BULLET_EXPLOSION_RANGE = True  # 绘制子弹爆炸范围
DRAW_BULLET_BOUNDING_BOX = False    # 绘制子弹碰撞箱
DRAW_OBSTACLE_BOUNDING_BOX = False  # 绘制障碍物碰撞箱

BULLET_INFO_TEXT = False            # 增加子弹发射和移除时的文本提示
UNIT_RECORD_TEXT = True             # 调用 print_record 时打印单位记录
PRINT_VISIBLE_UNIT = False          # 调用 print_record 时打印 unit_id=0 的可见单位

AUTO_COMMUNICATE = False            # 自动通信（关闭：仅保留显式广播逻辑）

USE_TEAR_DROP_VISION = False        # 使用水滴形视野，当此项为false时使用圆形视野
ENABLE_UNIT_COLLISION = True        # 是否启用单位之间的物理碰撞
