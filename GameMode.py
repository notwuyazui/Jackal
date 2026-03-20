'''
    在此处设定和更改游戏模式
    主要是控制可显示的信息
'''

DEBUG_MODE = False                  # 一键开启调试模式

DRAW_HEALTH_BAR = True              # 绘制坦克血条
DRAW_SIGHT_RANGE = True             # 绘制坦克视野范围
DRAW_MOUSE_TARGET_LINE = True       # 绘制鼠标瞄准线
DRAW_STATUS_MESSAGE = False         # 绘制单位状态信息
DRAW_ENEMY_STATE_MESSAGE = False    # 绘制敌方坦克状态信息
DRAW_ENEMY_SEE_LINE = False         # 绘制敌方坦克视野线
DRAW_BULLET_EXPLOSION_RANGE = True  # 绘制子弹爆炸范围
DRAW_BULLET_BOUNDING_BOX = False    # 绘制子弹碰撞箱
DRAW_OBSTACLE_BOUNDING_BOX = False  # 绘制障碍物碰撞箱

BULLET_INFO_TEXT = False            # 增加子弹发射和移除时的文本提示
ENEMY_STATE_TEXT = False            # 增加敌方坦克状态信息文本提示
UNIT_RECORD_TEXT = False            # 单位记录文本
PRINT_VISIBLE_UNIT = False          # 打印unit_0的可见单位（包含视野与通过通信同步的单位）

AUTO_COMMUNICATE = True             # 自动通信

USE_TEAR_DROP_VISION = False        # 使用水滴形视野，当此项为false时使用圆形视野
