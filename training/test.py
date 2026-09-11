import pygame
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from game.BattleWorld import BattleWorld
from game.Parameter import Team, FPS, ACC
from game.PCControl import PCControl
from game.utils import Action, set_font


def init_game():
    world = BattleWorld()
    world.load_builtin_map("test")
    units = [
        ("tank", 0, Team.PLAYER, (100, 500), False),
        ("tank", 101, Team.PLAYER, (256, 448), True),
        ("tank", 102, Team.PLAYER, (480, 448), True),
        ("tank", 103, Team.PLAYER, (768, 448), True),
        ("archie", 104, Team.PLAYER, (256, 512), True),
        ("plane", 105, Team.PLAYER, (480, 512), True),
        ("archie", 106, Team.PLAYER, (768, 512), True),
        ("archie", 201, Team.ENEMY, (256, 192), True),
        ("archie", 202, Team.ENEMY, (480, 192), True),
        ("archie", 203, Team.ENEMY, (768, 192), True),
        ("tank", 204, Team.ENEMY, (256, 128), True),
        ("plane", 205, Team.ENEMY, (480, 128), True),
        ("tank", 206, Team.ENEMY, (768, 128), True),
    ]
    for unit_type, unit_id, team, position, using_ai in units:
        world.create_unit(
            unit_type,
            team,
            position,
            unit_id=unit_id,
            using_ai=using_ai,
        )
    return world

if __name__ == "__main__":
    pygame.init()
    
    screen_width, screen_height = 960, 640
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("test")
    clock = pygame.time.Clock()
    
    font = set_font()
    
    # 添加地图和单位
    game_manager = init_game()
    
    # 用于显示信息的表面
    info_surface = pygame.Surface((200, 120), pygame.SRCALPHA)
    info_surface.fill((0, 0, 0, 128))  # 半透明黑色背景
    
    action = Action()
    
    running = True
    while running:
        delta_time = 1.0 / FPS
        clock.tick(FPS * ACC)
        
        # 应用键盘鼠标控制
        running, action = PCControl(game_manager, action, screen)
        game_manager.set_unit_action(0, action)
        
        # 更新、绘制
        game_manager.update(delta_time)
        screen.fill((50, 50, 70))
        game_manager.draw(screen)
        
        pygame.display.flip()
        
        # 每5秒打印一次单位记录
        game_manager.print_record()
    
    pygame.quit()
    sys.exit()
