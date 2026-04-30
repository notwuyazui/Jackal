import pygame
import sys
from Unit.Tank.Tank import *
from Unit.Archie.Archie import *
from Unit.Plane.Plane import *
from Unit.EnemyAI import *
from Map.GameMap import *
from GameMode import *
from Bullet.NormalShell.NormalShell import *
from Bullet.RocketShell.RocketShell import *
from Bullet.HeavyShell.HeavyShell import *
from Bullet.BulletManager import *
from Unit.UnitManager import *
from GameManager import *
from PCControl import *


def init_game():
    game_manager = GameManager()
    game_manager.set_test_map()
    game_manager.add_player_tank(unit_id=0, position=(100,500), usingAI=False, visible=True)      # 添加一个玩家坦克
    
    game_manager.add_player_tank(unit_id=101, position = (256, 448), usingAI = True)
    game_manager.add_player_tank(unit_id=102, position = (480, 448), usingAI = True)
    game_manager.add_player_tank(unit_id=103, position = (768, 448), usingAI = True)
    game_manager.add_player_archie(unit_id=104, position = (256, 512), usingAI = True)
    game_manager.add_player_plane(unit_id=105, position = (480, 512), usingAI = True)
    game_manager.add_player_archie(unit_id=106, position = (768, 512), usingAI = True)
    game_manager.add_enemy_archie(unit_id=201, position = (256, 192), usingAI = True)
    game_manager.add_enemy_archie(unit_id=202, position = (480, 192), usingAI = True)
    game_manager.add_enemy_archie(unit_id=203, position = (768, 192), usingAI = True)
    game_manager.add_enemy_tank(unit_id=204, position = (256, 128), usingAI = True)
    game_manager.add_enemy_plane(unit_id=205, position = (480, 128), usingAI = True)
    game_manager.add_enemy_tank(unit_id=206, position = (768, 128), usingAI = True)



    return game_manager

if __name__ == "__main__":
    pygame.init()
    
    screen_width, screen_height = 960, 640
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("test")
    clock = pygame.time.Clock()
    
    font = set_font()
    
    # 添加地图和单位
    game_manager = GameManager()
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
