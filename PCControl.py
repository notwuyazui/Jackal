import pygame
from GameManager import GameManager
from utils import *
from Parameter import *
from GameMode import *

def draw_debug_info(surface, tank, camera_offset, mouse_pos):
    debug_font = pygame.font.Font(None, 20)
    
    # 坦克中心点
    screen_x = tank.position[0] - camera_offset[0]
    screen_y = tank.position[1] - camera_offset[1]

    # 鼠标目标线
    world_mouse_x = mouse_pos[0] + camera_offset[0]
    world_mouse_y = mouse_pos[1] + camera_offset[1]
    screen_mouse_x = world_mouse_x - camera_offset[0]
    screen_mouse_y = world_mouse_y - camera_offset[1]
    if DRAW_MOUSE_TARGET_LINE or DEBUG_MODE:
        pygame.draw.line(surface, (255, 0, 255), (screen_x, screen_y), (screen_mouse_x, screen_mouse_y), 1)
        pygame.draw.circle(surface, (255, 0, 255), (int(screen_mouse_x), int(screen_mouse_y)), 3)

def PCControl(game_manager: GameManager, action: Action, screen):
        running = True
        action.switch_ammo = False
        
        keys = pygame.key.get_pressed()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    action.forward = True
                elif event.key == pygame.K_s:
                    action.backward = True
                elif event.key == pygame.K_a:
                    action.left = True
                elif event.key == pygame.K_d:
                    action.right = True
                elif event.key == pygame.K_F1:
                    debug_mode = not debug_mode
                elif event.key == pygame.K_p:
                    game_manager.save_map()
                elif event.key == pygame.K_SPACE:
                    # 空格键发射子弹
                    action.fire = True
                elif event.key == pygame.K_q:
                    action.switch_ammo = True
                elif event.key == pygame.K_c:
                    # 清空所有子弹
                    game_manager.clear_bullets()
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    action.forward = False
                elif event.key == pygame.K_s:
                    action.backward = False
                elif event.key == pygame.K_a:
                    action.left = False
                elif event.key == pygame.K_d:
                    action.right = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    action.fire = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 左键
                    action.fire = False
        
        # 处理相机移动
        if keys[pygame.K_LEFT]:
            game_manager.set_camera_offset_move(Direction.LEFT)
        if keys[pygame.K_RIGHT]:
            game_manager.set_camera_offset_move(Direction.RIGHT)
        if keys[pygame.K_UP]:
            game_manager.set_camera_offset_move(Direction.UP)
        if keys[pygame.K_DOWN]:
            game_manager.set_camera_offset_move(Direction.DOWN)
        
        action.mouse_pos = pygame.mouse.get_pos()      # 炮塔指向鼠标
        draw_debug_info(screen, game_manager.get_unit(0), game_manager.camera_offset, action.mouse_pos)
        
        return running, action