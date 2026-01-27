import pygame
import math
import sys
import os
from Unit.Tank.Tank import *
from Map.Map import *
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
    
    if DRAW_STATUS_MESSAGE or DEBUG_MODE:
        debug_text = [
            f"position: ({tank.position[0]:.1f}, {tank.position[1]:.1f})",
            f"speed: {tank.speed:.1f}",
            f"acc: {tank.acceleration:.1f}",
            f"target_angle: {tank.angular_speed:.1f}",
        ]
        for i, text in enumerate(debug_text):
            text_surface = debug_font.render(text, True, (255, 255, 255))
            surface.blit(text_surface, (screen_x + 40, screen_y - 40 + i * 20))

if __name__ == "__main__":
    pygame.init()
    
    screen_width, screen_height = 960, 640
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
    ]
    
    font = None
    for path in font_paths:
        try:
            font = pygame.font.Font(path, 24)
            break
        except:
            continue
    
    # map_data = [
    #     "xxxxxxxxxxxxxxxxxxx",
    #     "xoooooooooooooooxxx",
    #     "xooxxxxxxxxxxoooxxx",
    #     "xooxxxoooxxoooooxxx",
    #     "xoooooooooooooooxxx",
    #     "xooxxxxxxxxxxoooxxx",
    #     "xooooxoooxxoooooxxx",
    #     "xxxoooooooooooooxxx",
    #     "xooxxxxxxxxxxoooxxx",
    #     "xooxxxoooxxoooooxxx",
    #     "xoooooooooooooooxxx",
    #     "xxxxxxxxxxxxxxxxxxx"
    # ]
    
    # game_map = GameMap(map_data, tile_size=64)
    game_map = create_random_map()
    
    tank = create_tank(1, Team.PLAYER, position=(400, 300))
    
    # 相机偏移
    camera_offset = [0, 0]
    camera_speed = 5
    
    # 控制状态
    moving_forward = False
    moving_backward = False
    turning_left = False
    turning_right = False
    
    debug_mode = DEBUG_MODE
    
    running = True
    while running:
        delta_time = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    moving_forward = True
                elif event.key == pygame.K_s:
                    moving_backward = True
                elif event.key == pygame.K_a:
                    turning_left = True
                elif event.key == pygame.K_d:
                    turning_right = True
                elif event.key == pygame.K_F1:
                    debug_mode = not debug_mode
                elif keys[pygame.K_p]:
                    tank.save()
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    moving_forward = False
                elif event.key == pygame.K_s:
                    moving_backward = False
                elif event.key == pygame.K_a:
                    turning_left = False
                elif event.key == pygame.K_d:
                    turning_right = False
        
        # 处理相机移动
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            camera_offset[0] -= camera_speed
        if keys[pygame.K_RIGHT]:
            camera_offset[0] += camera_speed
        if keys[pygame.K_UP]:
            camera_offset[1] -= camera_speed
        if keys[pygame.K_DOWN]:
            camera_offset[1] += camera_speed
        
        # 坦克的状态更新
        tank.set_movement(forward=moving_forward, backward=moving_backward)
        tank.set_turning(left=turning_left, right=turning_right)
        
        # 炮塔指向鼠标
        mouse_pos = pygame.mouse.get_pos()
        tank.set_turret_target_to_mouse(mouse_pos, camera_offset)
        
        tank.update(delta_time, game_map.obstacles)
        
        screen.fill((50, 50, 70))
        
        game_map.draw(screen, camera_offset)
        tank.draw(screen, camera_offset)
        
        draw_debug_info(screen, tank, camera_offset, mouse_pos)

        pygame.display.flip()
    
    pygame.quit()
    sys.exit()