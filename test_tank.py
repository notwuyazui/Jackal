import pygame
import math
import sys
import os
from Unit.Tank.Tank import *
from Map.Map import *

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
    pygame.draw.line(surface, (255, 0, 255), (screen_x, screen_y), (screen_mouse_x, screen_mouse_y), 1)
    pygame.draw.circle(surface, (255, 0, 255), (int(screen_mouse_x), int(screen_mouse_y)), 3)
    
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
    
    screen_width, screen_height = 1000, 700
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
    
    map_data = [
        "xxxxxxxxxxxxxxxxxxx",
        "xoooooooooooooooxxx",
        "xooxxxxxxxxxxoooxxx",
        "xooxxxoooxxoooooxxx",
        "xoooooooooooooooxxx",
        "xooxxxxxxxxxxoooxxx",
        "xooooxoooxxoooooxxx",
        "xxxoooooooooooooxxx",
        "xooxxxxxxxxxxoooxxx",
        "xooxxxoooxxoooooxxx",
        "xoooooooooooooooxxx",
        "xxxxxxxxxxxxxxxxxxx"
    ]
    
    game_map = GameMap(map_data, tile_size=64)
    print("地图创建成功")
    
    tank = create_tank("test_tank", "red", position=(400, 300))
    
    # 相机偏移
    camera_offset = [0, 0]
    camera_speed = 5
    
    # 控制状态
    moving_forward = False
    moving_backward = False
    turning_left = False
    turning_right = False
    
    debug_mode = True
    
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
        if hasattr(tank, 'set_movement'):
            tank.set_movement(forward=moving_forward, backward=moving_backward)
        
        if hasattr(tank, 'set_turning'):
            tank.set_turning(left=turning_left, right=turning_right)
        
        # 炮塔指向鼠标
        mouse_pos = pygame.mouse.get_pos()
        if hasattr(tank, 'set_turret_target_to_mouse'):
            tank.set_turret_target_to_mouse(mouse_pos, camera_offset)
        
        if hasattr(tank, 'update_with_collision') and hasattr(game_map, 'obstacles'):
            tank.update_with_collision(delta_time, game_map.obstacles)
        elif hasattr(tank, 'update'):
            tank.update(delta_time)
        
        screen.fill((50, 50, 70))
        
        game_map.draw(screen, camera_offset)
        
        if hasattr(tank, 'draw'):
            tank.draw(screen, camera_offset)
        
        if debug_mode:
            draw_debug_info(screen, tank, camera_offset, mouse_pos)

        pygame.display.flip()
    
    pygame.quit()
    sys.exit()