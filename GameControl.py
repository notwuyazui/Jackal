import pygame
import math
import sys
import os
from Unit.Tank.Tank import *
from Map.Map import *

def run():
    pygame.init()
    
    # 设置窗口
    screen_width, screen_height = 1000, 700
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("坦克控制测试")
    clock = pygame.time.Clock()
    
    # 尝试加载中文字体
    try:
        font_paths = [
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyh.ttc",    # 微软雅黑
        ]
        
        font = None
        for path in font_paths:
            try:
                font = pygame.font.Font(path, 24)
                print(f"成功加载字体: {path}")
                break
            except:
                continue
        
        if font is None:
            font = pygame.font.Font(None, 24)
            
    except Exception as e:
        print(f"加载字体时出错: {e}")
        font = pygame.font.Font(None, 24)
    
    # 创建坦克
    tank = create_tank("test_tank", "red", position=(400, 300))
    
    print(f"坦克最大速度: {tank.max_speed}")
    print(f"坦克最大加速度: {tank.max_accerattion}")
    print(f"坦克最大角速度: {tank.max_angular_speed}")
    
    # 相机偏移
    camera_offset = [0, 0]
    camera_speed = 5
    
    # 控制状态
    moving_forward = False
    moving_backward = False
    turning_left = False
    turning_right = False
    
    # 主循环
    running = True
    while running:
        # 计算时间增量
        delta_time = clock.tick(60) / 1000.0
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    moving_forward = True
                    print("W键按下: 前进")
                elif event.key == pygame.K_s:
                    moving_backward = True
                    print("S键按下: 后退")
                elif event.key == pygame.K_a:
                    turning_left = True
                    print("A键按下: 左转")
                elif event.key == pygame.K_d:
                    turning_right = True
                    print("D键按下: 右转")
                elif event.key == pygame.K_F1:
                    debug_mode = not debug_mode
                    print(f"调试模式: {'开启' if debug_mode else '关闭'}")
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    moving_forward = False
                    print("W键释放")
                elif event.key == pygame.K_s:
                    moving_backward = False
                    print("S键释放")
                elif event.key == pygame.K_a:
                    turning_left = False
                    print("A键释放")
                elif event.key == pygame.K_d:
                    turning_right = False
                    print("D键释放")
        
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
        
        # 设置坦克控制
        if hasattr(tank, 'set_movement'):
            tank.set_movement(forward=moving_forward, backward=moving_backward)
        
        if hasattr(tank, 'set_turning'):
            tank.set_turning(left=turning_left, right=turning_right)
        
        # 设置炮塔指向鼠标
        mouse_pos = pygame.mouse.get_pos()
        if hasattr(tank, 'set_turret_target_to_mouse'):
            tank.set_turret_target_to_mouse(mouse_pos, camera_offset)
        
        # 更新坦克
        if hasattr(tank, 'update_with_collision') and hasattr(game_map, 'obstacles'):
            tank.update_with_collision(delta_time, game_map.obstacles)
        elif hasattr(tank, 'update'):
            tank.update(delta_time)
        
        screen.fill((50, 50, 70))
        
        # 绘制地图
        game_map.draw(screen, camera_offset)
        
        # 绘制坦克
        if hasattr(tank, 'draw'):
            tank.draw(screen, camera_offset)

        pygame.display.flip()
    
    pygame.quit()
    sys.exit()