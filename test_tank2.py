import pygame
import math
import sys
import os
import random
from Unit.Tank.Tank import *
from Map.Map import *
from GameMode import *
from Bullet.NormalShell.NormalShell import NormalShell  # 导入普通炮弹
from Unit.EnemyAI import *

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
    
    if DRAW_ENEMY_STATE_MESSAGE or DEBUG_MODE:
        # 绘制敌人AI状态
        debug_font = pygame.font.Font(None, 16)
        for i, ai in enumerate(enemy_ais):
            if ai.enemy.is_alive:
                screen_x = ai.enemy.position[0] - camera_offset[0]
                screen_y = ai.enemy.position[1] - camera_offset[1]
                
                # 绘制状态文本
                state_text = f"State: {ai.ai_state}"
                state_surface = debug_font.render(state_text, True, (255, 255, 0))
                screen.blit(state_surface, (screen_x + 40, screen_y - 20))
                
                # 绘制视线
                if DRAW_ENEMY_SEE_LINE or DEBUG_MODE:
                    if ai.can_see_player and tank.is_alive:
                        pygame.draw.line(screen, (255, 0, 0), 
                                        (screen_x, screen_y),
                                        (tank.position[0] - camera_offset[0], 
                                            tank.position[1] - camera_offset[1]),
                                        1)

class BulletManager:
    """子弹管理器，负责管理所有子弹的更新和绘制"""
    
    def __init__(self):
        self.bullets = []  # 存储所有活跃的子弹
        self.to_remove = []  # 存储待移除的子弹
        
    def add_bullet(self, bullet):
        """添加新子弹"""
        if bullet:
            self.bullets.append(bullet)
            if BULLET_INFO_TEXT or DEBUG_MODE:
                print(f"子弹发射: ID={bullet.id}, 位置={bullet.position}")
    
    def update(self, delta_time, units, obstacles):
        """更新所有子弹"""
        self.to_remove.clear()
        
        for bullet in self.bullets:
            # 更新子弹状态
            if not bullet.update(delta_time, units, obstacles):
                # 子弹不再活跃，标记为待移除
                if bullet in self.bullets and bullet not in self.to_remove:
                    self.to_remove.append(bullet)
            
            # 如果子弹爆炸了，应用爆炸伤害
            if bullet.has_exploded and bullet.is_explosive:
                bullet.apply_explosion_damage(units)
        
        # 移除不再活跃的子弹
        for bullet in self.to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
                if BULLET_INFO_TEXT or DEBUG_MODE:
                    print(f"子弹移除: ID={bullet.id}")
    
    def draw(self, surface, camera_offset):
        """绘制所有子弹"""
        for bullet in self.bullets:
            bullet.draw(surface, camera_offset)
    
    def get_active_count(self):
        """获取活跃子弹数量"""
        return len([b for b in self.bullets if b.is_active])
    
    def clear(self):
        """清空所有子弹"""
        self.bullets.clear()



def create_enemy_tank(unit_id, position=(0, 0)):
    """创建敌方坦克"""
    enemy = Tank(unit_id, Team.ENEMY)
    enemy.position = position
    return enemy

if __name__ == "__main__":
    pygame.init()
    
    screen_width, screen_height = 960, 640
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("坦克对战测试 - 敌方AI攻击")
    clock = pygame.time.Clock()
    
    # 字体设置
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
    
    if font is None:
        font = pygame.font.Font(None, 24)  # 使用默认字体
    
    # 创建地图
    game_map = create_border_map(width=15, height=10)
    
    # 创建玩家坦克
    tank = create_tank(1, Team.PLAYER, position=(400, 300))
    
    # 创建子弹管理器
    bullet_manager = BulletManager()
    
    # 创建敌方坦克和AI控制器
    enemies = []
    enemy_ais = []
    
    for i in range(3):
        # 为每个敌人随机生成位置，确保不重叠
        enemy_x = random.randint(100, 700)
        enemy_y = random.randint(100, 500)
        enemy = create_enemy_tank(100 + i, position=(enemy_x, enemy_y))
        enemies.append(enemy)
        
        # 为每个敌人创建AI控制器
        ai = EnemyAI(enemy, tank, bullet_manager, game_map.obstacles)
        enemy_ais.append(ai)
    
    # 相机偏移
    camera_offset = [0, 0]
    camera_speed = 5
    
    # 玩家控制状态
    moving_forward = False
    moving_backward = False
    turning_left = False
    turning_right = False
    
    # 鼠标状态
    mouse_left_pressed = False
    mouse_left_was_pressed = False
    
    # 发射冷却时间
    fire_cooldown = 0.0
    fire_cooldown_max = 0.2  # 200毫秒冷却时间，防止连续发射
    
    debug_mode = DEBUG_MODE
    
    # 用于显示信息的表面
    info_surface = pygame.Surface((200, 120), pygame.SRCALPHA)
    info_surface.fill((0, 0, 0, 128))  # 半透明黑色背景
    
    running = True
    while running:
        delta_time = clock.tick(60) / 1000.0
        
        # 更新发射冷却
        if fire_cooldown > 0:
            fire_cooldown -= delta_time
        
        # 获取键盘状态
        keys = pygame.key.get_pressed()
        
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
                elif event.key == pygame.K_p:
                    tank.save()
                elif event.key == pygame.K_SPACE:
                    # 空格键发射子弹（备用）
                    bullet = tank.fire(NormalShell)
                    bullet_manager.add_bullet(bullet)
                elif event.key == pygame.K_c:
                    # 清空所有子弹
                    bullet_manager.clear()
                elif event.key == pygame.K_r:
                    # 重新生成敌人
                    enemies.clear()
                    enemy_ais.clear()
                    for i in range(3):
                        enemy_x = random.randint(100, 700)
                        enemy_y = random.randint(100, 500)
                        enemy = create_enemy_tank(100 + i, position=(enemy_x, enemy_y))
                        enemies.append(enemy)
                        ai = EnemyAI(enemy, tank, bullet_manager, game_map.obstacles)
                        enemy_ais.append(ai)
            
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    moving_forward = False
                elif event.key == pygame.K_s:
                    moving_backward = False
                elif event.key == pygame.K_a:
                    turning_left = False
                elif event.key == pygame.K_d:
                    turning_right = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 左键
                    mouse_left_pressed = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # 左键
                    mouse_left_pressed = False
        
        # 处理相机移动
        if keys[pygame.K_LEFT]:
            camera_offset[0] -= camera_speed
        if keys[pygame.K_RIGHT]:
            camera_offset[0] += camera_speed
        if keys[pygame.K_UP]:
            camera_offset[1] -= camera_speed
        if keys[pygame.K_DOWN]:
            camera_offset[1] += camera_speed
        
        # 鼠标左键发射子弹（有冷却时间）
        if mouse_left_pressed and fire_cooldown <= 0:
            # 发射子弹
            bullet = tank.fire(NormalShell)
            if bullet:
                bullet_manager.add_bullet(bullet)
                fire_cooldown = fire_cooldown_max
        
        mouse_left_was_pressed = mouse_left_pressed
        
        # 玩家坦克的状态更新
        tank.set_movement(forward=moving_forward, backward=moving_backward)
        tank.set_turning(left=turning_left, right=turning_right)
        
        # 炮塔指向鼠标
        mouse_pos = pygame.mouse.get_pos()
        tank.set_turret_target_to_mouse(mouse_pos, camera_offset)
        
        # 更新玩家坦克
        tank.update(delta_time, game_map.obstacles)
        
        # 更新敌方坦克AI
        for ai in enemy_ais:
            ai.update(delta_time)
        
        # 更新所有子弹
        all_units = [tank] + enemies
        bullet_manager.update(delta_time, all_units, game_map.obstacles)
        
        # 绘制
        screen.fill((50, 50, 70))
        
        # 绘制地图
        game_map.draw(screen, camera_offset)
        
        # 绘制所有单位
        for enemy in enemies:
            if enemy.is_alive:
                enemy.draw(screen, camera_offset)
        tank.draw(screen, camera_offset)
        
        # 绘制所有子弹
        bullet_manager.draw(screen, camera_offset)
        
        # 在调试模式下绘制视线和状态
        if True:
            draw_debug_info(screen, tank, camera_offset, mouse_pos)


        
        # 显示游戏状态
        if not tank.is_alive:
            game_over_font = pygame.font.Font(None, 72)
            game_over_surface = game_over_font.render("GAME OVER", True, (255, 50, 50))
            screen.blit(game_over_surface, (screen_width//2 - 180, screen_height//2 - 50))
        
        pygame.display.flip()
    
    pygame.quit()
    sys.exit()