from Unit.Tank.Tank import *
from Map.Map import *

pygame.init()
clock = pygame.time.Clock()
screen = pygame.display.set_mode((800, 600))

map_data = [
    "oooxxxoooxxoo",
    "ooooooooooooo",
    "oooxxxxxxxxxx",
    "oooxxxoooxxoo",
    "xxxoooooooooo",
    "oooxxxxxxxxxx",
    "oooxxxoooxxoo",
    "ooooooooooooo",
    "oooxxxxxxxxxx",
    "oooxxxoooxxoo",
    "ooooooooooooo",
    "oooxxxxxxxxxx"
]
game_map = GameMap(map_data, tile_size=64)
game_map.draw(screen)

running = True
camera_x, camera_y = 0, 0
move_speed = 5

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_s:
                # 保存地图
                game_map.save()
            elif event.key == pygame.K_d:
                # 切换调试显示
                game_map.debug_draw = not getattr(game_map, 'debug_draw', False)
    
    # 相机移动（用箭头键）
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        camera_x -= move_speed
    if keys[pygame.K_RIGHT]:
        camera_x += move_speed
    if keys[pygame.K_UP]:
        camera_y -= move_speed
    if keys[pygame.K_DOWN]:
        camera_y += move_speed
    
    # 绘制
    screen.fill((50, 50, 50))  # 灰色背景
    game_map.draw(screen, (camera_x, camera_y))
    
    # 绘制信息
    font = pygame.font.Font(None, 24)
    info_text = f"相机位置: ({camera_x}, {camera_y}) | 箭头键移动相机 | S保存地图 | D切换调试"
    text_surface = font.render(info_text, True, (255, 255, 255))
    screen.blit(text_surface, (10, 10))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()