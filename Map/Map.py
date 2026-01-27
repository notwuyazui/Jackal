'''
地图绘制模块
根据字符地图绘制游戏地图
'''

import pygame
import os

class GameMap:
    
    def __init__(self, map_data, tile_size=64):
        
        self.map_data = map_data
        self.tile_size = tile_size      # 地图块大小
        self.height = len(map_data)
        self.width = len(map_data[0]) if map_data else 0
        
        # 加载地形图片
        self.flat_image = self._load_image("Map/flat.png", "平地图")
        self.barrier_image = self._load_image("Map/barrier.png", "障碍物图")
        
        self.map_surface = pygame.Surface((self.width * tile_size, self.height * tile_size))
        
        self.obstacles = []
        
        # 生成地图
        self._generate_map()
    
    def _load_image(self, image_path, image_name):
        try:
            image = pygame.image.load(image_path)
            # 调整到指定大小
            return pygame.transform.scale(image, (self.tile_size, self.tile_size))
        except pygame.error as e:
            print(f"无法加载{image_name}: {image_path}")
            print(f"错误信息: {e}")
            exit(1)
    
    def _generate_map(self):
        """生成地图表面和障碍物列表"""
        if not self.map_data:
            print("地图数据为空")
            return
        
        self.obstacles = []
        
        for y, row in enumerate(self.map_data):
            for x, cell in enumerate(row):
                pos_x = x * self.tile_size
                pos_y = y * self.tile_size
                
                if cell == 'o':  # 平地
                    self.map_surface.blit(self.flat_image, (pos_x, pos_y))
                    
                elif cell == 'x':  # 障碍物
                    self.map_surface.blit(self.barrier_image, (pos_x, pos_y))
                    
                    obstacle_rect = pygame.Rect(
                        pos_x,
                        pos_y,
                        self.tile_size,
                        self.tile_size
                    )
                    self.obstacles.append(obstacle_rect)
                    
                else:
                    print(f"地图位置({x},{y})有未知字符 '{cell}'，默认为平地")
                    self.map_surface.blit(self.flat_image, (pos_x, pos_y))
    
    def draw(self, surface, camera_offset=(0, 0)):
        map_x = -camera_offset[0]
        map_y = -camera_offset[1]
        
        surface.blit(self.map_surface, (map_x, map_y))
        
        # 绘制障碍物边框
        if hasattr(self, 'debug_draw') and self.debug_draw:
            self._draw_debug(surface, camera_offset)
    
    def _draw_debug(self, surface, camera_offset):
        for obstacle in self.obstacles:
            screen_x = obstacle.x - camera_offset[0]
            screen_y = obstacle.y - camera_offset[1]
            
            pygame.draw.rect(
                surface,
                (255, 0, 0),
                (screen_x, screen_y, obstacle.width, obstacle.height),
                2  # 边框宽度
            )
    
    def check_collision(self, rect):
        # 检查矩形是否与任何障碍物碰撞
        for obstacle in self.obstacles:
            if rect.colliderect(obstacle):
                return True
        return False
    
    def get_colliding_obstacles(self, rect):
        # 获取与指定矩形碰撞的所有障碍物
        colliding = []
        for obstacle in self.obstacles:
            if rect.colliderect(obstacle):
                colliding.append(obstacle)
        return colliding
    
    def get_map_size(self):
        return (
            self.width * self.tile_size,
            self.height * self.tile_size
        )
    
    def get_tile_at_position(self, x, y):
        # 获取指定位置的格子
        tile_x = x // self.tile_size
        tile_y = y // self.tile_size
        
        if 0 <= tile_x < self.width and 0 <= tile_y < self.height:
            return self.map_data[tile_y][tile_x]
        
        return None
    
    def is_walkable(self, x, y, width=0, height=0):
        # 创建要检查的矩形
        if width > 0 and height > 0:
            rect = pygame.Rect(x, y, width, height)
        else:
            # 如果没指定大小，检查单个格子
            tile_x = x // self.tile_size
            tile_y = y // self.tile_size
            
            if 0 <= tile_x < self.width and 0 <= tile_y < self.height:
                return self.map_data[tile_y][tile_x] != 'x'
            return False
        
        # 检查是否与任何障碍物碰撞
        return not self.check_collision(rect)
    
    def save_to_file(self, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for row in self.map_data:
                    f.write(row + '\n')
            
            print(f"地图已保存到: {file_path}")
            return True
            
        except Exception as e:
            print(f"保存地图失败: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path, tile_size=64):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                map_data = [line.strip() for line in f if line.strip()]
            
            if not map_data:
                print(f"文件 {file_path} 为空")
                return None
            
            # 检查所有行长度是否一致
            first_length = len(map_data[0])
            for i, row in enumerate(map_data):
                if len(row) != first_length:
                    print(f"第 {i} 行长度不一致")
            
            print(f"地图已从 {file_path} 加载")
            return cls(map_data, tile_size)
            
        except FileNotFoundError:
            print(f"地图文件不存在: {file_path}")
            return None
        except Exception as e:
            print(f"加载地图失败: {e}")
            return None


# 便捷函数
def create_map_from_strings(map_strings, tile_size=64):
    return GameMap(map_strings, tile_size)

def create_map_from_file(file_path, tile_size=64):
    return GameMap.load_from_file(file_path, tile_size)


# 预定义地图
def create_test_map():
    """创建测试地图"""
    map_data = [
        "oooxxxooo",
        "ooooooooo", 
        "oooxxxxxx",
        "ooooooooo",
        "xxxxxxooo",
        "ooooooooo",
    ]
    return GameMap(map_data)


def create_empty_map(width=10, height=10):
    # 创建空地图
    map_data = ["o" * width for _ in range(height)]
    return GameMap(map_data)


def create_border_map(width=10, height=10):
    # 创建有边框的地图
    map_data = []
    
    for y in range(height):
        if y == 0 or y == height - 1:
            map_data.append("x" * width)
        else:
            row = "x" + "o" * (width - 2) + "x"
            map_data.append(row)
    
    return GameMap(map_data)


if __name__ == "__main__":
    # 初始化pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("地图测试")
    clock = pygame.time.Clock()
    
    # 创建地图
    # 从字符串列表
    map_data = [
        "oooxxxooo",
        "ooooooooo",
        "oooxxxxxx",
        "ooooooooo",
        "xxxxxxooo",
        "ooooooooo",
    ]
    game_map = GameMap(map_data)
    
    # 从文件
    # game_map = GameMap.load_from_file("maps/test_map.txt")
    
    # 使用便捷函数
    # game_map = create_test_map()
    
    # 打印地图信息
    print(f"地图尺寸: {game_map.width}x{game_map.height}")
    print(f"地图像素尺寸: {game_map.get_map_size()}")
    print(f"障碍物数量: {len(game_map.obstacles)}")
    
    # 测试碰撞检测
    test_rect = pygame.Rect(100, 100, 50, 50)
    if game_map.check_collision(test_rect):
        print(f"测试矩形 {test_rect} 与障碍物碰撞")
    else:
        print(f"测试矩形 {test_rect} 未与障碍物碰撞")
    
    # 主循环
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
                    game_map.save_to_file("maps/saved_map.txt")
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