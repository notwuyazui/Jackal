'''
地图绘制模块
根据字符地图绘制游戏地图
'''

import random
from typing import List, Tuple
import pygame
import os
from Parameter import *
from utils import *


'''
    class GameMap:
        Attributes:
            map_data : List[str]    字符地图数据矩阵
            tile_size : int         每个地图块的像素大小
            
        Methods:
            draw(surface:pygame.Surface, camera_offset:Tuple[int, int]) -> None         绘制地图
            save_to_file(file_name: str) -> bool                                        保存地图
            save() -> bool                                                              便捷保存
            load_from_file(file_path: str, tile_size: int = 64) -> Optional['GameMap']  从文件加载地图（类方法）
            
            check_collision(rect:pygame.Rect) -> bool                                   检查矩形是否与地图障碍物碰撞
            is_walkable(x: int, y: int, width: int = 0, height: int = 0) -> bool        检查指定区域是否可通行
            get_colliding_obstacles(rect:pygame.Rect) -> List[pygame.Rect]              获取与矩形碰撞的障碍物列表
            get_map_size() -> Tuple[int, int]                                           获取地图尺寸
            get_tile_at_position(x: int, y: int) -> Optional[str]                       获取指定像素位置的格子字符
            
    Functions:
        create_map_from_strings(strings: List[str], tile_size: int = 64) -> 'GameMap'               从字符串列表创建地图（类方法）
        create_map_from_file(file_path: str, tile_size: int = 64) -> Optional['GameMap']            从文件创建地图（类方法）
        create_test_map() -> 'GameMap'                                                              创建测试地图
        create_empty_map(width: int = 10, height: int = 10) -> 'GameMap'                            创建空地图
        create_border_map(width: int = 10, height: int = 10) -> 'GameMap'                           创建有边框的地图
        create_random_map(width: int = 10, height: int = 10, density: float = 0.3) -> 'GameMap'     创建随机地图
'''

class GameMap:
    def __init__(self, map_data, tile_size=64):
        self.map_data = map_data
        self.tile_size = tile_size      # 地图块大小
        self.height = len(map_data)
        self.width = len(map_data[0]) if map_data else 0
        
        # 加载地形图片
        self.flat_image = self._load_image("Map/flat.png")          # 平地
        self.barrier_image = self._load_image("Map/barrier.png")    # 障碍物
        
        self.map_surface = pygame.Surface((self.width * tile_size, self.height * tile_size))
        
        self.obstacles = []
        
        # 生成地图
        self._generate_map()
    
    def draw(self, surface: pygame.Surface, camera_offset=(0, 0)) -> None:
        map_x = -camera_offset[0]
        map_y = -camera_offset[1]
        
        surface.blit(self.map_surface, (map_x, map_y))
        
        # 绘制障碍物边框
        if hasattr(self, 'debug_draw') and self.debug_draw:
            self._draw_debug(surface, camera_offset)
        
    def save_to_file(self, file_name: str) -> bool:
        # 保存地图到文件, file_name为文件名，包含后缀，不包括路径
        try:
            file_path = os.path.join(DEFAULT_MAP_PATH, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                for row in self.map_data:
                    f.write(row + '\n')
            
            print(f"地图已保存到: {file_path}")
            return True
            
        except Exception as e:
            print(f"保存地图失败: {e}")
            return False
        
    def save(self) -> bool:
        # 提供一种直接的保存方法
        save_path = get_next_filename(DEFAULT_MAP_PATH, 'default_map', '.txt')
        return self.save_to_file(save_path)
    
    @classmethod
    def load_from_file(cls, file_name: str, tile_size=64) -> 'GameMap':
        file_path = os.path.join(DEFAULT_MAP_PATH, file_name)
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

    def check_collision(self, rect: pygame.Rect) -> bool:
        # 检查矩形是否与任何障碍物碰撞
        for obstacle in self.obstacles:
            if rect.colliderect(obstacle):
                return True
        return False
    
    def is_walkable(self, x: float, y: float, width=0, height=0) -> bool:
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
    
    def get_colliding_obstacles(self, rect: pygame.Rect) -> List[pygame.Rect]:
        # 获取与指定矩形碰撞的所有障碍物
        colliding = []
        for obstacle in self.obstacles:
            if rect.colliderect(obstacle):
                colliding.append(obstacle)
        return colliding
    
    def get_map_size(self) -> Tuple[int, int]:
        return (
            self.width * self.tile_size,
            self.height * self.tile_size
        )
    
    def get_tile_at_position(self, x: float, y: float) -> str:
        # 获取指定位置的格子
        tile_x = x // self.tile_size
        tile_y = y // self.tile_size
        
        if 0 <= tile_x < self.width and 0 <= tile_y < self.height:
            return self.map_data[tile_y][tile_x]
        
        return None
    
    def _load_image(self, image_path: str) -> pygame.Surface:
        image = load_image(image_path)
        return pygame.transform.scale(image, (self.tile_size, self.tile_size))      # 调整到指定大小
    
    def _generate_map(self) -> None:
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
    
    def _draw_debug(self, surface: pygame.Surface, camera_offset: Tuple[int, int]) -> None:
        for obstacle in self.obstacles:
            screen_x = obstacle.x - camera_offset[0]
            screen_y = obstacle.y - camera_offset[1]
            
            pygame.draw.rect(
                surface,
                (255, 0, 0),
                (screen_x, screen_y, obstacle.width, obstacle.height),
                2  # 边框宽度
            )
    


# 便捷函数
def create_map_from_strings(map_strings: List[str], tile_size=64) -> GameMap:
    return GameMap(map_strings, tile_size)

def create_map_from_file(file_name: str, tile_size=64) -> GameMap:
    return GameMap.load_from_file(file_name, tile_size)

# 预定义地图
def create_test_map() -> GameMap:
    """创建测试地图"""
    return GameMap.load_from_file("test_map.txt")

def create_empty_map(width=15, height=10) -> GameMap:
    # 创建空地图
    map_data = ["o" * width for _ in range(height)]
    return GameMap(map_data)

def create_border_map(width=15, height=10) -> GameMap:
    # 创建有边框的地图
    map_data = []
    for y in range(height):
        if y == 0 or y == height - 1:
            map_data.append("x" * width)
        else:
            row = "x" + "o" * (width - 2) + "x"
            map_data.append(row)
    return GameMap(map_data)

def create_random_map(width=15, height=10, density=0.3) -> GameMap:
    # 创建随机地图, 但是会生成一个边界
    map_data = []
    for y in range(height):
        row = []
        for x in range(width):
            is_border = (y == 0 or y == height - 1 or x == 0 or x == width - 1)
            if is_border:
                # 边界总是障碍物
                row.append('x')
            else:
                # 内部区域根据密度随机生成
                if random.random() < density:
                    row.append('x')
                else:
                    row.append('o') 
        map_data.append(''.join(row))
    return GameMap(map_data)