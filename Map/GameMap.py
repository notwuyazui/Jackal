import pygame
from Map.BaseTile import BaseTile
from Map.FlatTile.FlatTile import *
from Map.BarrierTile.BarrierTile import *
from Map.WaterTile.WaterTile import *
from typing import List, Tuple, Optional
import os
from Parameter import *
from GameMode import *
from utils import *
import random

CHAR_TO_TILE = {
    'o': FlatTile,
    'x': BarrierTile,
    'w': WaterTile,
}

class GameMap:
    def __init__(self, map_data=None, tile_size: int = 64):
        self.tile_size = tile_size
        self.tiles = []                      # 二维地块列表
        self.unit_obstacles = []             # 阻挡单位的矩形列表
        self.bullet_obstacles = []           # 阻挡子弹的矩形列表
        self.width = 0
        self.height = 0
        self.map_surface = None              # 地图表面（用于快速绘制）

        if map_data:
            if len(map_data) > 0:
                if isinstance(map_data[0], str):
                    # 字符串列表 → 构建地块
                    self._build_from_strings(map_data)
                else:
                    # 已经是地块列表
                    self.tiles = map_data
                    self.height = len(map_data)
                    self.width = len(map_data[0]) if map_data else 0
                    self._create_map_surface()
                    self._update_obstacles_from_tiles()
                    self._render_all()
        else:
            # 空地图
            pass

    def _build_from_strings(self, map_strings: List[str]) -> None:
        """从字符串列表构建地图"""
        self.height = len(map_strings)
        self.width = len(map_strings[0]) if map_strings else 0
        self.tiles = []
        self.unit_obstacles.clear()
        self.bullet_obstacles.clear()

        for row_idx, row_str in enumerate(map_strings):
            tile_row = []
            for col_idx, ch in enumerate(row_str):
                x = col_idx * self.tile_size
                y = row_idx * self.tile_size
                tile_class = CHAR_TO_TILE.get(ch, FlatTile)   # 未知字符默认平地
                print(x,y,ch)
                if not isinstance(tile_class, type):
                    raise TypeError(f"Expected a class, got {type(tile_class)}")
                tile = tile_class(x, y, self.tile_size)
                tile_row.append(tile)

                # 根据属性加入障碍物列表
                if tile.blocks_unit:
                    self.unit_obstacles.append(tile.rect)
                if tile.blocks_bullet:
                    self.bullet_obstacles.append(tile.rect)

            self.tiles.append(tile_row)

        self._create_map_surface()
        self._render_all()

    def _create_map_surface(self) -> None:
        """创建地图表面"""
        self.map_surface = pygame.Surface((self.width * self.tile_size, self.height * self.tile_size))

    def _render_tile(self, row: int, col: int) -> None:
        """将指定地块绘制到地图表面"""
        tile = self.tiles[row][col]
        if tile.image and self.map_surface is not None:
            self.map_surface.blit(tile.image, (tile.x, tile.y))

    def _render_all(self) -> None:
        """将所有地块绘制到地图表面"""
        for row_idx, row in enumerate(self.tiles):
            for col_idx, _ in enumerate(row):
                self._render_tile(row_idx, col_idx)

    def _update_obstacles_from_tiles(self) -> None:
        """根据当前地块重新生成障碍物列表"""
        self.unit_obstacles.clear()
        self.bullet_obstacles.clear()
        for row in self.tiles:
            for tile in row:
                if tile.blocks_unit:
                    self.unit_obstacles.append(tile.rect)
                if tile.blocks_bullet:
                    self.bullet_obstacles.append(tile.rect)

    def draw(self, surface: pygame.Surface, camera_offset: Tuple[float, float] = (0, 0)) -> None:
        """绘制地图"""
        for row in self.tiles:
            for tile in row:
                tile.draw(surface, camera_offset)
        if DRAW_OBSTACLE_BOUNDING_BOX or DEBUG_MODE:
            self._draw_debug(surface, camera_offset)

    def update(self, delta_time: float) -> None:
        """更新所有地块（例如生命恢复、动画等）"""
        for row in self.tiles:
            for tile in row:
                tile.update(delta_time)

    def save_to_file(self, file_name: str) -> bool:
        """保存地图到文件（字符串格式）"""
        try:
            file_path = os.path.join(DEFAULT_MAP_PATH, file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                for row in self.tiles:
                    line = ''.join(tile.letter for tile in row)
                    f.write(line + '\n')

            print(f"地图已保存到: {file_path}")
            return True
        except Exception as e:
            print(f"保存地图失败: {e}")
            return False

    def save(self, file_name = None) -> bool:
        """便捷保存方法（自动生成文件名）"""
        if file_name is None:
            save_path = get_next_filename(DEFAULT_MAP_PATH, 'default_map', '.txt')
            return self.save_to_file(save_path)
        return self.save_to_file(file_name)

    @classmethod
    def create_map_from_strings(cls, strings: List[str], tile_size: int = 64) -> 'GameMap':
        """从字符串列表创建地图"""
        return cls(strings, tile_size)

    @classmethod
    def load_from_file(cls, file_name: str, tile_size: int = 64) -> Optional['GameMap']:
        """从文件加载地图"""
        file_path = os.path.join(DEFAULT_MAP_PATH, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                map_data = [line.strip() for line in f if line.strip()]

            if not map_data:
                print(f"文件 {file_path} 为空")
                return None

            # 检查行长度一致性
            first_len = len(map_data[0])
            for i, row in enumerate(map_data):
                if len(row) != first_len:
                    print(f"警告：第 {i} 行长度不一致")

            print(f"地图已从 {file_path} 加载")
            return cls(map_data, tile_size)

        except FileNotFoundError:
            print(f"地图文件不存在: {file_path}")
            return None
        except Exception as e:
            print(f"加载地图失败: {e}")
            return None

    def check_collision(self, rect: pygame.Rect) -> bool:
        """检查矩形是否与任何单位障碍物碰撞"""
        for obstacle in self.unit_obstacles:
            if rect.colliderect(obstacle):
                return True
        return False

    def is_walkable(self, x: float, y: float, width: float = 0, height: float = 0) -> bool:
        """检查区域是否可通行"""
        if width == 0 and height == 0:
            col = int(x // self.tile_size)
            row = int(y // self.tile_size)
            if 0 <= row < self.height and 0 <= col < self.width:
                return not self.tiles[row][col].blocks_unit
            return False
        else:
            rect = pygame.Rect(x, y, width, height)
            return not self.check_collision(rect)

    def get_colliding_obstacles(self, rect: pygame.Rect) -> List[pygame.Rect]:
        """获取与矩形碰撞的所有单位障碍物"""
        colliding = []
        for obstacle in self.unit_obstacles:
            if rect.colliderect(obstacle):
                colliding.append(obstacle)
        return colliding

    def is_bullet_blocked(self, rect: pygame.Rect) -> bool:
        """检查子弹是否被阻挡"""
        for obstacle in self.bullet_obstacles:
            if rect.colliderect(obstacle):
                return True
        return False

    def get_map_size(self) -> Tuple[int, int]:
        """获取地图总尺寸（像素）"""
        return self.width * self.tile_size, self.height * self.tile_size

    def get_tile_at_position(self, x: float, y: float) -> Optional[str]:
        """获取指定位置的地块字符"""
        col = int(x // self.tile_size)
        row = int(y // self.tile_size)
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.tiles[row][col].letter
        return None

    def _draw_debug(self, surface: pygame.Surface, camera_offset: Tuple[float, float]) -> None:
        """绘制调试信息（障碍物边框）"""
        # 单位障碍物（红色边框）
        for obs in self.unit_obstacles:
            screen_rect = obs.move(-camera_offset[0], -camera_offset[1])
            pygame.draw.rect(surface, (255, 0, 0), screen_rect, 2)
        # 子弹障碍物（绿色边框）
        for obs in self.bullet_obstacles:
            screen_rect = obs.move(-camera_offset[0], -camera_offset[1])
            pygame.draw.rect(surface, (0, 255, 0), screen_rect, 1)
    
    def to_strings(self) -> List[str]:
        """返回地图的字符串表示（每行一个字符串）"""
        return [''.join(tile.letter for tile in row) for row in self.tiles]


def create_map_from_strings(map_strings: List[str], tile_size: int = 64) -> GameMap:
    return GameMap.create_map_from_strings(map_strings, tile_size)

def create_map_from_file(file_name: str, tile_size: int = 64) -> Optional[GameMap]:
    return GameMap.load_from_file(file_name, tile_size)

def create_test_map() -> Optional[GameMap]:
    """创建测试地图"""
    return GameMap.load_from_file("test_map.txt")

def create_empty_map(width: int = 15, height: int = 10) -> GameMap:
    map_data = ["o" * width for _ in range(height)]
    return GameMap(map_data)

def create_border_map(width: int = 15, height: int = 10) -> GameMap:
    map_data = []
    for y in range(height):
        if y == 0 or y == height - 1:
            map_data.append("x" * width)
        else:
            map_data.append("x" + "o" * (width - 2) + "x")
    return GameMap(map_data)

def create_random_map(width: int = 15, height: int = 10, density: float = 0.3) -> GameMap:
    map_data = []
    for y in range(height):
        row = []
        for x in range(width):
            is_border = (y == 0 or y == height - 1 or x == 0 or x == width - 1)
            if is_border:
                row.append('x')
            else:
                row.append('x' if random.random() < density else 'o')
        map_data.append(''.join(row))
    return GameMap(map_data)