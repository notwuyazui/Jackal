import pygame
from game.Map.BaseTile import BaseTile
from game.Map.FlatTile.FlatTile import *
from game.Map.BarrierTile.BarrierTile import *
from game.Map.WaterTile.WaterTile import *
from game.Map.SandTile.SandTile import SandTile
from game.Map.TrapTile.TrapTile import TrapTile
from typing import Callable, Dict, List, Optional, Tuple
import os
from game.Parameter import *
from game.utils import *
import random

CHAR_TO_TILE = {
    'o': FlatTile,
    'x': BarrierTile,
    'w': WaterTile,
    's': SandTile,
    't': TrapTile,
}

class GameMap:
    def __init__(self, map_data=None, tile_size: int = 64):
        self.tile_size = tile_size
        self.tiles = []                      # 二维地块列表
        self.unit_obstacles = []             # 阻挡单位的矩形列表
        self.bullet_obstacles = []           # 阻挡子弹的矩形列表
        # 空间哈希索引：将静态障碍按网格分桶，减少碰撞时的全量遍历。
        self._spatial_cell_size = max(1, tile_size)
        self._unit_obstacle_index = SpatialIndex[pygame.Rect](self._spatial_cell_size)
        self._bullet_obstacle_index = SpatialIndex[pygame.Rect](self._spatial_cell_size)
        self._indexed_unit_obstacle_count = 0
        self._indexed_bullet_obstacle_count = 0
        self.width = 0
        self.height = 0

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
                    self._update_obstacles_from_tiles()
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
                #print(x,y,ch)
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

        self._rebuild_spatial_indices()

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
        self._rebuild_spatial_indices()

    def iter_spatial_cells(self, rect: pygame.Rect):
        """枚举 rect 覆盖到的空间哈希网格坐标。"""
        yield from self._unit_obstacle_index.cells_for_rect(rect)

    def _rebuild_spatial_indices(self) -> None:
        """重建静态障碍空间索引。地图构建后障碍通常不变，只需少量重建。"""
        self._unit_obstacle_index.rebuild(self.unit_obstacles, lambda rect: rect)
        self._bullet_obstacle_index.rebuild(self.bullet_obstacles, lambda rect: rect)
        self._indexed_unit_obstacle_count = len(self.unit_obstacles)
        self._indexed_bullet_obstacle_count = len(self.bullet_obstacles)

    def _ensure_spatial_indices(self) -> None:
        """兼容直接修改障碍列表的调用方。"""
        if (
            len(self.unit_obstacles) != self._indexed_unit_obstacle_count
            or len(self.bullet_obstacles) != self._indexed_bullet_obstacle_count
        ):
            self._rebuild_spatial_indices()

    def get_candidate_unit_obstacles(self, rect: pygame.Rect) -> List[pygame.Rect]:
        """单位碰撞 broad-phase 候选障碍。"""
        self._ensure_spatial_indices()
        return self._unit_obstacle_index.query_rect(rect)

    def get_candidate_bullet_obstacles(self, rect: pygame.Rect) -> List[pygame.Rect]:
        """子弹碰撞 broad-phase 候选障碍。"""
        self._ensure_spatial_indices()
        return self._bullet_obstacle_index.query_rect(rect)

    def get_candidate_line_obstacles(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> List[pygame.Rect]:
        """返回射线穿过网格中的障碍候选。"""
        self._ensure_spatial_indices()
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        line_bounds = pygame.Rect(
            min(x0, x1),
            min(y0, y1),
            abs(x1 - x0) + 1,
            abs(y1 - y0) + 1,
        )
        return self._bullet_obstacle_index.query_rect(line_bounds)

    def has_line_of_sight(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
    ) -> bool:
        """判断两点之间是否被可阻挡子弹的地块遮挡。"""
        return not any(
            obstacle.clipline(start, end)
            for obstacle in self.get_candidate_line_obstacles(start, end)
        )

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

            #print(f"地图已从 {file_path} 加载")
            return cls(map_data, tile_size)

        except FileNotFoundError:
            print(f"地图文件不存在: {file_path}")
            return None
        except Exception as e:
            print(f"加载地图失败: {e}")
            return None

    def check_collision(self, rect: pygame.Rect) -> bool:
        """检查矩形是否与任何单位障碍物碰撞"""
        for obstacle in self.get_candidate_unit_obstacles(rect):
            if rect.colliderect(obstacle):
                return True
        return False

    def contains_rect(self, rect: pygame.Rect) -> bool:
        """检查矩形是否完整位于地图边界内。无尺寸空地图视为无边界。"""

        map_width, map_height = self.get_map_size()
        if map_width <= 0 or map_height <= 0:
            return True
        return (
            rect.left >= 0
            and rect.top >= 0
            and rect.right <= map_width
            and rect.bottom <= map_height
        )

    def can_place_unit(
        self,
        position: Tuple[float, float],
        collision_size: Tuple[float, float],
    ) -> bool:
        """检查单位占用区域是否在边界内且不与不可通行地块重叠。"""

        rect = centered_rect(position, collision_size)
        return self.contains_rect(rect) and not self.check_collision(rect)

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
            return self.contains_rect(rect) and not self.check_collision(rect)

    def get_colliding_obstacles(self, rect: pygame.Rect) -> List[pygame.Rect]:
        """获取与矩形碰撞的所有单位障碍物"""
        colliding = []
        for obstacle in self.get_candidate_unit_obstacles(rect):
            if rect.colliderect(obstacle):
                colliding.append(obstacle)
        return colliding

    def is_bullet_blocked(self, rect: pygame.Rect) -> bool:
        """检查子弹是否被阻挡"""
        for obstacle in self.get_candidate_bullet_obstacles(rect):
            if rect.colliderect(obstacle):
                return True
        return False

    def get_map_size(self) -> Tuple[int, int]:
        """获取地图总尺寸（像素）"""
        return self.width * self.tile_size, self.height * self.tile_size

    def get_tile_char_at_position(self, x: float, y: float) -> Optional[str]:
        """获取指定位置的地块字符"""
        col = int(x // self.tile_size)
        row = int(y // self.tile_size)
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.tiles[row][col].letter
        return None

    def get_tile_at_position(self, x: float, y: float) -> Optional[BaseTile]:
        """获取指定位置的地块对象。"""
        col = int(x // self.tile_size)
        row = int(y // self.tile_size)
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.tiles[row][col]
        return None

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

def create_big_map_test_map() -> Optional[GameMap]:
    """创建大地图键鼠测试地图。"""
    return GameMap.load_from_file("08_big_map_test.txt")

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

def create_maze_map() -> GameMap:
    """15x10 maze-like map used by Jackal experiments."""
    map_data = [
        "xxxxxxxxxxxxxxx",
        "xoooxooooxoooox",
        "xoxoxxxxxoxoxox",
        "xoxoooooooxoxox",
        "xoxxxxxoxxxxxox",
        "xoooooxoxooooox",
        "xoxxxxxoxxxxxox",
        "xoxooooooooxoxx",
        "xooooxooooxooox",
        "xxxxxxxxxxxxxxx",
    ]
    return GameMap(map_data)

def create_valley_map() -> Optional[GameMap]:
    """从 saved/01_valley_map.txt 创建溪谷地图。"""
    return GameMap.load_from_file("01_valley_map.txt")

def create_river_map() -> Optional[GameMap]:
    """从 saved/02_river_map.txt 创建河流地图。"""
    return GameMap.load_from_file("02_river_map.txt")

def create_spindle_map() -> Optional[GameMap]:
    """从 saved/03_spindle_map.txt 创建纺锤形地图。"""
    return GameMap.load_from_file("03_spindle_map.txt")

def create_corridor_map() -> Optional[GameMap]:
    """从 saved/04_corridor_map.txt 创建走廊地图。"""
    return GameMap.load_from_file("04_corridor_map.txt")

def create_dual_corridor_map() -> Optional[GameMap]:
    """从 saved/05_dual_corridor_map.txt 创建双走廊地图。"""
    return GameMap.load_from_file("05_dual_corridor_map.txt")

def create_square_ring_map() -> Optional[GameMap]:
    """从 saved/06_square_ring_map.txt 创建回字形地图。"""
    return GameMap.load_from_file("06_square_ring_map.txt")

def create_four_blocks_map() -> Optional[GameMap]:
    """从 saved/07_four_blocks_map.txt 创建田字形地图。"""
    return GameMap.load_from_file("07_four_blocks_map.txt")

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


def create_builtin_map(name: str) -> GameMap:
    """按稳定名称创建项目内置地图。"""
    factories: Dict[str, Callable[[], Optional[GameMap]]] = {
        "border": create_border_map,
        "empty": create_empty_map,
        "maze": create_maze_map,
        "random": create_random_map,
        "test": create_test_map,
        "big_map_test": create_big_map_test_map,
        "valley": create_valley_map,
        "river": create_river_map,
        "spindle": create_spindle_map,
        "corridor": create_corridor_map,
        "dual_corridor": create_dual_corridor_map,
        "square_ring": create_square_ring_map,
        "four_blocks": create_four_blocks_map,
    }
    normalized_name = str(name).lower().removesuffix("_map")
    try:
        game_map = factories[normalized_name]()
    except KeyError as exc:
        available = ", ".join(sorted(factories))
        raise ValueError(
            f"Unknown built-in map {name!r}; available maps: {available}"
        ) from exc
    if game_map is None:
        raise ValueError(f"Failed to load built-in map: {name}")
    return game_map
