import pygame
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from game.BattleWorld import BattleWorld
from game.Parameter import Team, FPS, ACC
from game.PCControl import PCControl
from game.utils import Action
from environment.rendering import PygameRenderer


def init_game(map_name="test_map"):
    world = BattleWorld()
    state_names = {
        "test_map": ("test_map_7v7_1", (100, 500)),
        "big_map_test": ("big_map_test_9v9_1", (100, 100)),
    }
    if map_name not in state_names:
        raise ValueError(
            f"Unsupported test map {map_name!r}; "
            "expected 'test_map' or 'big_map_test'"
        )
    state_name, keyboard_spawn = state_names[map_name]
    world.load_game_state(state_name)
    # unit_id=0 is intentionally local to keyboard play and never belongs to a
    # reusable game-state file or a reinforcement-learning episode.
    world.create_unit(
        "tank",
        Team.PLAYER,
        keyboard_spawn,
        unit_id=0,
        using_ai=False,
    )
    return world

if __name__ == "__main__":
    screen_width, screen_height = 960, 640
    renderer = PygameRenderer(screen_width, screen_height, visible=True, title="test")
    clock = pygame.time.Clock()
    
    # 添加地图和单位
    map_name = sys.argv[1] if len(sys.argv) > 1 else "big_map_test"
    game_manager = init_game(map_name)
    
    action = Action()
    
    running = True
    while running:
        delta_time = 1.0 / FPS
        clock.tick(FPS * ACC)
        
        # 应用键盘鼠标控制
        running, action = PCControl(game_manager, action)
        game_manager.set_unit_action(0, action)
        
        # 更新、绘制
        game_manager.update(delta_time)
        renderer.draw(game_manager, mouse_pos=action.mouse_pos)
        renderer.present()
        
        # 每5秒打印一次单位记录
        game_manager.print_record()
    
    renderer.close()
    sys.exit()
