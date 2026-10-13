"""Keyboard battle simulation and presentation behind the environment boundary."""
import time

from engine.parameters import Team, FPS, ACC
from engine.utils import Action
from engine.battle_world import BattleWorld
from render.api import create_viewer


def create_keyboard_world(map_name="test_map"):
    """Compatibility world builder; KeyboardSession is the normal play API."""
    world = BattleWorld()
    state_names = {
        "test_map": ("test_map_3t2a1p_vs_2t3a1p", (100, 500)),
        "big_map_test": ("big_map_test_4t4a1p_vs_4t4a1p", (100, 100)),
    }
    if map_name not in state_names:
        raise ValueError(
            f"Unsupported test map {map_name!r}; "
            "expected 'test_map' or 'big_map_test'"
        )
    state_name, keyboard_spawn = state_names[map_name]
    world.load_game_state(state_name)
    # Keyboard unit 0 never belongs to a saved game or an RL episode.
    world.create_unit("tank", Team.PLAYER, keyboard_spawn, unit_id=0, using_ai=False)
    return world


class KeyboardSession:
    def __init__(self, map_name="test_map", width=960, height=640):
        self._world = create_keyboard_world(map_name)
        self.width, self.height = width, height

    def run(self):
        action = Action()
        with create_viewer(self._world, "2D", width=self.width, height=self.height,
                           dt=1.0 / FPS) as viewer:
            while True:
                time.sleep(1.0 / (FPS * ACC))
                running, action = viewer.keyboard_controls(self._world, action)
                if not running:
                    break
                self._world.set_unit_action(0, action)
                self._world.update(1.0 / FPS)
                viewer.render(self._world.render_snapshot())
                self._world.print_record()
