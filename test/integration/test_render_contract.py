import ast
from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import unittest
from environment.presentation.adapter import build_world
from engine.contracts.generate_types import generate

ROOT=Path(__file__).resolve().parents[2]

class RenderContractTests(unittest.TestCase):
    def test_snapshot_is_immutable_and_reading_does_not_change_physics(self):
        world=build_world()
        first=world.render_snapshot()
        self.assertEqual(first,world.render_snapshot())
        with self.assertRaises(FrozenInstanceError): first.units[0].health=0
        self.assertEqual(world.get_unit(first.units[0].id).health,first.units[0].health)
        self.assertEqual(first.units[0].hitbox,tuple(world.get_unit(first.units[0].id).bounding_box))
        world.step(.03)
        self.assertEqual(first.tick,0)
        self.assertEqual(world.render_snapshot().tick,1)

    def test_browser_types_are_generated_from_python_contract(self):
        self.assertEqual((ROOT/'render/render3D/src/generated/engine.ts').read_text(),generate())

    def test_engine_does_not_import_renderers_environment_or_training(self):
        for path in (ROOT/'engine').rglob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                names=([alias.name for alias in node.names] if isinstance(node,ast.Import)
                       else [node.module or ''] if isinstance(node,ast.ImportFrom) else [])
                for name in names:
                    self.assertNotIn(name.split('.')[0],{'render','render2D','render3D','environment','training'},str(path))

    def test_headless_training_imports_no_renderer_and_initializes_no_sdl(self):
        code='''import sys,pygame,json
from environment import JackalEnv
e=JackalEnv(); e.reset(seed=42); e.step([0])
assert not pygame.display.get_init()
assert not any(k.startswith(('render.render2D','render.render3D','environment.presentation')) for k in sys.modules)
assert e.renderer is None
e.close()
print('headless-ok')'''
        result=subprocess.run([sys.executable,'-c',code],cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_resources_are_available_from_unrelated_working_directory(self):
        code='''from engine.battle_world import BattleWorld
from environment.presentation.adapter import build_world
from training.marl2.utils.config import load_config
w=BattleWorld();w.load_builtin_map('test_map');w.load_game_state('border_map_3t_vs_3t')
assert len(w.iter_units())==6
assert load_config('training/configs/marl2/jackal_autoaim_3v3_qmix.json')['env']['game_state_file']=='border_map_3t_vs_3t'
'''
        result=subprocess.run([sys.executable,'-c',code],cwd='/tmp',capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
