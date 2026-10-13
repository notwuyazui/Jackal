"""Enforce the training/environment boundary and evaluation result lifecycle."""
import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from environment.evaluation import BattleSession
from environment.jackal_env import JackalEnv
from environment.scenario import GAME_STATE_DIR
from environment.presentation.adapter import build_world
from render.api import RenderSession, ViewerClosed
from training.test_model import parse_args, resolve_settings, run_battle

ROOT = Path(__file__).resolve().parents[2]


class EvaluationSessionTests(unittest.TestCase):
    def test_training_does_not_import_engine_or_render_or_access_the_world(self):
        for path in (ROOT / 'training').rglob('*.py'):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                for name in names:
                    self.assertNotIn(name.split('.')[0], {'engine', 'render'}, str(path))
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, {'world', 'render_snapshot', 'render_manifest'}, str(path))

    def test_model_and_ai_horizons_are_relative_to_restored_tick(self):
        payload = json.loads((GAME_STATE_DIR / 'border_map_3t_vs_3t.json').read_text())
        payload['world'] = {**payload.get('world', {}), 'tick': 300, 'elapsed_time': 9.}
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'restored.json'
            state.write_text(json.dumps(payload))
            for model in (False, True):
                with self.subTest(model=model), BattleSession(state_file=state,
                        env_config={'game_state_file': str(state)} if model else None,
                        max_steps=2, render_mode=None) as battle:
                    battle.reset()
                    self.assertFalse(battle.get_battle_result()['finished'])
                    actions = [0] * battle.get_env_info()['n_agents'] if model else None
                    _, _, _, done, info = battle.step(actions)
                    self.assertFalse(done)
                    _, _, _, done, info = battle.step(actions)
                    self.assertTrue(done)
                    self.assertEqual(info['battle_result'], battle.get_battle_result())
                    report = battle.get_report()
                    self.assertEqual((report['reason'], report['tick']), ('timeout', 302))
                    self.assertAlmostEqual(report['time'], 9.06)
                    with self.assertRaisesRegex(RuntimeError, 'finished'):
                        battle.step(actions)

    def test_public_result_cannot_mutate_a_later_report(self):
        with BattleSession(render_mode=None, max_steps=1) as battle:
            battle.reset()
            result = battle.get_battle_result()
            result['alive']['PLAYER'] = 99
            self.assertEqual(battle.get_report()['alive']['PLAYER'], 3)
            self.assertTrue(battle.render())
            self.assertTrue(battle.is_open())
            self.assertTrue(battle.wait_frame())
            battle.wait_for_close()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            battle.get_report()

    def test_invalid_action_mode_does_not_advance_physics(self):
        with BattleSession(render_mode=None) as battle:
            battle.reset()
            with self.assertRaisesRegex(ValueError, 'does not accept'):
                battle.step([0, 0, 0])
            self.assertEqual(battle.get_report()['tick'], 0)
        with BattleSession(env_config={}, render_mode=None) as battle:
            battle.reset()
            with self.assertRaisesRegex(ValueError, 'requires agent actions'):
                battle.step()
            self.assertEqual(battle.get_report()['tick'], 0)

    def test_model_validation_failure_closes_environment_before_opening_a_view(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'invalid.pt'
            checkpoint.touch()
            args = parse_args(['--checkpoint', str(checkpoint)])
            env = Mock()
            with patch('environment.evaluation.JackalEnv', return_value=env), \
                    patch('training.test_model.load_policy', side_effect=ValueError('bad weights')), \
                    patch('environment.evaluation.create_viewer') as create:
                with self.assertRaisesRegex(ValueError, 'bad weights'):
                    run_battle(args, resolve_settings(args))
                create.assert_not_called()
            env.close.assert_called_once()

    def test_environment_cleanup_continues_if_view_cleanup_fails(self):
        battle = BattleSession(env_config={}, render_mode=None)
        battle.reset()
        view = Mock()
        view.close.side_effect = RuntimeError('view cleanup failed')
        battle._viewer = view
        assert battle._env is not None
        with patch.object(battle._env, 'close', wraps=battle._env.close) as close:
            with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                battle.close()
            close.assert_called_once()
        battle.close()
        view.close.assert_called_once()

    def test_render_handles_closure_once_and_preserves_real_render_errors(self):
        frame = build_world().render_snapshot()
        with RenderSession('rgb_array') as view:
            with patch.object(view, 'render', side_effect=ViewerClosed()), redirect_stdout(io.StringIO()) as output:
                self.assertFalse(view.show_battle(frame, {}))
                self.assertFalse(view.poll_window())
                self.assertEqual(output.getvalue().count('用户关闭测试窗口'), 1)
        with RenderSession('rgb_array') as view:
            with patch.object(view, 'render', side_effect=RuntimeError('WebGL failure')):
                with self.assertRaisesRegex(RuntimeError, 'WebGL failure'):
                    view.show_battle(frame, {})

    def test_base_environment_exposes_result_without_changing_training_info(self):
        env = JackalEnv(max_steps=1)
        try:
            env.reset(seed=42)
            _, _, _, done, info = env.step([0])
            self.assertTrue(done)
            self.assertEqual(env.get_battle_result()['reason'], 'timeout')
            self.assertNotIn('battle_result', info)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
