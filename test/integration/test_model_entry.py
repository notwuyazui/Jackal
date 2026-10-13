"""Configuration, policy and presentation checks for the unified single-battle entry."""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
import tempfile
from pathlib import Path
import unittest

from training.test_model import (parse_args, resolve_settings,
    load_policy, STATE_DIR, CONFIG_DIR)
from render.api import apply_map
from environment.presentation.adapter import build_world
from render.api import SnapshotStream


class UnifiedModelTestEntryTests(unittest.TestCase):
    def test_default_runs_level_five_ai_on_border_with_both_views(self):
        args = parse_args([])
        settings = resolve_settings(args)
        self.assertEqual(args.render, '2D3D')
        self.assertEqual(args.ai_intelligence, 5)
        self.assertEqual(settings.config_path, STATE_DIR / 'border_map_3t_vs_3t.json')
        self.assertEqual(settings.max_steps, 6000)
        world = build_world(settings.config_path, ai_intelligence_level=args.ai_intelligence,
                            normal_shell_only=False)
        self.assertEqual(len(world.unit_manager.enemy_ais), 6)
        self.assertTrue(all(ai.intelligence_level == 5 for ai in world.unit_manager.enemy_ais))
        # Attaching AI must retain the externally controlled tank's 150 deg/s turret.
        self.assertEqual(world.get_unit(2).turret_angular_speed, 150)
        apply_map(world)

    def test_config_catalog_is_selected_by_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d) / 'model.pt'
            checkpoint.touch()
            settings = resolve_settings(parse_args(['--checkpoint', str(checkpoint)]))
            self.assertEqual(settings.config_path, CONFIG_DIR / 'jackal_autoaim_3v3_qmix.json')
            self.assertIsNone(settings.env_config['render_mode'])
            self.assertFalse(settings.env_config['use_video'])
            self.assertEqual(settings.max_steps, 700)
            with self.assertRaisesRegex(ValueError, '找不到'):
                resolve_settings(parse_args(['--checkpoint', str(checkpoint), '--config', 'border_map_3t_vs_3t']))
        with self.assertRaisesRegex(ValueError, '找不到'):
            resolve_settings(parse_args(['--config', 'jackal_autoaim_3v3_qmix']))

    def test_other_maps_require_native_2d(self):
        config = 'valley_map_3t2a_vs_3t2a'
        settings = resolve_settings(parse_args(['--config', config]))
        world = build_world(settings.config_path)
        with self.assertRaisesRegex(ValueError, '--render 2D'):
            apply_map(world)
        settings = resolve_settings(parse_args(['--config', config, '--render', '2d']))
        world = build_world(settings.config_path, ai_intelligence_level=5, normal_shell_only=False)
        self.assertEqual(len(world.unit_manager.units), 10)

    def test_generated_border_training_scenario_is_supported_without_rewriting_it(self):
        from environment.jackal_env import JackalEnv
        for count in (3, 5):
            env = JackalEnv(n_agents=count, n_enemies=count)
            try:
                env.reset(seed=42)
                self.assertEqual(len(apply_map(env.world)['initial_units']), count * 2)
            finally:
                env.close()
        for config in ({'n_agents': 3, 'n_enemies': 3, 'ally_unit_types': ['tank', 'tank', 'archie']},):
            env = JackalEnv(**config)
            try:
                env.reset(seed=42)
                with self.assertRaisesRegex(ValueError, '--render 2D'):
                    apply_map(env.world)
            finally:
                env.close()

    def test_cli_overrides_do_not_edit_training_config(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d) / 'model.pt'
            checkpoint.touch()
            path = CONFIG_DIR / 'jackal_autoaim_3v3_qmix.json'
            original = path.read_bytes()
            args = parse_args(['--checkpoint', str(checkpoint), '--max-steps', '6000',
                               '--fixed-delta-time', '.04', '--seed', '7', '--enemy-intelligence', '6'])
            settings = resolve_settings(args)
            self.assertEqual((settings.max_steps, settings.dt, settings.seed), (6000, .04, 7))
            self.assertEqual(settings.env_config['enemy_ai_intelligence_level'], 6)
            self.assertEqual(path.read_bytes(), original)

    def test_allies_and_enemies_can_have_separate_ai_levels(self):
        world = build_world(ai_intelligence_level=7, enemy_intelligence_level=4, normal_shell_only=False)
        levels = {ai.unit.team.name: ai.intelligence_level for ai in world.unit_manager.enemy_ais}
        self.assertEqual(levels, {'PLAYER': 7, 'ENEMY': 4})

    def test_pure_3d_does_not_generate_or_require_an_inset(self):
        world = build_world()
        hub = SnapshotStream(show_inset=False)
        try:
            hub.initialize(world.render_manifest(.03))
            hub.publish(world.render_snapshot())
            self.assertIsNone(hub.renderer)
            self.assertEqual(hub.png, '')
            self.assertFalse(hub.info['show_inset'])
            self.assertEqual(hub.latest['jackal_png_sha256'], '')
            frame = hub.latest
            hub.acknowledge({'view_token': hub.view_token, 'version': frame['version'],
                             'tick': frame['tick'], 'hash': frame['hash']})
            self.assertEqual(hub.acks['3d']['tick'], 0)
        finally:
            hub.close()

    def test_model_weights_load_and_mismatched_config_fails_before_opening_a_view(self):
        import torch
        from environment.jackal_env import JackalEnv
        from training.marl2.controllers.basic_mac import BasicMAC
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d) / 'model.pt'
            checkpoint.touch()
            settings = resolve_settings(parse_args(['--checkpoint', str(checkpoint)]))
            env = JackalEnv(**settings.env_config)
            try:
                info = env.get_env_info()
                mac = BasicMAC(info['n_agents'], info['n_actions'], info['obs_shape'],
                               settings.config['agent'], torch.device('cpu'))
                torch.save({'agent': mac.agent.state_dict()}, checkpoint)
                policy = load_policy(env, settings, 'cpu')
                obs, _ = env.reset(seed=settings.seed)
                actions = policy.select_actions(obs, env.get_avail_actions(), epsilon=0, test_mode=True)
                self.assertTrue(all(env.get_avail_agent_actions(i)[action] for i, action in enumerate(actions)))
                self.assertFalse(policy.agent.training)
                self.assertTrue(all(not unit.usingAI for unit in env.agents))
                settings.config['agent']['rnn_hidden_dim'] = 33
                with self.assertRaisesRegex(ValueError, '输入维度'):
                    load_policy(env, settings, 'cpu')
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
