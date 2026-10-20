"""Watch one battle, driven by a MARL checkpoint or EnemyAI.

Supports direct execution and ``python -m training.run_game_ai``. The environment
facade owns battle simulation and presentation; this entry owns model inference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
from environment.scenario import GAME_STATE_DIR as STATE_DIR, DEFAULT_GAME_STATE as BORDER_STATE
CONFIG_DIR = ROOT / 'training/configs/marl2'
MODEL_CONFIG = 'jackal_autoaim_3v3_qmix'


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('必须大于 0')
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('必须是大于 0 的有限数值')
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='统一模型 / EnemyAI 单局测试（2D、3D 或 2D3D）')
    parser.add_argument('--render', type=str.upper, choices=['2D', '3D', '2D3D'], default='2D3D',
                        help='2D=Jackal 窗口；3D=CoT 无小窗；2D3D=CoT 内嵌 Jackal（默认）')
    parser.add_argument('--config', '--state', help='配置名称或 JSON 路径；EnemyAI 从 engine/scenarios/game_states '
                        '读取，模型从 training/configs/marl2 读取；默认 border 3 对 3')
    parser.add_argument('--scene', help='显式选择 render/render3D/maps/catalog 中的场景 ID；默认按地图布局匹配')
    parser.add_argument('--checkpoint', type=Path, help='train_qmix_marl2 保存的 .pt 文件；不传则使用 EnemyAI')
    parser.add_argument('--device', type=str.lower, default='auto',
                        help='与训练入口一致：auto/cpu/cuda/cuda:0/mps（仅模型推理需要）')
    parser.add_argument('--seed', type=int, help='覆盖随机种子；默认使用训练配置中的 seed，EnemyAI 为 42')
    parser.add_argument('--max-steps', type=positive_int,
                        help='单局步数上限；模型默认沿用训练配置，EnemyAI 默认 6000')
    parser.add_argument('--fixed-delta-time', '--dt', type=positive_float,
                        help='覆盖物理步长；默认训练配置的步长，EnemyAI 为 0.03 秒')
    parser.add_argument('--ai-intelligence', type=int, choices=range(1, 10), default=5,
                        help='未指定模型时己方 EnemyAI 等级，默认 5')
    parser.add_argument('--enemy-intelligence', type=int, choices=range(1, 10),
                        help='覆盖敌方 EnemyAI 等级；默认沿用训练配置，AI 对战默认与己方相同')
    parser.add_argument('--speed', type=positive_float, default=1,
                        help='播放速度，默认 1 倍；只影响墙钟等待，不改变固定物理步长')
    parser.add_argument('--follow-id', type=int, default=2, help='3D 炮塔跟随的单位 ID，默认 2')
    parser.add_argument('--port', type=int, default=0, help='本地数据服务端口，默认自动选择')
    parser.add_argument('--view-port', type=int, default=0, help='3D 浏览器服务端口，默认自动选择')
    parser.add_argument('--exit-on-finish', action='store_true',
                        help='结束后退出；默认保留结果窗口，关闭测试窗口退出（2D 也支持 Esc）')
    parser.add_argument('--no-open-browser', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not all(0 <= port <= 65535 for port in (args.port, args.view_port)):
        parser.error('端口必须在 0 到 65535 之间')
    if args.device not in {'auto', 'cpu', 'cuda', 'mps'} and not (
            args.device.startswith('cuda:') and args.device[5:].isdigit()):
        parser.error('--device 支持 auto/cpu/cuda/cuda:N/mps')
    return args


def resolve_file(value, directory):
    """Resolve names, project-relative paths and absolute paths in the right catalog."""
    path = Path(value).expanduser()
    if not path.suffix:
        path = path.with_suffix('.json')
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, ROOT / path, directory / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_file() and candidate.is_relative_to(directory.resolve()):
            return candidate
    raise ValueError(f'找不到该目录中的配置：{value}；应从 {directory} 选择 JSON 文件')


def merge_config(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        result[key] = (merge_config(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value))
    return result


@dataclass
class Settings:
    config_path: Path
    config: dict
    env_config: dict | None
    seed: int
    max_steps: int
    dt: float
    checkpoint: Path | None


def resolve_settings(args):
    directory = CONFIG_DIR if args.checkpoint else STATE_DIR
    default = MODEL_CONFIG if args.checkpoint else BORDER_STATE
    path = resolve_file(args.config or default, directory)
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('配置必须是 JSON 对象')
    env = None
    checkpoint = None
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise ValueError(f'检查点文件不存在：{checkpoint}')
        if not all(isinstance(config.get(key), dict) for key in ('env', 'agent', 'algo')):
            raise ValueError('模型模式需要包含 env、agent、algo 的 MARL 训练配置')
        env = merge_config(config['env'], config.get('eval_env', config.get('train', {}).get('eval_env', {})))
        if env.pop('name', 'jackal') != 'jackal':
            raise ValueError('统一测试入口目前只支持 Jackal 环境')
        if env.get('game_state_file'):
            env['game_state_file'] = str(resolve_file(env['game_state_file'], STATE_DIR))
        dt = args.fixed_delta_time or float(env.get('fixed_delta_time', .03))
        max_steps = args.max_steps or int(env.get('max_steps', 500))
        seed = args.seed if args.seed is not None else int(config.get('seed', 42))
        env.update(headless=True, render_mode=None, use_video=False, seed=seed,
                   max_steps=max_steps, fixed_delta_time=dt)
        env.pop('render_options', None)
        if args.enemy_intelligence is not None:
            env['enemy_ai_intelligence_level'] = args.enemy_intelligence
    else:
        if config.get('format') != 'jackal-game-state':
            raise ValueError('EnemyAI 模式需要 engine/scenarios/game_states 中的游戏状态配置')
        dt = args.fixed_delta_time or .03
        max_steps = args.max_steps or 6000
        seed = args.seed if args.seed is not None else 42
    if not math.isfinite(dt) or dt <= 0 or max_steps <= 0:
        raise ValueError('配置中的 fixed_delta_time 和 max_steps 必须大于 0')
    return Settings(path, config, env, seed, max_steps, dt, checkpoint)


def load_policy(env, settings, device_name):
    """Execution uses the agent network, not the training-only mixer/optimizer."""
    import torch
    from training.utils.device import get_device, print_device_info
    from training.marl2.controllers.basic_mac import BasicMAC
    device = get_device('cuda' if device_name.startswith('cuda:') else device_name)
    if device_name.startswith('cuda:'):
        device = torch.device(device_name)
        if device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA 设备不存在：{device_name}')
    print_device_info(device)
    torch.manual_seed(settings.seed)
    info = env.get_env_info()
    mac = BasicMAC(info['n_agents'], info['n_actions'], info['obs_shape'], settings.config['agent'], device)
    payload = torch.load(settings.checkpoint, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or 'agent' not in payload:
        raise ValueError('检查点不包含 agent 权重；请使用 train_qmix_marl2.py 保存的模型')
    try:
        mac.agent.load_state_dict(payload['agent'], strict=True)
    except RuntimeError as exc:
        raise ValueError('模型结构或输入维度与 --config 不一致，请选择该模型的训练配置') from exc
    mac.agent.eval()
    mac.init_hidden(batch_size=1)
    return mac


def run_battle(args, settings):
    from environment.evaluation import BattleSession
    controller = (f'模型 {settings.checkpoint.name}' if settings.checkpoint
                  else f'EnemyAI Lv.{args.ai_intelligence}')
    with BattleSession(state_file=settings.config_path, env_config=settings.env_config,
            seed=settings.seed, dt=settings.dt, max_steps=settings.max_steps,
            ai_intelligence=args.ai_intelligence, enemy_intelligence=args.enemy_intelligence,
            render_mode=args.render, controller=controller,
            render_options={'follow_id': args.follow_id, 'scene_id': args.scene,
                            'port': args.port, 'view_port': args.view_port,
                            'open_browser': not args.no_open_browser}) as battle:
        # Validate weights before any view opens. Model inference stays here;
        # observations, actions, battle results and presentation cross the env API.
        mac = load_policy(battle, settings, args.device) if settings.checkpoint else None
        obs, _ = battle.reset(seed=settings.seed)
        print(f'[Test] render={args.render} controller={controller} seed={settings.seed} '
              f'dt={settings.dt} max_steps={settings.max_steps}\n[Config] {settings.config_path}', flush=True)
        result = battle.get_battle_result()
        if not battle.render():
            return None
        while not result['finished']:
            if not battle.is_open():
                return None
            if mac is not None:
                actions = mac.select_actions(obs, battle.get_avail_actions(), epsilon=0, test_mode=True)
            else:
                actions = None
            obs, _, _, _, info = battle.step(actions)
            result = info['battle_result']
            if not battle.render() or not battle.wait_frame(args.speed):
                return None
        report = battle.get_report()
        print('[Result] ' + json.dumps(report, ensure_ascii=False), flush=True)
        if not args.exit_on_finish:
            battle.wait_for_close()
        return report


def main(argv=None):
    args = parse_args(argv)
    try:
        return run_battle(args, resolve_settings(args))
    except KeyboardInterrupt:
        print('\n[Test] 测试已停止', flush=True)
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        raise SystemExit(f'测试启动或运行失败：{exc}') from exc


def cli() -> None:
    """Use a successful exit status when the demonstration returns a report."""
    main()


if __name__ == '__main__':
    main()
