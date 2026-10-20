"""Evaluate DQN, DRQN and MARL checkpoints with one CLI and metric schema."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的有限数值")
    return number


def parse_args(argv=None):
    from training.utils.device import add_device_argument
    parser = argparse.ArgumentParser(description="统一评估 DQN、DRQN 和 QMIX 系列检查点")
    parser.add_argument("--config", help="训练配置 JSON；默认查找检查点旁的运行配置，QMIX 回退到标准 3v3 配置")
    parser.add_argument("--checkpoint", "--model-path", required=True,
                        help="模型检查点；根据权重结构自动选择网络，--model-path 为旧参数别名")
    parser.add_argument("--episodes", type=positive_int, default=10)
    parser.add_argument("--video-dir", default=None, help="指定后录制每局 Jackal 2D 视频")
    parser.add_argument("--seed-base", type=int, default=None, help="逐局种子起点，默认配置 seed + 100000")
    add_device_argument(parser)
    parser.add_argument("--render-mode", choices=["none", "human", "browser", "browser_lockstep", "2D", "3D", "2D3D"], default="none")
    parser.add_argument("--max-steps", type=positive_int, default=None, help="覆盖评估步数上限，不修改训练配置")
    parser.add_argument("--fixed-delta-time", type=positive_float, default=None, help="覆盖评估物理步长")
    parser.add_argument("--output", default=None, help="将汇总和逐局指标另存为 JSON 文件")
    return parser.parse_args(argv)


def run_episode(session, policy, seed: int) -> dict:
    obs, state = session.reset(seed=seed)
    policy.reset()
    episode_return = 0.0
    length = 0
    info = {}
    result = session.get_battle_result()
    while not result["finished"]:
        actions = policy.select_actions(obs, state, session.get_avail_actions())
        reward, _, info, obs, state = session.step(actions)
        episode_return += float(reward)
        length += 1
        result = session.get_battle_result()
    return {"episode_return": episode_return, "episode_length": length,
            "battle_won": result["winner"] == "PLAYER", "winner": result["winner"],
            "reason": result["reason"], "episode_limit": result["reason"] == "timeout",
            "no_kill_timeout": bool(info.get("no_kill_timeout", False)), "episode_seed": seed}


def summarize(episodes: list[dict]) -> dict:
    import numpy as np
    count = len(episodes)
    returns = [episode["episode_return"] for episode in episodes]
    lengths = [episode["episode_length"] for episode in episodes]
    wins = sum(episode["battle_won"] for episode in episodes)
    draws = sum(episode["winner"] == "DRAW" for episode in episodes)
    return {"episodes_completed": count, "win_rate": wins / count if count else None,
            "draw_rate": draws / count if count else None,
            "loss_rate": (count - wins - draws) / count if count else None,
            "return_mean": float(np.mean(returns)) if count else None,
            "return_std": float(np.std(returns)) if count else None,
            "ep_len_mean": float(np.mean(lengths)) if count else None,
            "ep_len_std": float(np.std(lengths)) if count else None,
            "episode_stats": episodes}


def print_summary(report: dict) -> None:
    print("[Summary]")
    print(f"  seed_base={report['seed_base']}")
    print(f"  episodes_completed={report['episodes_completed']}/{report['episodes_requested']}")
    for name, precision in (("win_rate", 3), ("return_mean", 2), ("ep_len_mean", 1),
                            ("draw_rate", 3), ("loss_rate", 3), ("return_std", 2), ("ep_len_std", 1)):
        value = report[name]
        print(f"  {name}=" + ("n/a" if value is None else f"{value:.{precision}f}"))
    if report["interrupted"]:
        print(f"  stopped={report['stop_reason']} (未完成的对局不计入统计)")


def evaluate(args) -> dict:
    import numpy as np
    from environment.evaluation import EvaluationStopped, ModelEvaluationSession
    from training.model_evaluation import (build_env_config, create_policy, load_checkpoint,
                                           load_evaluation_config, resolve_path)
    from training.utils.device import get_device, print_device_info, seed_torch

    if args.episodes <= 0:
        raise ValueError("episodes 必须大于 0")
    # Register optional device backends before deserializing their checkpoints.
    device = get_device(args.device)
    checkpoint = resolve_path(args.checkpoint)
    kind, weights = load_checkpoint(checkpoint)
    config, config_path = load_evaluation_config(args.config, checkpoint, kind)
    if args.output and Path(args.output).expanduser().resolve() in {checkpoint, config_path}:
        raise ValueError("--output 不能覆盖检查点或训练配置，请选择单独的 JSON 报告路径")
    env_config = build_env_config(config, kind, weights, render_mode=args.render_mode,
        video_dir=args.video_dir, max_steps=args.max_steps, fixed_delta_time=args.fixed_delta_time)
    seed_base = int(args.seed_base) if args.seed_base is not None else int(config.get("seed", 42)) + 100000
    print_device_info(device)
    print(f"[Model] type={kind} checkpoint={checkpoint}")
    print(f"[Config] {config_path if config_path is not None else '使用默认单智能体环境；动作空间从权重识别'}")

    episodes = []
    stop_reason = None
    with ModelEvaluationSession(env_config) as session:
        env_info = session.get_env_info()
        policy = create_policy(kind, weights, config, env_info, device)
        del weights
        try:
            for episode in range(1, args.episodes + 1):
                seed = seed_base + episode - 1
                random.seed(seed)
                np.random.seed(seed % (2**32))
                seed_torch(seed, device)
                stats = run_episode(session, policy, seed)
                episodes.append(stats)
                print(f"[Test] ep={episode}/{args.episodes} ret={stats['episode_return']:.2f} "
                      f"len={stats['episode_length']} win={int(stats['battle_won'])} seed={stats['episode_seed']}")
        except EvaluationStopped:
            stop_reason = "window_closed"
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"

    report = {"model_type": kind, "checkpoint": str(checkpoint),
              "config": str(config_path) if config_path is not None else None,
              "seed_base": seed_base, "episodes_requested": args.episodes,
              "render_mode": args.render_mode, "device": str(device),
              "env_config": env_config, "env_info": env_info,
              "interrupted": stop_reason is not None, "stop_reason": stop_reason,
              **summarize(episodes)}
    print_summary(report)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[Report] {output.resolve()}")
    return report


def main(argv=None):
    args = parse_args(argv)
    try:
        return evaluate(args)
    except KeyboardInterrupt:
        print("\n[Test] 评估已停止")
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        raise SystemExit(f"模型评估失败：{exc}") from exc


def cli() -> None:
    """Console scripts must not pass the result dictionary to sys.exit()."""
    main()


if __name__ == "__main__":
    main()
