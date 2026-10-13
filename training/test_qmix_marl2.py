from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import random


import numpy as np
import torch

from training.utils.device import get_device, print_device_info

from training.marl2.controllers.basic_mac import BasicMAC
from training.marl2.modules.mixers.qmix import QMixer
from training.marl2.registry import ENV_REGISTRY, LEARNER_REGISTRY
from training.marl2.runners.episode_runner import EpisodeRunner
from training.marl2.utils.config import load_config, merge_dict


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate QMIX checkpoint on Jackal")
    parser.add_argument("--config", type=str, default="training/configs/marl2/jackal_autoaim_3v3_qmix.json")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--seed-base", type=int, default=None, help="Per-episode seed starts from this value")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    parser.add_argument("--render-mode", choices=["none", "human", "browser", "browser_lockstep", "2D", "3D", "2D3D"], default="none")
    parser.add_argument("--max-steps", type=int, default=None, help="Evaluation horizon; does not change training config")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)

    device = get_device(args.device)
    print_device_info(device)

    env_override = cfg.get("eval_env", cfg.get("train", {}).get("eval_env", {}))
    env_cfg = merge_dict(cfg["env"], env_override) if env_override else dict(cfg["env"])
    env_cfg["headless"] = args.render_mode not in {"human", "2D"}
    env_cfg["render_mode"] = None if args.render_mode == "none" else args.render_mode
    if args.max_steps is not None:
        env_cfg["max_steps"] = args.max_steps
    if args.video_dir:
        env_cfg["use_video"] = True
        env_cfg["video_dir"] = args.video_dir

    env_name = env_cfg.get("name", "jackal")
    env_cls = ENV_REGISTRY[env_name]
    env = env_cls(env_cfg)
    try:
        env_info = env.get_env_info()

        mac = BasicMAC(
            n_agents=env_info["n_agents"],
            n_actions=env_info["n_actions"],
            obs_shape=env_info["obs_shape"],
            agent_cfg=cfg["agent"],
            device=device,
        )
        mixer = QMixer(
            n_agents=env_info["n_agents"],
            state_dim=env_info["state_shape"],
            embed_dim=int(cfg["algo"].get("mixing_embed_dim", 32)),
        ).to(device)

        learner_name = cfg["algo"].get("name", "qmix")
        learner_cls = LEARNER_REGISTRY[learner_name]
        learner = learner_cls(mac, mixer, cfg["algo"], device)
        learner.load_models(args.checkpoint)

        runner = EpisodeRunner(env, mac)
        seed_base = int(args.seed_base) if args.seed_base is not None else int(cfg.get("seed", 42)) + 100000

        returns = []
        wins = 0
        lengths = []

        for ep in range(1, args.episodes + 1):
            ep_seed = seed_base + (ep - 1)
            random.seed(ep_seed)
            np.random.seed(ep_seed % (2**32))
            torch.manual_seed(ep_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(ep_seed)
            elif device.type == "mps":
                torch.mps.manual_seed(ep_seed)

            _, stats = runner.run(
                test_mode=True,
                epsilon=0.0,
                episode_seed=ep_seed,
            )
            returns.append(stats["episode_return"])
            lengths.append(stats["episode_length"])
            wins += int(stats["battle_won"])
            print(
                f"[Test] ep={ep}/{args.episodes} "
                f"ret={stats['episode_return']:.2f} "
                f"len={stats['episode_length']} "
                f"win={int(stats['battle_won'])} "
                f"seed={stats['episode_seed']}"
            )

        n = max(1, len(returns))
        print("[Summary]")
        print(f"  seed_base={seed_base}")
        print(f"  win_rate={wins / n:.3f}")
        print(f"  return_mean={sum(returns) / n:.2f}")
        print(f"  ep_len_mean={sum(lengths) / n:.1f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
