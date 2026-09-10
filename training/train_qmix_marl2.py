import argparse
from collections import deque
from datetime import datetime
import os
import random
import re
import sys
from typing import Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from training.utils.device import get_device, print_device_info

from training.marl2.components.replay_buffer import EpisodeReplayBuffer
from training.marl2.controllers.basic_mac import BasicMAC
from training.marl2.modules.mixers.dvd import DVDMixer
from training.marl2.modules.mixers.qmix import QMixer
from training.marl2.registry import ENV_REGISTRY, LEARNER_REGISTRY
from training.marl2.runners.episode_runner import EpisodeRunner
from training.marl2.runners.parallel_episode_runner import ParallelEpisodeRunner
from training.marl2.utils.config import ensure_dir, load_config, merge_dict, set_global_seeds


class LinearEpsilonSchedule:
    def __init__(self, start, finish, anneal_time):
        self.start = float(start)
        self.finish = float(finish)
        self.anneal_time = max(1, int(anneal_time))

    def eval(self, t_env):
        frac = min(1.0, float(t_env) / float(self.anneal_time))
        return self.start + frac * (self.finish - self.start)


def parse_args():
    parser = argparse.ArgumentParser(description="Train QMIX in a PyMARL2-style framework on Jackal")
    parser.add_argument("--config", type=str, default="training/configs/marl2/jackal_autoaim_3v3_qmix.json")
    parser.add_argument("--t-max", type=int, default=None, help="Override total environment steps")
    parser.add_argument("--save-dir", type=str, default="artifacts/checkpoints/marl2")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path to resume training")
    parser.add_argument("--resume-t-env", type=int, default=None, help="Optional resumed t_env when checkpoint has no metadata")
    parser.add_argument("--device", type=str, default="auto", help="device: auto/cpu/cuda/mps")
    parser.add_argument("--tensorboard-dir", type=str, default=None, help="Override TensorBoard log directory")
    return parser.parse_args()


def evaluate(
    runner: Any,
    n_episodes: int,
    seed_base: Optional[int] = None,
    device: Optional[torch.device] = None,
):
    returns = []
    wins = 0
    lengths = []
    episode_limits = 0
    no_kill_timeouts = 0

    use_seed = seed_base is not None
    resolved_seed_base = int(seed_base) if seed_base is not None else 0
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        for idx in range(int(n_episodes)):
            if use_seed:
                ep_seed = resolved_seed_base + idx
                random.seed(ep_seed)
                np.random.seed(ep_seed)
                torch.manual_seed(ep_seed)
                if device is not None and device.type == "cuda":
                    torch.cuda.manual_seed_all(ep_seed)
                elif device is not None and device.type == "mps":
                    torch.mps.manual_seed(ep_seed)

            _, stats = runner.run(test_mode=True, epsilon=0.0)
            returns.append(stats["episode_return"])
            lengths.append(stats["episode_length"])
            wins += int(stats["battle_won"])
            episode_limits += int(stats.get("episode_limit", False))
            no_kill_timeouts += int(stats.get("no_kill_timeout", False))
    finally:
        if use_seed:
            random.setstate(py_rng_state)
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)

    n = max(1, len(returns))
    ret_mean = float(sum(returns) / n)
    len_mean = float(sum(lengths) / n)
    ret_std = float(np.std(np.asarray(returns, dtype=np.float32))) if returns else 0.0
    return {
        "return_mean": ret_mean,
        "return_std": ret_std,
        "ep_length_mean": len_mean,
        "battle_won_mean": float(wins / n),
        "episode_limit_mean": float(episode_limits / n),
        "no_kill_timeout_mean": float(no_kill_timeouts / n),
        # Backward-compatible aliases used by existing checkpoint selection code.
        "test_return_mean": ret_mean,
        "test_return_std": ret_std,
        "test_ep_length_mean": len_mean,
        "test_win_rate": float(wins / n),
        "test_episode_limit_rate": float(episode_limits / n),
        "test_no_kill_timeout_rate": float(no_kill_timeouts / n),
    }


def _safe_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_std(values):
    if not values:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float32)))


def _fmt_stat(name, value):
    return f"{name:<26}{float(value):>10.4f}"


def _new_episode_stat_buffer():
    return {
        "episode_return": [],
        "episode_length": [],
        "battle_won": [],
        "episode_limit": [],
        "no_kill_timeout": [],
    }


def _append_episode_stat(buffer, stat):
    buffer["episode_return"].append(float(stat.get("episode_return", 0.0)))
    buffer["episode_length"].append(float(stat.get("episode_length", 0.0)))
    buffer["battle_won"].append(float(int(stat.get("battle_won", False))))
    buffer["episode_limit"].append(float(int(stat.get("episode_limit", False))))
    buffer["no_kill_timeout"].append(float(int(stat.get("no_kill_timeout", False))))


def _clear_episode_stat_buffer(buffer):
    for values in buffer.values():
        values.clear()


def _log_info(message):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[INFO {now}] my_main {message}")


def _sync_eval_mac_params(train_mac, eval_mac):
    eval_mac.agent.load_state_dict(train_mac.agent.state_dict())


def _set_optimizer_lr(learner, lr):
    lr = float(lr)
    for param_group in learner.optimizer.param_groups:
        param_group["lr"] = lr


def _map_slug(map_name):
    name = str(map_name or "border").lower()
    aliases = {
        "spindle_map": "spindle",
        "corridor_map": "corridor",
    }
    return aliases.get(name, name)


def _create_summary_writer(train_cfg, exp_name, args, env_cfg=None):
    use_tb = bool(train_cfg.get("tensorboard", False))
    if not use_tb:
        return None
    if SummaryWriter is None:
        print("[TensorBoard] torch.utils.tensorboard is unavailable; disable TensorBoard logging.")
        return None

    log_dir = args.tensorboard_dir or train_cfg.get("tensorboard_dir", None)
    if log_dir is None:
        map_name = (env_cfg or {}).get("map_name", "border")
        log_dir = os.path.join("artifacts", "tensorboard", "marl2", "by_map", _map_slug(map_name), exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[TensorBoard] logging to {log_dir}")
    return writer


def _build_mixer(algo_cfg, env_info, mac, device):
    mixer_name = str(algo_cfg.get("mixer", "qmix")).lower()
    if mixer_name == "qmix":
        mixer = QMixer(
            n_agents=env_info["n_agents"],
            state_dim=env_info["state_shape"],
            embed_dim=int(algo_cfg.get("mixing_embed_dim", 32)),
        )
    elif mixer_name == "dvd":
        mixer = DVDMixer(
            n_agents=env_info["n_agents"],
            state_dim=env_info["state_shape"],
            rnn_hidden_dim=mac.hidden_dim,
            embed_dim=int(algo_cfg.get("mixing_embed_dim", 64)),
            dvd_heads=int(algo_cfg.get("dvd_heads", 4)),
            gat_embed_dim=int(algo_cfg.get("gat_embed_dim", 32)),
            hypernet_layers=int(algo_cfg.get("hypernet_layers", 1)),
            hypernet_embed=int(algo_cfg.get("hypernet_embed", 64)),
            abs_weights=bool(algo_cfg.get("abs", True)),
        )
    else:
        raise ValueError(f"Unsupported mixer={mixer_name!r}. Expected 'qmix' or 'dvd'.")
    return mixer.to(device)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_global_seeds(seed)

    train_cfg = cfg["train"]
    algo_cfg = cfg["algo"]
    agent_cfg = cfg["agent"]

    t_max = int(args.t_max) if args.t_max is not None else int(train_cfg.get("t_max", 50000))

    device = get_device(args.device)
    print_device_info(device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)

    env_name = cfg["env"].get("name", "jackal")
    env_cls = ENV_REGISTRY[env_name]
    train_env: Any = env_cls(cfg["env"])
    env_info = train_env.get_env_info()
    eval_env_override = cfg.get("eval_env", train_cfg.get("eval_env", {}))
    eval_env_cfg = merge_dict(cfg["env"], eval_env_override) if eval_env_override else cfg["env"]
    eval_env = env_cls(eval_env_cfg)

    mac = BasicMAC(
        n_agents=env_info["n_agents"],
        n_actions=env_info["n_actions"],
        obs_shape=env_info["obs_shape"],
        agent_cfg=agent_cfg,
        device=device,
    )
    eval_mac = BasicMAC(
        n_agents=env_info["n_agents"],
        n_actions=env_info["n_actions"],
        obs_shape=env_info["obs_shape"],
        agent_cfg=agent_cfg,
        device=device,
    )
    mixer = _build_mixer(algo_cfg, env_info, mac, device)

    learner_name = algo_cfg.get("name", "qmix")
    learner_cls = LEARNER_REGISTRY[learner_name]
    learner = learner_cls(mac, mixer, algo_cfg, device)

    buffer = EpisodeReplayBuffer(buffer_size=int(algo_cfg.get("buffer_size", 5000)))
    eval_runner = EpisodeRunner(eval_env, eval_mac)
    _sync_eval_mac_params(mac, eval_mac)

    eps_schedule = LinearEpsilonSchedule(
        start=float(train_cfg.get("epsilon_start", 1.0)),
        finish=float(train_cfg.get("epsilon_finish", 0.05)),
        anneal_time=int(train_cfg.get("epsilon_anneal_time", 20000)),
    )

    exp_name = cfg.get("experiment", {}).get("name", "jackal_qmix")
    save_root = ensure_dir(os.path.join(args.save_dir, exp_name))
    latest_path = os.path.join(save_root, "latest.pt")
    best_path = os.path.join(save_root, "best.pt")
    best_single_path = os.path.join(save_root, "best_single.pt")
    tb_writer = _create_summary_writer(train_cfg, exp_name, args, env_cfg=cfg.get("env", {}))
    if tb_writer is not None:
        tb_writer.add_text("config/path", args.config, 0)
        tb_writer.add_text("experiment/name", exp_name, 0)

    batch_size = int(algo_cfg.get("batch_size", 16))
    updates_per_collect = int(algo_cfg.get("updates_per_collect", 1))
    test_interval = int(train_cfg.get("test_interval", 2000))
    test_nepisode = int(train_cfg.get("test_nepisode", 8))
    eval_extra_nepisode = int(train_cfg.get("eval_extra_nepisode", train_cfg.get("best_eval_nepisode", test_nepisode)))
    test_seed_base = train_cfg.get("test_seed_base", None)
    log_interval = int(train_cfg.get("log_interval", 1000))
    log_interval_episodes = train_cfg.get("log_interval_episodes", None)
    if log_interval_episodes is not None:
        log_interval_episodes = int(log_interval_episodes)
    save_interval = int(train_cfg.get("save_interval", 5000))

    runner_type = str(train_cfg.get("runner", "episode")).lower()
    parallel_envs = int(train_cfg.get("parallel_envs", 1))
    parallel_start_method = str(train_cfg.get("parallel_start_method", "spawn"))

    if runner_type == "parallel" and parallel_envs > 1:
        runner: Any = ParallelEpisodeRunner(
            env_cls=env_cls,
            env_args=cfg["env"],
            mac=mac,
            n_envs=parallel_envs,
            seed=seed,
            start_method=parallel_start_method,
        )
        runner_is_parallel = True
        train_env.close()
        train_env = None
    else:
        if runner_type == "parallel" and parallel_envs <= 1:
            print("[Train] parallel runner requested but parallel_envs<=1, fallback to episode runner.")
        runner = EpisodeRunner(train_env, mac)
        runner_is_parallel = False

    # 早停：在评估胜率稳定达到阈值后停止训练并保存最终模型。
    early_stop_win_rate = train_cfg.get("early_stop_win_rate", None)
    early_stop_consecutive_evals = int(train_cfg.get("early_stop_consecutive_evals", 1))
    min_t_env_before_stop = int(train_cfg.get("min_t_env_before_stop", 0))

    # best checkpoint 保存过滤：使用窗口均值减少单次评估尖峰误导。
    best_model_window = max(1, int(train_cfg.get("best_model_window", 1)))
    best_model_require_full_window = bool(train_cfg.get("best_model_require_full_window", True))
    best_model_max_recent_zero = train_cfg.get("best_model_max_recent_zero", None)
    if best_model_max_recent_zero is not None:
        best_model_max_recent_zero = int(best_model_max_recent_zero)
    best_model_min_recent_win_rate = train_cfg.get("best_model_min_recent_win_rate", None)
    if best_model_min_recent_win_rate is not None:
        best_model_min_recent_win_rate = float(best_model_min_recent_win_rate)
    save_best_single = bool(train_cfg.get("save_best_single", False))
    best_single_min_win_rate = float(train_cfg.get("best_single_min_win_rate", 0.0))
    reset_best_score_on_resume = bool(train_cfg.get("reset_best_score_on_resume", False))

    stabilize_after_window = bool(train_cfg.get("stabilize_after_window", False))
    stabilize_window_win_rate = float(train_cfg.get("stabilize_window_win_rate", 0.75))
    stabilize_min_t_env = int(train_cfg.get("stabilize_min_t_env", 0))
    stabilize_lr = train_cfg.get("stabilize_lr", None)
    if stabilize_lr is not None:
        stabilize_lr = float(stabilize_lr)
    stabilize_updates_per_collect = train_cfg.get("stabilize_updates_per_collect", None)
    if stabilize_updates_per_collect is not None:
        stabilize_updates_per_collect = int(stabilize_updates_per_collect)
    stabilize_single_win_rate = train_cfg.get("stabilize_single_win_rate", None)
    if stabilize_single_win_rate is not None:
        stabilize_single_win_rate = float(stabilize_single_win_rate)
    stabilize_stage2_single_win_rate = train_cfg.get("stabilize_stage2_single_win_rate", None)
    if stabilize_stage2_single_win_rate is not None:
        stabilize_stage2_single_win_rate = float(stabilize_stage2_single_win_rate)
    stabilize_stage2_window_win_rate = train_cfg.get("stabilize_stage2_window_win_rate", None)
    if stabilize_stage2_window_win_rate is not None:
        stabilize_stage2_window_win_rate = float(stabilize_stage2_window_win_rate)
    stabilize_stage2_lr = train_cfg.get("stabilize_stage2_lr", None)
    if stabilize_stage2_lr is not None:
        stabilize_stage2_lr = float(stabilize_stage2_lr)
    stabilize_stage2_updates_per_collect = train_cfg.get("stabilize_stage2_updates_per_collect", None)
    if stabilize_stage2_updates_per_collect is not None:
        stabilize_stage2_updates_per_collect = int(stabilize_stage2_updates_per_collect)
    stabilize_stage3_single_win_rate = train_cfg.get("stabilize_stage3_single_win_rate", None)
    if stabilize_stage3_single_win_rate is not None:
        stabilize_stage3_single_win_rate = float(stabilize_stage3_single_win_rate)
    stabilize_stage3_window_win_rate = train_cfg.get("stabilize_stage3_window_win_rate", None)
    if stabilize_stage3_window_win_rate is not None:
        stabilize_stage3_window_win_rate = float(stabilize_stage3_window_win_rate)
    stabilize_stage3_lr = train_cfg.get("stabilize_stage3_lr", None)
    if stabilize_stage3_lr is not None:
        stabilize_stage3_lr = float(stabilize_stage3_lr)
    stabilize_stage3_updates_per_collect = train_cfg.get("stabilize_stage3_updates_per_collect", None)
    if stabilize_stage3_updates_per_collect is not None:
        stabilize_stage3_updates_per_collect = int(stabilize_stage3_updates_per_collect)

    rollback_on_collapse = bool(train_cfg.get("rollback_on_collapse", False))
    rollback_min_t_env = int(train_cfg.get("rollback_min_t_env", stabilize_min_t_env))
    rollback_drop_from_best = float(train_cfg.get("rollback_drop_from_best", 0.25))
    rollback_min_best_win_rate = float(train_cfg.get("rollback_min_best_win_rate", 0.45))
    rollback_cooldown_evals = max(0, int(train_cfg.get("rollback_cooldown_evals", 3)))
    rollback_max_times_cfg = train_cfg.get("rollback_max_times", None)
    rollback_max_times = None if rollback_max_times_cfg is None else int(rollback_max_times_cfg)
    rollback_clear_buffer = bool(train_cfg.get("rollback_clear_buffer", True))
    rollback_lr = train_cfg.get("rollback_lr", None)
    if rollback_lr is not None:
        rollback_lr = float(rollback_lr)
    rollback_updates_per_collect = train_cfg.get("rollback_updates_per_collect", None)
    if rollback_updates_per_collect is not None:
        rollback_updates_per_collect = int(rollback_updates_per_collect)
    rollback_stabilization_stage = train_cfg.get("rollback_stabilization_stage", None)
    if rollback_stabilization_stage is not None:
        rollback_stabilization_stage = int(rollback_stabilization_stage)

    best_eval_win_hist: deque[float] = deque(maxlen=best_model_window)
    best_eval_ret_hist: deque[float] = deque(maxlen=best_model_window)

    stop_by_win = early_stop_win_rate is not None
    stable_eval_hits = 0
    best_win_rate = -1.0
    best_selector_score = -1.0
    best_single_win_rate = -1.0
    best_single_return = float("-inf")
    stabilization_active = False
    stabilization_stage = 0
    eval_count = 0
    rollback_count = 0
    last_rollback_eval_count = -rollback_cooldown_evals
    rollback_override_updates_per_collect = None

    t_env = 0
    episode = 0
    next_test_t = test_interval
    next_log_t = log_interval
    next_log_episode = log_interval_episodes if log_interval_episodes is not None else None
    next_save_t = save_interval

    recent_train_episode_stats = _new_episode_stat_buffer()
    recent_learner_stats: dict[str, list[float]] = {
        "loss_td": [],
        "q_taken_mean": [],
        "target_mean": [],
        "td_error_abs": [],
        "grad_norm": [],
    }

    last_eval_stats = {
        "battle_won_mean": 0.0,
        "episode_limit_mean": 0.0,
        "no_kill_timeout_mean": 0.0,
        "ep_length_mean": 0.0,
        "return_mean": 0.0,
        "return_std": 0.0,
        "test_win_rate": 0.0,
        "test_episode_limit_rate": 0.0,
        "test_no_kill_timeout_rate": 0.0,
        "test_ep_length_mean": 0.0,
        "test_return_mean": 0.0,
        "test_return_std": 0.0,
    }

    if args.checkpoint:
        payload = learner.load_models(args.checkpoint)
        _sync_eval_mac_params(mac, eval_mac)
        print(f"[Resume] loaded checkpoint from {args.checkpoint}")

        resume_meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        if isinstance(resume_meta, dict) and resume_meta.get("t_env") is not None:
            t_env = int(resume_meta.get("t_env", 0))
            episode = int(resume_meta.get("episode", 0))
            next_test_t = int(resume_meta.get("next_test_t", ((t_env // test_interval) + 1) * test_interval))
            next_log_t = int(resume_meta.get("next_log_t", ((t_env // log_interval) + 1) * log_interval))
            if log_interval_episodes is not None:
                next_log_episode = int(
                    resume_meta.get(
                        "next_log_episode",
                        ((episode // log_interval_episodes) + 1) * log_interval_episodes,
                    )
                )
            next_save_t = int(resume_meta.get("next_save_t", ((t_env // save_interval) + 1) * save_interval))
            best_win_rate = float(resume_meta.get("best_win_rate", best_win_rate))
            best_selector_score = float(resume_meta.get("best_selector_score", best_win_rate))
            best_single_win_rate = float(resume_meta.get("best_single_win_rate", best_single_win_rate))
            best_single_return = float(resume_meta.get("best_single_return", best_single_return))
            stabilization_active = bool(resume_meta.get("stabilization_active", stabilization_active))
            stabilization_stage = int(resume_meta.get("stabilization_stage", 1 if stabilization_active else 0))
            eval_count = int(resume_meta.get("eval_count", eval_count))
            rollback_count = int(resume_meta.get("rollback_count", rollback_count))
            last_rollback_eval_count = int(
                resume_meta.get("last_rollback_eval_count", last_rollback_eval_count)
            )
            rollback_override_updates_per_collect = resume_meta.get(
                "rollback_override_updates_per_collect",
                rollback_override_updates_per_collect,
            )
            if rollback_override_updates_per_collect is not None:
                rollback_override_updates_per_collect = int(rollback_override_updates_per_collect)
            resume_best_seen_win_rate = max(best_win_rate, best_selector_score, best_single_win_rate)
            if (
                stabilize_stage3_single_win_rate is not None
                and resume_best_seen_win_rate >= stabilize_stage3_single_win_rate
            ):
                stabilization_stage = max(stabilization_stage, 3)
            elif (
                stabilize_stage2_single_win_rate is not None
                and resume_best_seen_win_rate >= stabilize_stage2_single_win_rate
            ):
                stabilization_stage = max(stabilization_stage, 2)
            elif (
                stabilize_single_win_rate is not None
                and resume_best_seen_win_rate >= stabilize_single_win_rate
            ):
                stabilization_stage = max(stabilization_stage, 1)
            stabilization_active = stabilization_stage > 0
            if stabilization_stage >= 3 and stabilize_stage3_lr is not None:
                _set_optimizer_lr(learner, stabilize_stage3_lr)
            elif stabilization_stage >= 2 and stabilize_stage2_lr is not None:
                _set_optimizer_lr(learner, stabilize_stage2_lr)
            elif stabilization_active and stabilize_lr is not None:
                _set_optimizer_lr(learner, stabilize_lr)
            stable_eval_hits = int(resume_meta.get("stable_eval_hits", stable_eval_hits))
            print(
                f"[Resume] restored state from checkpoint meta: "
                f"t_env={t_env}, episode={episode}, best_win_rate={best_win_rate:.3f}"
            )
        else:
            inferred_t_env = None
            if args.resume_t_env is not None:
                inferred_t_env = int(args.resume_t_env)
            else:
                match = re.match(r"step_(\d+)\.pt$", os.path.basename(args.checkpoint))
                if match:
                    inferred_t_env = int(match.group(1))

            if inferred_t_env is not None:
                t_env = inferred_t_env
                next_test_t = ((t_env // test_interval) + 1) * test_interval
                next_log_t = ((t_env // log_interval) + 1) * log_interval
                if log_interval_episodes is not None:
                    next_log_episode = ((episode // log_interval_episodes) + 1) * log_interval_episodes
                next_save_t = ((t_env // save_interval) + 1) * save_interval
                print(
                    f"[Resume] no meta found; inferred t_env={t_env} from checkpoint/args. "
                    "episode and best-win statistics restart from zero."
                )
            else:
                print("[Resume] no meta found; t_env restart at 0. Consider using --resume-t-env for continuity.")

        if reset_best_score_on_resume:
            best_win_rate = -1.0
            best_selector_score = -1.0
            best_single_win_rate = -1.0
            best_single_return = float("-inf")
            stabilization_active = False
            stabilization_stage = 0
            print("[Resume] reset best checkpoint score state by config.")

        if t_max <= t_env:
            raise ValueError(
                f"t_max ({t_max}) must be greater than resumed t_env ({t_env}). "
                "Use --t-max to continue training."
            )

    def checkpoint_meta():
        return {
            "t_env": int(t_env),
            "episode": int(episode),
            "next_test_t": int(next_test_t),
            "next_log_t": int(next_log_t),
            "next_log_episode": int(next_log_episode) if next_log_episode is not None else None,
            "next_save_t": int(next_save_t),
            "best_win_rate": float(best_win_rate),
            "best_selector_score": float(best_selector_score),
            "best_single_win_rate": float(best_single_win_rate),
            "best_single_return": float(best_single_return),
            "stabilization_active": bool(stabilization_active),
            "stabilization_stage": int(stabilization_stage),
            "stable_eval_hits": int(stable_eval_hits),
            "eval_count": int(eval_count),
            "rollback_count": int(rollback_count),
            "last_rollback_eval_count": int(last_rollback_eval_count),
            "rollback_override_updates_per_collect": (
                int(rollback_override_updates_per_collect)
                if rollback_override_updates_per_collect is not None
                else None
            ),
        }

    print(f"[Train] device={device} n_agents={env_info['n_agents']} n_enemies={cfg['env'].get('n_enemies')} n_actions={env_info['n_actions']}")
    print(f"[Train] t_max={t_max} episode_limit={env_info['episode_limit']} exp={exp_name}")
    if runner_is_parallel:
        print(f"[Train] runner=parallel parallel_envs={parallel_envs} start_method={parallel_start_method}")
    else:
        print("[Train] runner=episode")
    if log_interval_episodes is not None:
        print(f"[Train] log_interval_episodes={log_interval_episodes}")
    else:
        print(f"[Train] log_interval_steps={log_interval}")
    if stop_by_win:
        print(
            f"[Train] early-stop enabled: win_rate>={float(early_stop_win_rate):.3f} "
            f"for {early_stop_consecutive_evals} evals (after t_env>={min_t_env_before_stop})"
        )
    if (
        best_model_window > 1
        or best_model_max_recent_zero is not None
        or best_model_min_recent_win_rate is not None
    ):
        print(
            "[Train] best-save filter: "
            f"window={best_model_window}, "
            f"require_full_window={best_model_require_full_window}, "
            f"max_recent_zero={best_model_max_recent_zero}, "
            f"min_recent_win_rate={best_model_min_recent_win_rate}"
        )
    if rollback_on_collapse:
        print(
            "[Train] rollback-on-collapse enabled: "
            f"drop_from_best>={rollback_drop_from_best:.3f}, "
            f"min_best={rollback_min_best_win_rate:.3f}, "
            f"cooldown_evals={rollback_cooldown_evals}, "
            f"clear_buffer={rollback_clear_buffer}"
        )

    while t_env < t_max:
        epsilon = eps_schedule.eval(t_env)
        if runner_is_parallel:
            episode_batches, rollout_stats_list = runner.run(test_mode=False, epsilon=epsilon)
            for episode_batch in episode_batches:
                buffer.insert_episode_batch(episode_batch)
            for stat in rollout_stats_list:
                _append_episode_stat(recent_train_episode_stats, stat)

            collected_episodes = len(rollout_stats_list)
            collected_steps = int(sum(stat["episode_length"] for stat in rollout_stats_list))
            episode += collected_episodes
            t_env += collected_steps

            n_stats = max(1, len(rollout_stats_list))
            rollout_stats: dict[str, Any] = {
                "episode_return": float(sum(stat["episode_return"] for stat in rollout_stats_list) / n_stats),
                "episode_length": float(sum(stat["episode_length"] for stat in rollout_stats_list) / n_stats),
                "battle_won": float(sum(int(stat["battle_won"]) for stat in rollout_stats_list) / n_stats),
                "episode_limit": float(sum(int(stat.get("episode_limit", False)) for stat in rollout_stats_list) / n_stats),
                "no_kill_timeout": float(sum(int(stat.get("no_kill_timeout", False)) for stat in rollout_stats_list) / n_stats),
                "n_collected": collected_episodes,
            }
        else:
            episode_batch, episode_stats = runner.run(test_mode=False, epsilon=epsilon)
            rollout_stats = episode_stats
            buffer.insert_episode_batch(episode_batch)
            _append_episode_stat(recent_train_episode_stats, rollout_stats)

            episode += 1
            t_env += int(rollout_stats["episode_length"])

        if tb_writer is not None:
            tb_writer.add_scalar("rollout/episode_return", float(rollout_stats["episode_return"]), t_env)
            tb_writer.add_scalar("rollout/episode_length", float(rollout_stats["episode_length"]), t_env)
            tb_writer.add_scalar("rollout/battle_won", float(rollout_stats["battle_won"]), t_env)
            tb_writer.add_scalar("rollout/epsilon", float(epsilon), t_env)
            tb_writer.add_scalar("buffer/episodes", float(len(buffer)), t_env)
            if "episode_limit" in rollout_stats:
                tb_writer.add_scalar("rollout/episode_limit", float(rollout_stats["episode_limit"]), t_env)
            if "no_kill_timeout" in rollout_stats:
                tb_writer.add_scalar("rollout/no_kill_timeout", float(rollout_stats["no_kill_timeout"]), t_env)

        train_stats = None
        if buffer.can_sample(batch_size):
            if rollback_override_updates_per_collect is not None:
                current_updates_per_collect = rollback_override_updates_per_collect
            elif stabilization_stage >= 3 and stabilize_stage3_updates_per_collect is not None:
                current_updates_per_collect = stabilize_stage3_updates_per_collect
            elif stabilization_stage >= 2 and stabilize_stage2_updates_per_collect is not None:
                current_updates_per_collect = stabilize_stage2_updates_per_collect
            elif stabilization_stage >= 1 and stabilize_updates_per_collect is not None:
                current_updates_per_collect = stabilize_updates_per_collect
            else:
                current_updates_per_collect = updates_per_collect
            for _ in range(current_updates_per_collect):
                sampled = buffer.sample(batch_size)
                train_stats = learner.train(sampled)
                recent_learner_stats["loss_td"].append(float(train_stats.get("loss_td", train_stats.get("loss", 0.0))))
                recent_learner_stats["q_taken_mean"].append(float(train_stats.get("q_taken_mean", train_stats.get("q_tot_mean", 0.0))))
                recent_learner_stats["target_mean"].append(float(train_stats.get("target_mean", 0.0)))
                recent_learner_stats["td_error_abs"].append(float(train_stats.get("td_error_abs", 0.0)))
                recent_learner_stats["grad_norm"].append(float(train_stats.get("grad_norm", 0.0)))
                if tb_writer is not None:
                    for key, value in train_stats.items():
                        tb_writer.add_scalar(f"learner/{key}", float(value), t_env)
                    tb_writer.add_scalar("train/updates_per_collect", float(current_updates_per_collect), t_env)
                    tb_writer.add_scalar("train/stabilization_active", float(stabilization_active), t_env)
                    tb_writer.add_scalar("train/stabilization_stage", float(stabilization_stage), t_env)
                    if learner.optimizer.param_groups:
                        tb_writer.add_scalar("train/lr", float(learner.optimizer.param_groups[0]["lr"]), t_env)

        if t_env >= next_test_t:
            _sync_eval_mac_params(mac, eval_mac)
            eval_stats = evaluate(eval_runner, test_nepisode, seed_base=test_seed_base, device=device)
            eval_count += 1
            last_eval_stats = eval_stats
            print(
                f"[Eval] t_env={t_env} "
                f"test_battle_won_mean={eval_stats['battle_won_mean']:.3f} "
                f"test_return_mean={eval_stats['return_mean']:.2f} "
                f"test_ep_length_mean={eval_stats['ep_length_mean']:.1f} "
                f"test_episode_limit_mean={eval_stats['episode_limit_mean']:.3f} "
                f"test_no_kill_timeout_mean={eval_stats['no_kill_timeout_mean']:.3f}"
            )
            if tb_writer is not None:
                for key, value in eval_stats.items():
                    tb_writer.add_scalar(f"eval/{key}", float(value), t_env)

            eval_extra_stats = eval_stats
            if eval_extra_nepisode != test_nepisode:
                best_seed_base = None if test_seed_base is None else int(test_seed_base) + 1000000
                eval_extra_stats = evaluate(eval_runner, eval_extra_nepisode, seed_base=best_seed_base, device=device)
                print(
                    f"[Eval-Extra] t_env={t_env} "
                    f"test_battle_won_mean={eval_extra_stats['battle_won_mean']:.3f} "
                    f"n_ep={eval_extra_nepisode}"
                )
                if tb_writer is not None:
                    for key, value in eval_extra_stats.items():
                        tb_writer.add_scalar(f"eval_extra/{key}", float(value), t_env)

            best_eval_win_hist.append(float(eval_extra_stats["test_win_rate"]))
            best_eval_ret_hist.append(float(eval_extra_stats["test_return_mean"]))

            if save_best_single:
                single_win_rate = float(eval_extra_stats["test_win_rate"])
                single_return = float(eval_extra_stats["test_return_mean"])
                single_improved = (
                    single_win_rate > best_single_win_rate
                    or (
                        single_win_rate == best_single_win_rate
                        and single_return > best_single_return
                    )
                )
                if single_win_rate >= best_single_min_win_rate and single_improved:
                    best_single_win_rate = single_win_rate
                    best_single_return = single_return
                    learner.save_models(best_single_path, meta=checkpoint_meta())
                    print(
                        f"[Save] single best checkpoint updated: {best_single_path} "
                        f"(win_rate={best_single_win_rate:.3f}, return={best_single_return:.2f})"
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("eval_best_single/win_rate", best_single_win_rate, t_env)
                        tb_writer.add_scalar("eval_best_single/return_mean", best_single_return, t_env)

            window_n = len(best_eval_win_hist)
            window_mean_wr = float(sum(best_eval_win_hist) / max(1, window_n))
            window_mean_ret = float(sum(best_eval_ret_hist) / max(1, window_n))
            window_zero_cnt = int(sum(1 for wr in best_eval_win_hist if wr <= 0.0))

            window_ready = window_n >= best_model_window if best_model_require_full_window else window_n > 0
            window_pass = bool(window_ready)
            fail_reasons = []
            if not window_ready:
                fail_reasons.append("window_not_full")
            if best_model_max_recent_zero is not None and window_zero_cnt > best_model_max_recent_zero:
                window_pass = False
                fail_reasons.append(f"zero>{best_model_max_recent_zero}")
            if best_model_min_recent_win_rate is not None and window_mean_wr < best_model_min_recent_win_rate:
                window_pass = False
                fail_reasons.append(f"mean_wr<{best_model_min_recent_win_rate:.3f}")

            if (
                best_model_window > 1
                or best_model_max_recent_zero is not None
                or best_model_min_recent_win_rate is not None
            ):
                status = "pass" if window_pass else "hold"
                reason = "ok" if not fail_reasons else ",".join(fail_reasons)
                print(
                    f"[Eval-Window] t_env={t_env} "
                    f"n={window_n}/{best_model_window} "
                    f"mean_wr={window_mean_wr:.3f} "
                    f"mean_ret={window_mean_ret:.2f} "
                    f"zero={window_zero_cnt} "
                    f"status={status} reason={reason}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("eval_window/mean_win_rate", window_mean_wr, t_env)
                    tb_writer.add_scalar("eval_window/mean_return", window_mean_ret, t_env)
                    tb_writer.add_scalar("eval_window/zero_count", window_zero_cnt, t_env)

            selector_score = window_mean_wr

            # 保存当前最佳模型（窗口过滤通过时）
            if window_pass and selector_score > best_selector_score:
                best_selector_score = selector_score
                best_win_rate = selector_score
                learner.save_models(best_path, meta=checkpoint_meta())
                print(
                    f"[Save] best checkpoint updated: {best_path} "
                    f"(window_mean_win_rate={best_selector_score:.3f})"
                )

            eval_extra_win_rate = float(eval_extra_stats["test_win_rate"])
            stabilize_stage1_window_hit = bool(window_ready and window_mean_wr >= stabilize_window_win_rate)
            stabilize_stage1_single_hit = bool(
                stabilize_single_win_rate is not None
                and eval_extra_win_rate >= stabilize_single_win_rate
            )
            stabilize_stage2_window_hit = bool(
                stabilize_stage2_window_win_rate is not None
                and window_ready
                and window_mean_wr >= stabilize_stage2_window_win_rate
            )
            stabilize_stage2_single_hit = bool(
                stabilize_stage2_single_win_rate is not None
                and eval_extra_win_rate >= stabilize_stage2_single_win_rate
            )
            stabilize_stage3_window_hit = bool(
                stabilize_stage3_window_win_rate is not None
                and window_ready
                and window_mean_wr >= stabilize_stage3_window_win_rate
            )
            stabilize_stage3_single_hit = bool(
                stabilize_stage3_single_win_rate is not None
                and eval_extra_win_rate >= stabilize_stage3_single_win_rate
            )
            if (
                stabilize_after_window
                and stabilization_stage < 3
                and t_env >= stabilize_min_t_env
                and (stabilize_stage3_window_hit or stabilize_stage3_single_hit)
            ):
                stabilization_active = True
                stabilization_stage = 3
                stabilize_reason = (
                    f"single_win_rate={eval_extra_win_rate:.3f}"
                    if stabilize_stage3_single_hit
                    else f"window_mean_win_rate={window_mean_wr:.3f}"
                )
                if stabilize_stage3_lr is not None:
                    _set_optimizer_lr(learner, stabilize_stage3_lr)
                print(
                    f"[Stabilize] stage=3 activated at t_env={t_env}: "
                    f"{stabilize_reason}, "
                    f"lr={learner.optimizer.param_groups[0]['lr'] if learner.optimizer.param_groups else 'n/a'}, "
                    f"updates_per_collect="
                    f"{stabilize_stage3_updates_per_collect if stabilize_stage3_updates_per_collect is not None else updates_per_collect}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/stabilization_active", 1.0, t_env)
                    tb_writer.add_scalar("train/stabilization_stage", float(stabilization_stage), t_env)
                    tb_writer.add_scalar(
                        "train/stabilization_trigger_win_rate",
                        eval_extra_win_rate,
                        t_env,
                    )
                    if learner.optimizer.param_groups:
                        tb_writer.add_scalar("train/lr", float(learner.optimizer.param_groups[0]["lr"]), t_env)
                    tb_writer.add_scalar(
                        "train/updates_per_collect",
                        float(
                            stabilize_stage3_updates_per_collect
                            if stabilize_stage3_updates_per_collect is not None
                            else updates_per_collect
                        ),
                        t_env,
                    )
            elif (
                stabilize_after_window
                and stabilization_stage < 2
                and t_env >= stabilize_min_t_env
                and (stabilize_stage2_window_hit or stabilize_stage2_single_hit)
            ):
                stabilization_active = True
                stabilization_stage = 2
                stabilize_reason = (
                    f"single_win_rate={eval_extra_win_rate:.3f}"
                    if stabilize_stage2_single_hit
                    else f"window_mean_win_rate={window_mean_wr:.3f}"
                )
                if stabilize_stage2_lr is not None:
                    _set_optimizer_lr(learner, stabilize_stage2_lr)
                print(
                    f"[Stabilize] stage=2 activated at t_env={t_env}: "
                    f"{stabilize_reason}, "
                    f"lr={learner.optimizer.param_groups[0]['lr'] if learner.optimizer.param_groups else 'n/a'}, "
                    f"updates_per_collect="
                    f"{stabilize_stage2_updates_per_collect if stabilize_stage2_updates_per_collect is not None else updates_per_collect}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/stabilization_active", 1.0, t_env)
                    tb_writer.add_scalar("train/stabilization_stage", float(stabilization_stage), t_env)
                    tb_writer.add_scalar(
                        "train/stabilization_trigger_win_rate",
                        eval_extra_win_rate,
                        t_env,
                    )
                    if learner.optimizer.param_groups:
                        tb_writer.add_scalar("train/lr", float(learner.optimizer.param_groups[0]["lr"]), t_env)
                    tb_writer.add_scalar(
                        "train/updates_per_collect",
                        float(
                            stabilize_stage2_updates_per_collect
                            if stabilize_stage2_updates_per_collect is not None
                            else updates_per_collect
                        ),
                        t_env,
                    )
            elif (
                stabilize_after_window
                and stabilization_stage < 1
                and t_env >= stabilize_min_t_env
                and (stabilize_stage1_window_hit or stabilize_stage1_single_hit)
            ):
                stabilization_active = True
                stabilization_stage = 1
                stabilize_reason = (
                    f"single_win_rate={eval_extra_win_rate:.3f}"
                    if stabilize_stage1_single_hit
                    else f"window_mean_win_rate={window_mean_wr:.3f}"
                )
                if stabilize_lr is not None:
                    _set_optimizer_lr(learner, stabilize_lr)
                print(
                    f"[Stabilize] stage=1 activated at t_env={t_env}: "
                    f"{stabilize_reason}, "
                    f"lr={learner.optimizer.param_groups[0]['lr'] if learner.optimizer.param_groups else 'n/a'}, "
                    f"updates_per_collect="
                    f"{stabilize_updates_per_collect if stabilize_updates_per_collect is not None else updates_per_collect}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/stabilization_active", 1.0, t_env)
                    tb_writer.add_scalar("train/stabilization_stage", float(stabilization_stage), t_env)
                    tb_writer.add_scalar(
                        "train/stabilization_trigger_win_rate",
                        eval_extra_win_rate,
                        t_env,
                    )
                    if learner.optimizer.param_groups:
                        tb_writer.add_scalar("train/lr", float(learner.optimizer.param_groups[0]["lr"]), t_env)
                    tb_writer.add_scalar(
                        "train/updates_per_collect",
                        float(stabilize_updates_per_collect if stabilize_updates_per_collect is not None else updates_per_collect),
                        t_env,
                    )

            rollback_drop = best_selector_score - window_mean_wr
            rollback_cooldown_ready = (eval_count - last_rollback_eval_count) >= rollback_cooldown_evals
            rollback_times_ready = rollback_max_times is None or rollback_count < rollback_max_times
            rollback_ready = (
                rollback_on_collapse
                and t_env >= rollback_min_t_env
                and window_ready
                and os.path.exists(best_path)
                and best_selector_score >= rollback_min_best_win_rate
                and rollback_drop >= rollback_drop_from_best
                and rollback_cooldown_ready
                and rollback_times_ready
            )
            if rollback_ready:
                learner.load_models(best_path)
                _sync_eval_mac_params(mac, eval_mac)
                rollback_count += 1
                last_rollback_eval_count = eval_count
                if rollback_clear_buffer:
                    buffer.clear()
                if rollback_stabilization_stage is not None:
                    stabilization_stage = rollback_stabilization_stage
                    stabilization_active = stabilization_stage > 0
                if rollback_lr is not None:
                    _set_optimizer_lr(learner, rollback_lr)
                if rollback_updates_per_collect is not None:
                    rollback_override_updates_per_collect = rollback_updates_per_collect
                learner.save_models(latest_path, meta=checkpoint_meta())
                print(
                    f"[Rollback] collapse detected at t_env={t_env}: "
                    f"window_mean_win_rate={window_mean_wr:.3f}, "
                    f"best_window={best_selector_score:.3f}, "
                    f"drop={rollback_drop:.3f}. "
                    f"Loaded {best_path}, "
                    f"clear_buffer={rollback_clear_buffer}, "
                    f"lr={learner.optimizer.param_groups[0]['lr'] if learner.optimizer.param_groups else 'n/a'}, "
                    f"updates_per_collect="
                    f"{rollback_override_updates_per_collect if rollback_override_updates_per_collect is not None else 'stage/default'}"
                )
                if tb_writer is not None:
                    tb_writer.add_scalar("train/rollback_count", float(rollback_count), t_env)
                    tb_writer.add_scalar("train/rollback_drop", float(rollback_drop), t_env)
                    tb_writer.add_scalar("buffer/episodes", float(len(buffer)), t_env)
                    if learner.optimizer.param_groups:
                        tb_writer.add_scalar("train/lr", float(learner.optimizer.param_groups[0]["lr"]), t_env)

            # 稳定胜率早停逻辑
            if stop_by_win and t_env >= min_t_env_before_stop:
                if eval_stats["test_win_rate"] >= float(early_stop_win_rate):
                    stable_eval_hits += 1
                else:
                    stable_eval_hits = 0

                print(
                    f"[Eval] stable-win progress: {stable_eval_hits}/{early_stop_consecutive_evals} "
                    f"(threshold={float(early_stop_win_rate):.3f})"
                )

                if stable_eval_hits >= early_stop_consecutive_evals:
                    learner.save_models(latest_path, meta=checkpoint_meta())
                    print(
                        f"[Stop] early stop triggered at t_env={t_env}. "
                        f"Stable win rate reached threshold {float(early_stop_win_rate):.3f}."
                    )
                    break

            next_test_t += test_interval

        should_log = (
            episode >= next_log_episode
            if next_log_episode is not None
            else t_env >= next_log_t
        )
        if should_log:
            train_episode_count = len(recent_train_episode_stats["episode_return"])
            battle_won_mean = _safe_mean(recent_train_episode_stats["battle_won"])
            ep_length_mean = _safe_mean(recent_train_episode_stats["episode_length"])
            return_mean = _safe_mean(recent_train_episode_stats["episode_return"])
            return_std = _safe_std(recent_train_episode_stats["episode_return"])
            episode_limit_mean = _safe_mean(recent_train_episode_stats["episode_limit"])
            no_kill_timeout_mean = _safe_mean(recent_train_episode_stats["no_kill_timeout"])

            loss_td = _safe_mean(recent_learner_stats["loss_td"])
            q_taken_mean = _safe_mean(recent_learner_stats["q_taken_mean"])
            target_mean = _safe_mean(recent_learner_stats["target_mean"])
            td_error_abs = _safe_mean(recent_learner_stats["td_error_abs"])
            grad_norm = _safe_mean(recent_learner_stats["grad_norm"])

            _log_info(f"Recent Stats | t_env:{t_env:10d} | Episode:{episode:10d}")
            print(
                f"{_fmt_stat('battle_won_mean:', battle_won_mean)} "
                f"{_fmt_stat('ep_length_mean:', ep_length_mean)} "
                f"{_fmt_stat('epsilon:', epsilon)} "
                f"{_fmt_stat('train_episodes:', train_episode_count)}"
            )
            print(
                f"{_fmt_stat('return_mean:', return_mean)} "
                f"{_fmt_stat('return_std:', return_std)} "
                f"{_fmt_stat('episode_limit_mean:', episode_limit_mean)} "
                f"{_fmt_stat('no_kill_timeout_mean:', no_kill_timeout_mean)}"
            )
            print(
                f"{_fmt_stat('loss_td:', loss_td)} "
                f"{_fmt_stat('q_taken_mean:', q_taken_mean)} "
                f"{_fmt_stat('target_mean:', target_mean)} "
                f"{_fmt_stat('td_error_abs:', td_error_abs)} "
                f"{_fmt_stat('grad_norm:', grad_norm)}"
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train_stats/battle_won_mean", battle_won_mean, t_env)
                tb_writer.add_scalar("train_stats/ep_length_mean", ep_length_mean, t_env)
                tb_writer.add_scalar("train_stats/return_mean", return_mean, t_env)
                tb_writer.add_scalar("train_stats/return_std", return_std, t_env)
                tb_writer.add_scalar("train_stats/episode_limit_mean", episode_limit_mean, t_env)
                tb_writer.add_scalar("train_stats/no_kill_timeout_mean", no_kill_timeout_mean, t_env)
                tb_writer.add_scalar("train_stats/n_episodes", float(train_episode_count), t_env)

            _clear_episode_stat_buffer(recent_train_episode_stats)
            for key in recent_learner_stats:
                recent_learner_stats[key].clear()

            if next_log_episode is not None:
                assert log_interval_episodes is not None
                while episode >= next_log_episode:
                    next_log_episode += log_interval_episodes
            else:
                next_log_t += log_interval

        if t_env >= next_save_t:
            step_path = os.path.join(save_root, f"step_{t_env}.pt")
            meta = checkpoint_meta()
            learner.save_models(step_path, meta=meta)
            learner.save_models(latest_path, meta=meta)
            print(f"[Save] {step_path}")
            next_save_t += save_interval

    learner.save_models(latest_path, meta=checkpoint_meta())
    print(f"[Done] training finished. latest checkpoint: {latest_path}")

    if runner_is_parallel:
        runner.close()
    if train_env is not None:
        train_env.close()
    eval_env.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
