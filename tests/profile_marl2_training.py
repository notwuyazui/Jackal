"""Profile one representative MARL2 collect-and-learn cycle.

This is an executable diagnostic script, not a unit test.  It intentionally
runs the configured environment in-process so timings inside BattleWorld and
EnemyAI remain visible; a profiler attached only to ParallelEpisodeRunner's
parent process would mostly report pipe waiting.

Example:
    python tests/profile_marl2_training.py \
      --config training/configs/marl2/jackal_autoaim_5v5_etdqmix_hetero_valley_mapfeat7_10m_v2.json \
      --device auto --steps 300 --batch-size 4

For the closest match to the production update, use the configured episode
length and batch size (this can take minutes and use substantial GPU memory):
    python tests/profile_marl2_training.py --steps 2200 --batch-size 16
"""

from __future__ import annotations

import argparse
import cProfile
from dataclasses import dataclass
import functools
import io
import json
import os
import pstats
import sys
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from environment.action_controller import ActionController
from environment.observation.manager import ObservationManager
from environment.reward.manager import RewardManager
from game.BattleWorld import BattleWorld
from game.Bullet.BulletManager import BulletManager
from game.Map.GameMap import GameMap
from game.Unit.BaseUnit import BaseUnit
from game.Unit.EnemyAI import EnemyAI
from game.Unit.UnitManager import UnitManager
from training.marl2.components.replay_buffer import EpisodeReplayBuffer
from training.marl2.controllers.basic_mac import BasicMAC
from training.marl2.registry import ENV_REGISTRY, LEARNER_REGISTRY
from training.marl2.utils.config import load_config, set_global_seeds
from training.train_qmix_marl2 import _build_mixer
from training.utils.device import get_device, print_device_info


DEFAULT_CONFIG = (
    "training/configs/marl2/"
    "jackal_autoaim_5v5_etdqmix_hetero_valley_mapfeat7_10m_v2.json"
)


@dataclass
class TimingStat:
    seconds: float = 0.0
    calls: int = 0


class MethodTimings:
    """Temporarily time selected methods without editing production code."""

    def __init__(self) -> None:
        self.stats: dict[str, TimingStat] = {}
        self._originals: list[tuple[type, str, Callable[..., Any]]] = []

    def add(self, label: str, seconds: float) -> None:
        stat = self.stats.setdefault(label, TimingStat())
        stat.seconds += float(seconds)
        stat.calls += 1

    def install(self, owner: type, method_name: str, label: str) -> None:
        original = getattr(owner, method_name)
        timings = self

        @functools.wraps(original)
        def timed(*args, **kwargs):
            started = perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timings.add(label, perf_counter() - started)

        self._originals.append((owner, method_name, original))
        setattr(owner, method_name, timed)

    def clear(self) -> None:
        self.stats.clear()

    def close(self) -> None:
        for owner, method_name, original in reversed(self._originals):
            setattr(owner, method_name, original)
        self._originals.clear()

    def value(self, label: str) -> TimingStat:
        return self.stats.get(label, TimingStat())


def _install_environment_timers(timings: MethodTimings) -> None:
    methods = (
        (ActionController, "apply", "env.action_apply"),
        (ActionController, "available_actions", "env.available_actions_one_agent"),
        (BattleWorld, "step", "world.step"),
        (GameMap, "update", "world.map_update"),
        (UnitManager, "update", "world.unit_manager_update"),
        (EnemyAI, "update", "world.enemy_ai_update"),
        (BaseUnit, "update", "world.base_unit_update"),
        (BulletManager, "update", "world.bullet_manager_update"),
        (BattleWorld, "refresh_vision", "world.refresh_vision"),
        (BattleWorld, "snapshot", "env.snapshot"),
        (RewardManager, "calculate", "env.reward_calculate"),
        (ObservationManager, "get_observations", "env.observations"),
        (ObservationManager, "get_state", "env.global_state"),
        (UnitManager, "record_damage", "record.damage_event"),
        (UnitManager, "record_unit_collision", "record.unit_collision"),
        (BattleWorld, "print_record", "record.print_record"),
        (BaseUnit, "get_record", "record.get_record"),
    )
    for owner, name, label in methods:
        timings.install(owner, name, label)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _new_episode_batch(env_info: dict[str, Any]) -> dict[str, np.ndarray]:
    episode_limit = int(env_info["episode_limit"])
    n_agents = int(env_info["n_agents"])
    n_actions = int(env_info["n_actions"])
    obs_shape = int(env_info["obs_shape"])
    state_shape = int(env_info["state_shape"])
    return {
        "obs": np.zeros(
            (episode_limit + 1, n_agents, obs_shape),
            dtype=np.float32,
        ),
        "state": np.zeros((episode_limit + 1, state_shape), dtype=np.float32),
        "avail_actions": np.zeros(
            (episode_limit + 1, n_agents, n_actions),
            dtype=np.float32,
        ),
        "actions": np.zeros((episode_limit, n_agents, 1), dtype=np.int64),
        "reward": np.zeros((episode_limit, 1), dtype=np.float32),
        "terminated": np.zeros((episode_limit, 1), dtype=np.float32),
        "filled": np.zeros((episode_limit, 1), dtype=np.float32),
    }


def _warm_up(env, mac: BasicMAC, device: torch.device, steps: int) -> None:
    if steps <= 0:
        return
    obs, _ = env.reset()
    mac.init_hidden(batch_size=1)
    for _ in range(int(steps)):
        available = env.get_avail_actions()
        actions = mac.select_actions(
            obs,
            available,
            epsilon=0.0,
            test_mode=True,
        )
        _, done, _, obs, _ = env.step(actions)
        if done:
            break
    _synchronize(device)


def _collect_profile_episode(
    env,
    mac: BasicMAC,
    env_info: dict[str, Any],
    device: torch.device,
    epsilon: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], int, bool]:
    """Run EpisodeRunner semantics with explicit timers around record writes."""

    phases = {
        "reset": 0.0,
        "available_actions": 0.0,
        "record_before_step": 0.0,
        "policy": 0.0,
        "environment_step": 0.0,
        "record_after_step": 0.0,
        "terminal_record": 0.0,
    }
    episode_limit = int(env_info["episode_limit"])
    n_agents = int(env_info["n_agents"])
    n_actions = int(env_info["n_actions"])
    batch = _new_episode_batch(env_info)

    started = perf_counter()
    obs, state = env.reset()
    mac.init_hidden(batch_size=1)
    phases["reset"] = perf_counter() - started

    terminated = False
    t_episode = 0
    for t in range(episode_limit):
        started = perf_counter()
        available = env.get_avail_actions()
        phases["available_actions"] += perf_counter() - started

        started = perf_counter()
        batch["obs"][t] = np.asarray(obs, dtype=np.float32)
        batch["state"][t] = np.asarray(state, dtype=np.float32)
        batch["avail_actions"][t] = np.asarray(available, dtype=np.float32)
        phases["record_before_step"] += perf_counter() - started

        _synchronize(device)
        started = perf_counter()
        actions = mac.select_actions(
            obs,
            available,
            epsilon=epsilon,
            test_mode=False,
        )
        _synchronize(device)
        phases["policy"] += perf_counter() - started

        started = perf_counter()
        reward, terminated, _, next_obs, next_state = env.step(actions)
        phases["environment_step"] += perf_counter() - started

        started = perf_counter()
        batch["actions"][t, :, 0] = np.asarray(actions, dtype=np.int64)
        batch["reward"][t, 0] = float(reward)
        batch["terminated"][t, 0] = float(terminated)
        batch["filled"][t, 0] = 1.0
        phases["record_after_step"] += perf_counter() - started

        obs = next_obs
        state = next_state
        t_episode = t + 1
        if terminated:
            break

    started = perf_counter()
    if terminated:
        final_available = np.zeros((n_agents, n_actions), dtype=np.float32)
        final_available[:, 0] = 1.0
    else:
        final_available = np.asarray(env.get_avail_actions(), dtype=np.float32)
    batch["obs"][t_episode] = np.asarray(obs, dtype=np.float32)
    batch["state"][t_episode] = np.asarray(state, dtype=np.float32)
    batch["avail_actions"][t_episode] = final_available
    phases["terminal_record"] = perf_counter() - started
    return batch, phases, t_episode, bool(terminated)


def _print_table(title: str, rows: list[tuple[str, float, int]], total: float) -> None:
    print(f"\n{title}")
    print(f"{'项目':<34}{'秒':>11}{'占比':>10}{'调用':>10}{'平均毫秒':>13}")
    print("-" * 78)
    safe_total = max(float(total), 1e-12)
    for label, seconds, calls in rows:
        average_ms = seconds * 1000.0 / calls if calls else 0.0
        print(
            f"{label:<34}{seconds:>11.4f}"
            f"{seconds / safe_total * 100.0:>9.2f}%"
            f"{calls:>10d}{average_ms:>13.4f}"
        )


def _environment_rows(timings: MethodTimings) -> list[tuple[str, float, int]]:
    labels = (
        ("应用动作", "env.action_apply"),
        ("世界物理总计", "world.step"),
        ("  更新地图", "world.map_update"),
        ("  更新单位管理器", "world.unit_manager_update"),
        ("    应用 EnemyAI", "world.enemy_ai_update"),
        ("    更新 BaseUnit", "world.base_unit_update"),
        ("  更新子弹", "world.bullet_manager_update"),
        ("  刷新视野", "world.refresh_vision"),
        ("生成世界 snapshot/record", "env.snapshot"),
        ("计算 reward", "env.reward_calculate"),
        ("计算局部 observation", "env.observations"),
        ("计算全局 state", "env.global_state"),
        ("记录伤害事件 record_damage", "record.damage_event"),
        ("记录单位碰撞", "record.unit_collision"),
        ("print_record（训练通常不调用）", "record.print_record"),
        ("get_record（训练通常不调用）", "record.get_record"),
    )
    rows = []
    for display, key in labels:
        stat = timings.value(key)
        rows.append((display, stat.seconds, stat.calls))
    return rows


def _profile_stat(
    profiler: cProfile.Profile,
    *,
    filename_suffix: str,
    function_name: str,
) -> tuple[float, int]:
    cumulative = 0.0
    calls = 0
    stats = pstats.Stats(profiler)
    for (filename, _, name), values in stats.stats.items():
        if os.path.basename(filename) == filename_suffix and name == function_name:
            calls += int(values[1])
            cumulative += float(values[3])
    return cumulative, calls


def _learner_rows(
    profiler: cProfile.Profile,
) -> list[tuple[str, float, int]]:
    targets = (
        ("在线网络前向", "qmix_learner.py", "_online_forward"),
        ("目标网络前向", "qmix_learner.py", "_target_forward"),
        ("ETD agent forward（含于前向）", "etd_rnn_agent.py", "forward"),
        ("RNN agent forward（含于前向）", "rnn_agent.py", "forward"),
        ("QMIX mixer forward", "qmix.py", "forward"),
        ("DVD mixer forward", "dvd.py", "forward"),
        ("反向传播 backward", "_tensor.py", "backward"),
    )
    rows = []
    for label, suffix, function_name in targets:
        seconds, calls = _profile_stat(
            profiler,
            filename_suffix=suffix,
            function_name=function_name,
        )
        if calls:
            rows.append((label, seconds, calls))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile MARL2 rollout, EnemyAI, record/reward, and learner time"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Profile episode length; use 2200 for the production configuration",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Learner batch size; use 16 for the production configuration",
    )
    parser.add_argument("--learner-updates", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--skip-learner", action="store_true")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.learner_updates <= 0:
        raise ValueError("steps, batch-size, and learner-updates must be > 0")

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_global_seeds(seed)
    device = get_device(args.device)
    print_device_info(device)

    env_cfg = dict(cfg["env"])
    env_cfg["headless"] = True
    env_cfg["use_video"] = False
    env_cfg["max_steps"] = int(args.steps)
    env_name = env_cfg.get("name", "jackal")
    env_cls = ENV_REGISTRY[env_name]

    timings = MethodTimings()
    _install_environment_timers(timings)
    env = None
    try:
        env = env_cls(env_cfg)
        env_info = env.get_env_info()
        mac = BasicMAC(
            n_agents=env_info["n_agents"],
            n_actions=env_info["n_actions"],
            obs_shape=env_info["obs_shape"],
            agent_cfg=cfg["agent"],
            device=device,
        )
        mixer = _build_mixer(cfg["algo"], env_info, mac, device)
        learner_cls = LEARNER_REGISTRY[cfg["algo"].get("name", "qmix")]
        learner = learner_cls(mac, mixer, cfg["algo"], device)
        if args.checkpoint:
            learner.load_models(args.checkpoint)

        _warm_up(env, mac, device, min(args.warmup_steps, args.steps))
        timings.clear()
        rollout_started = perf_counter()
        episode_batch, rollout_phases, episode_steps, terminated = (
            _collect_profile_episode(
                env,
                mac,
                env_info,
                device,
                epsilon=float(args.epsilon),
            )
        )
        rollout_wall = perf_counter() - rollout_started

        rollout_rows = [
            ("环境 reset", rollout_phases["reset"], 1),
            ("计算可用动作", rollout_phases["available_actions"], episode_steps),
            ("写入 record：obs/state/action-mask", rollout_phases["record_before_step"], episode_steps),
            ("策略网络选择动作", rollout_phases["policy"], episode_steps),
            ("环境 step", rollout_phases["environment_step"], episode_steps),
            ("写入 record：action/reward/done", rollout_phases["record_after_step"], episode_steps),
            ("写入终止帧 record", rollout_phases["terminal_record"], 1),
        ]
        accounted = sum(row[1] for row in rollout_rows)
        rollout_rows.append(("其他 Python 调度", max(0.0, rollout_wall - accounted), 1))
        _print_table("=== Rollout 顶层时长（互斥，可直接相加）===", rollout_rows, rollout_wall)

        env_step_total = rollout_phases["environment_step"]
        _print_table(
            "=== 环境 step 内部时长（分层包含，不应跨缩进相加）===",
            _environment_rows(timings),
            env_step_total,
        )

        replay = EpisodeReplayBuffer(buffer_size=max(args.batch_size, 2))
        started = perf_counter()
        for _ in range(args.batch_size):
            replay.insert_episode_batch(episode_batch)
        replay_insert_seconds = perf_counter() - started

        started = perf_counter()
        learner_batch = replay.sample(args.batch_size)
        replay_sample_seconds = perf_counter() - started

        learner_seconds = 0.0
        learner_profiler = cProfile.Profile()
        if not args.skip_learner:
            _synchronize(device)
            started = perf_counter()
            learner_profiler.enable()
            for _ in range(args.learner_updates):
                learner.train(learner_batch)
            learner_profiler.disable()
            _synchronize(device)
            learner_seconds = perf_counter() - started
            _print_table(
                "=== Learner 关键调用（累计包含，不能直接相加）===",
                _learner_rows(learner_profiler),
                learner_seconds,
            )
            print(f"\n=== Learner cProfile 前 {args.top} 项 ===")
            stream = io.StringIO()
            pstats.Stats(learner_profiler, stream=stream).sort_stats(
                "cumulative"
            ).print_stats(args.top)
            print(stream.getvalue())
            if device.type != "cpu":
                print(
                    "[注意] GPU 上表内 learner 总时长已通过设备同步校准；"
                    "cProfile 的各 Python 调用主要反映算子提交时间，"
                    "不等同于每个 GPU kernel 的独占执行时间。"
                )

        total_cycle = rollout_wall + replay_insert_seconds + replay_sample_seconds + learner_seconds
        cycle_rows = [
            ("收集 rollout", rollout_wall, 1),
            ("写入 replay buffer", replay_insert_seconds, args.batch_size),
            ("采样并堆叠 replay batch", replay_sample_seconds, 1),
        ]
        if not args.skip_learner:
            cycle_rows.append(("Learner 更新", learner_seconds, args.learner_updates))
        _print_table("=== 单次收集—学习循环总占比 ===", cycle_rows, total_cycle)

        print(
            "\n[说明] 此脚本使用单进程环境，以便看到 EnemyAI 等 worker 内部耗时。"
            "它不包含 ParallelEpisodeRunner 的进程启动、管道序列化和等待耗时，也不包含评估、"
            "TensorBoard 和保存 checkpoint。环境内部表是嵌套累计时间；同一层级才适合比较。"
        )
        print(
            f"[样本] steps={episode_steps}/{args.steps} terminated={terminated} "
            f"batch_size={args.batch_size} learner_updates={args.learner_updates} "
            f"device={device}"
        )

        if args.json_output:
            payload = {
                "config": args.config,
                "device": str(device),
                "requested_steps": int(args.steps),
                "episode_steps": int(episode_steps),
                "terminated": bool(terminated),
                "batch_size": int(args.batch_size),
                "learner_updates": int(args.learner_updates),
                "rollout_wall_seconds": float(rollout_wall),
                "rollout_phases": rollout_phases,
                "environment_method_timings": {
                    key: {"seconds": stat.seconds, "calls": stat.calls}
                    for key, stat in timings.stats.items()
                },
                "replay_insert_seconds": float(replay_insert_seconds),
                "replay_sample_seconds": float(replay_sample_seconds),
                "learner_seconds": float(learner_seconds),
                "total_cycle_seconds": float(total_cycle),
            }
            output_dir = os.path.dirname(os.path.abspath(args.json_output))
            os.makedirs(output_dir, exist_ok=True)
            with open(args.json_output, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            print(f"[JSON] {os.path.abspath(args.json_output)}")
    finally:
        if env is not None:
            env.close()
        timings.close()


if __name__ == "__main__":
    main()
