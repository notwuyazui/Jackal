"""Compare working-tree EnemyAI with its pre-change Git revision, headlessly.

Example: python tests/benchmark_enemy_ai_tactics.py --baseline-ref HEAD --repeats 5
No network inference, learning, rendering or video recording is timed here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from game.BattleWorld import BattleWorld
from game.Parameter import Team
from training.marl2.envs.jackal_env import JackalMultiAgentEnv

AI_MODULE = importlib.import_module("game.Unit.EnemyAI")
CONFIG = (PROJECT_ROOT / "training/configs/marl2/"
          "jackal_autoaim_5v5_etdqmix_hetero_valley_mapfeat7_10m_v2.json")


def baseline_class(ref):
    source = subprocess.check_output(
        ["git", "show", f"{ref}:game/Unit/EnemyAI.py"], cwd=PROJECT_ROOT, text=True,
    )
    namespace = {"__name__": "enemy_ai_benchmark_baseline"}
    exec(compile(source, f"{ref}:EnemyAI.py", "exec"), namespace)
    return namespace["EnemyAI"]


def timed_class(base, counters):
    class TimedAI(base):
        def update(self):
            started = perf_counter()
            super().update()
            counters["ai_seconds"] += perf_counter() - started
            counters["ai_calls"] += 1
            counters[self.ai_state] += 1
    return TimedAI


def battle_world(level, seed):
    world = BattleWorld().load_game_state(
        "border_map_3t_vs_3t",
        using_ai_by_team={Team.PLAYER: True, Team.ENEMY: True},
        ai_intelligence_by_team={Team.PLAYER: level, Team.ENEMY: level},
        position_jitter=3.0, rng=random.Random(seed),
    )
    for unit in world.unit_manager.units:
        unit.direction_angle = unit.turret_direction_angle = unit.turret_target_angle = (
            90.0 if unit.team == Team.PLAYER else 270.0
        )
    world.refresh_vision()
    return world


def run_battle(base, level, seed, steps):
    counters = Counter()
    random.seed(seed)
    with patch.object(AI_MODULE, "EnemyAI", timed_class(base, counters)):
        world = battle_world(level, seed)
        steps_run = 0
        seconds = 0.0
        for _ in range(steps):
            if len({unit.team for unit in world.unit_manager.units if unit.is_alive}) < 2:
                break
            started = perf_counter()
            world.step(0.03)
            seconds += perf_counter() - started
            steps_run += 1
        winner = next(iter({unit.team.name for unit in world.unit_manager.units
                            if unit.is_alive}), "none")
        if len({unit.team for unit in world.unit_manager.units if unit.is_alive}) == 2:
            winner = "unfinished"
    return {"step_us": seconds / max(1, steps_run) * 1e6,
            "ai_us": counters["ai_seconds"] / max(1, counters["ai_calls"]) * 1e6,
            "steps": steps_run, "winner": winner,
            "maneuver_calls": sum(counters[state] for state in counters
                                  if str(state).startswith(("reposition", "spread_", "relieve_", "clear_fire_")))}


def run_training_env(base, level, seed, steps):
    counters = Counter()
    env_args = json.loads(CONFIG.read_text(encoding="utf-8"))["env"]
    env_args["enemy_ai_intelligence_level"] = level
    random.seed(seed)
    pattern_rng = random.Random(seed)
    actions = [[pattern_rng.randrange(10) for _ in range(5)] for _ in range(steps)]
    with patch.object(AI_MODULE, "EnemyAI", timed_class(base, counters)):
        env = JackalMultiAgentEnv(env_args)
        try:
            env.reset(seed=seed)
            seconds = 0.0
            episodes = 1
            for action in actions:
                started = perf_counter()
                _, done, _, _, _ = env.step(action)
                seconds += perf_counter() - started
                if done:
                    episodes += 1
                    env.reset(seed=seed + episodes - 1)
        finally:
            env.close()
    return {"step_us": seconds / steps * 1e6,
            "ai_us": counters["ai_seconds"] / max(1, counters["ai_calls"]) * 1e6,
            "steps": steps, "episodes": episodes}


def run_frozen_combat(base, level, seed, steps):
    """Identical stationary inputs isolate decision cost from changed battles."""
    random.seed(seed)
    with patch.object(AI_MODULE, "EnemyAI", base):
        world = battle_world(level, seed)
        for unit in world.unit_manager.units:
            unit.position = (350.0 if unit.team == Team.PLAYER else 570.0,
                             220.0 + (unit.id % 100 - (1 if unit.team == Team.PLAYER else 0)) * 100.0)
            unit._update_collision_box()
            unit._update_bounding_box()
        world.refresh_vision()
        for _ in range(30):
            for ai in world.unit_manager.enemy_ais:
                ai.update()
        seconds = 0.0
        for _ in range(steps):
            world.unit_manager.invalidate_perception_cache()
            started = perf_counter()
            for ai in world.unit_manager.enemy_ais:
                ai.update()
            seconds += perf_counter() - started
    return {"ai_us": seconds / (steps * 6) * 1e6}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="HEAD")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--levels", type=int, nargs="+", default=[5, 9])
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1 or any(not 1 <= level <= 9 for level in args.levels):
        parser.error("positive repeats/steps and levels 1..9 required")
    versions = {"baseline": baseline_class(args.baseline_ref), "updated": AI_MODULE.EnemyAI}
    results = {}
    for level in args.levels:
        for name, runner in (("frozen_combat", run_frozen_combat),
                             ("border_3v3", run_battle), ("valley_5v5_env", run_training_env)):
            rows = {version: [] for version in versions}
            for repeat in range(args.repeats):
                order = list(versions) if repeat % 2 == 0 else list(reversed(versions))
                for version in order:
                    row = runner(versions[version], level, 20261001 + repeat, args.steps)
                    rows[version].append(row)
            summary = {}
            for metric in ("ai_us", "step_us"):
                if metric not in rows["baseline"][0]:
                    continue
                before = statistics.median(row[metric] for row in rows["baseline"])
                after = statistics.median(row[metric] for row in rows["updated"])
                summary[metric] = {"before": round(before, 3), "after": round(after, 3),
                                   "change_percent": round((after / before - 1) * 100, 2)}
            results[f"{name}_level_{level}"] = {"median": summary, "runs": rows}
            print(f"[Benchmark] {name} level={level}: {json.dumps(summary)}", flush=True)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
