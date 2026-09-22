"""Benchmark 5v5 EnemyAI strength levels on a small symmetric arena.

This is intentionally a standalone benchmark rather than a unittest: the
default matrix runs 540 mirrored battle legs and should not slow down normal
test discovery.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game.BattleWorld import BattleWorld
from game.Map.GameMap import GameMap
from game.Parameter import Team
from game.Unit.UnitManager import UnitManager


MAP_ROWS = (
    "xxxxxxxxxx",
    "xoooooooox",
    "xoooooooox",
    "xoooooooox",
    "xoooooooox",
    "xoooooooox",
    "xxxxxxxxxx",
)
BASE_Y_POSITIONS = (96.0, 160.0, 224.0, 288.0, 352.0)


def _other_team(team: Team) -> Team:
    return Team.ENEMY if team == Team.PLAYER else Team.PLAYER


def _build_world(
    our_level: int,
    opponent_level: int,
    seed: int,
    our_on_left: bool,
) -> tuple[BattleWorld, Team, Team]:
    """Create one battle leg from a reproducible symmetric layout."""

    rng = random.Random(seed)
    random.seed(seed)
    game_map = GameMap(list(MAP_ROWS), tile_size=64)
    world = BattleWorld(
        game_map,
        unit_manager=UnitManager(
            enable_unit_collision=True,
            use_tear_drop_vision=False,
            auto_communicate=False,
        ),
    )

    # Every seed is run twice with the levels swapped across the same mirrored
    # layout. This separates AI strength from a layout or physical-side bias.
    our_team = Team.PLAYER if our_on_left else Team.ENEMY
    opponent_team = _other_team(our_team)
    left_team = Team.PLAYER
    right_team = Team.ENEMY
    width, _ = game_map.get_map_size()

    lane_order = list(range(5))
    rng.shuffle(lane_order)
    for rank, lane_index in enumerate(lane_order):
        y = BASE_Y_POSITIONS[lane_index] + rng.uniform(-8.0, 8.0)
        left_x = 128.0 + rng.uniform(-8.0, 8.0)
        right_x = float(width) - left_x
        heading_offset = rng.uniform(-8.0, 8.0)

        for team, position, heading, unit_id in (
            (left_team, (left_x, y), 90.0 + heading_offset, rank + 1),
            (
                right_team,
                (right_x, y),
                270.0 - heading_offset,
                rank + 100,
            ),
        ):
            level = our_level if team == our_team else opponent_level
            world.create_unit(
                "tank",
                team,
                position,
                unit_id=unit_id,
                using_ai=True,
                ai_intelligence_level=level,
                initial_heading=heading,
            )

    world.refresh_vision()
    return world, our_team, opponent_team


def _team_units(world: BattleWorld, team: Team):
    return [unit for unit in world.unit_manager.units if unit.team == team]


def run_battle(
    our_level: int,
    opponent_level: int,
    seed: int,
    game_index: int,
    leg: int,
    *,
    max_steps: int,
    delta_time: float,
) -> dict[str, float | int | str]:
    world, our_team, opponent_team = _build_world(
        our_level,
        opponent_level,
        seed,
        our_on_left=leg == 1,
    )
    our_units = _team_units(world, our_team)
    opponent_units = _team_units(world, opponent_team)

    steps = 0
    while steps < max_steps:
        if not any(unit.is_alive for unit in our_units):
            break
        if not any(unit.is_alive for unit in opponent_units):
            break
        world.step(delta_time)
        steps += 1

    our_alive = sum(unit.is_alive for unit in our_units)
    opponent_alive = sum(unit.is_alive for unit in opponent_units)
    our_health = sum(max(0.0, float(unit.health)) for unit in our_units)
    opponent_health = sum(max(0.0, float(unit.health)) for unit in opponent_units)
    if our_alive > 0 and opponent_alive == 0:
        result = "win"
    elif opponent_alive > 0 and our_alive == 0:
        result = "loss"
    else:
        result = "draw"

    return {
        "our_level": our_level,
        "opponent_level": opponent_level,
        "game": game_index + 1,
        "leg": leg,
        "seed": seed,
        "our_side": "left" if our_team == Team.PLAYER else "right",
        "result": result,
        "steps": steps,
        "seconds": steps * delta_time,
        "our_alive": our_alive,
        "opponent_alive": opponent_alive,
        "our_health": our_health,
        "opponent_health": opponent_health,
    }


def _summarize(rows: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    games = len(rows)
    wins = sum(row["result"] == "win" for row in rows)
    losses = sum(row["result"] == "loss" for row in rows)
    draws = games - wins - losses
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games,
        "non_draw_win_rate": wins / max(1, wins + losses),
        "avg_seconds": sum(float(row["seconds"]) for row in rows) / games,
        "avg_our_alive": sum(int(row["our_alive"]) for row in rows) / games,
        "avg_opponent_alive": sum(int(row["opponent_alive"]) for row in rows) / games,
        "avg_health_margin": sum(
            float(row["our_health"]) - float(row["opponent_health"])
            for row in rows
        ) / games,
    }


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--our-levels", type=int, nargs="+", default=[3, 6, 9])
    parser.add_argument(
        "--opponent-levels",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
    )
    parser.add_argument("--base-seed", type=int, default=202609220)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--delta-time", type=float, default=0.03)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/benchmarks/enemy_ai_level_5v5.csv",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    levels = [*args.our_levels, *args.opponent_levels]
    if args.games <= 0:
        parser.error("--games must be > 0")
    if any(level < 1 or level > 9 for level in levels):
        parser.error("AI levels must be between 1 and 9")
    if args.max_steps <= 0 or args.delta_time <= 0.0:
        parser.error("--max-steps and --delta-time must be > 0")

    all_rows: list[dict[str, float | int | str]] = []
    started = time.perf_counter()
    total_matchups = len(args.our_levels) * len(args.opponent_levels)
    completed_matchups = 0
    for our_level in args.our_levels:
        for opponent_level in args.opponent_levels:
            matchup_rows = []
            for game_index in range(args.games):
                seed = args.base_seed + game_index
                for leg in (1, 2):
                    row = run_battle(
                        our_level,
                        opponent_level,
                        seed,
                        game_index,
                        leg,
                        max_steps=args.max_steps,
                        delta_time=args.delta_time,
                    )
                    matchup_rows.append(row)
                    all_rows.append(row)
                    if args.verbose:
                        print(
                            f"L{our_level} vs L{opponent_level} "
                            f"pair={game_index + 1}/{args.games} leg={leg} "
                            f"seed={seed} side={row['our_side']} "
                            f"result={row['result']} "
                            f"alive={row['our_alive']}:{row['opponent_alive']} "
                            f"time={float(row['seconds']):.1f}s",
                            flush=True,
                        )

            completed_matchups += 1
            summary = _summarize(matchup_rows)
            elapsed = time.perf_counter() - started
            print(
                f"[{completed_matchups:02d}/{total_matchups}] "
                f"L{our_level} vs L{opponent_level}: "
                f"W-L-D={summary['wins']}-{summary['losses']}-{summary['draws']} "
                f"WR={float(summary['win_rate']):.0%} "
                f"alive={float(summary['avg_our_alive']):.2f}:"
                f"{float(summary['avg_opponent_alive']):.2f} "
                f"avg_time={float(summary['avg_seconds']):.1f}s "
                f"wall={elapsed:.1f}s",
                flush=True,
            )

    _write_csv(args.output, all_rows)
    print("\n[Overall by tested level]")
    for level in args.our_levels:
        summary = _summarize(
            [row for row in all_rows if row["our_level"] == level]
        )
        print(
            f"L{level}: W-L-D={summary['wins']}-{summary['losses']}-"
            f"{summary['draws']} WR={float(summary['win_rate']):.1%} "
            f"decisive_WR={float(summary['non_draw_win_rate']):.1%}"
        )

    print("\n[Physical side balance]")
    for side in ("left", "right"):
        summary = _summarize(
            [row for row in all_rows if row["our_side"] == side]
        )
        print(
            f"{side}: W-L-D={summary['wins']}-{summary['losses']}-"
            f"{summary['draws']} WR={float(summary['win_rate']):.1%} "
            f"decisive_WR={float(summary['non_draw_win_rate']):.1%}"
        )
    print(f"Detailed results: {args.output}")


if __name__ == "__main__":
    main()
