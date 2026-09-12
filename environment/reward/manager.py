"""Reward facade operating on snapshots and game-engine combat events."""

from typing import Any, Dict, Sequence, Tuple

from environment.reward.auto_aim import calculate_auto_aim_reward
from environment.reward.common import (
    BattleStats,
    RewardContext,
    collect_battle_stats,
    summarize_combat_events,
)
from environment.reward.manual_aim import calculate_manual_aim_reward
from game.BattleState import WorldSnapshot


class RewardManager:
    """Own episode reward state without retaining the mutable environment."""

    def __init__(
        self,
        *,
        auto_aim: bool,
        max_steps: int,
        arena_size: tuple[float, float],
        config: Dict[str, Dict[str, float]],
    ) -> None:
        self.auto_aim = bool(auto_aim)
        self.max_steps = int(max_steps)
        self.arena_size = arena_size
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.episode_return = 0.0
        self.has_enemy_kill = False

    def battle_stats(self, snapshot: WorldSnapshot) -> BattleStats:
        return collect_battle_stats(snapshot, self.arena_size)

    def calculate(
        self,
        before: WorldSnapshot,
        after: WorldSnapshot,
        actions: Sequence[int],
    ) -> Tuple[float, Dict[str, Any]]:
        before_stats = self.battle_stats(before)
        after_stats = self.battle_stats(after)
        combat = summarize_combat_events(after.combat_events)
        if combat.enemies_destroyed:
            self.has_enemy_kill = True
        context = RewardContext(
            step=after.tick,
            max_steps=self.max_steps,
            episode_return=self.episode_return,
            has_enemy_kill=self.has_enemy_kill,
        )

        if self.auto_aim:
            reward, info = calculate_auto_aim_reward(
                self.config["auto_aim"],
                before_stats,
                after_stats,
                combat,
                context,
            )
        else:
            reward, info = calculate_manual_aim_reward(
                self.config["manual_aim"],
                before_stats,
                after_stats,
                combat,
                context,
                after,
                actions,
            )
        self.episode_return += float(reward)
        return reward, info
