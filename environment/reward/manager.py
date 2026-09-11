"""Reward facade selected by the environment's aiming mode."""

from typing import TYPE_CHECKING, Any, Dict, Sequence, Tuple

from environment.reward.auto_aim import calculate_auto_aim_reward
from environment.reward.common import (
    BattleStats,
    collect_battle_stats,
)
from environment.reward.manual_aim import calculate_manual_aim_reward

if TYPE_CHECKING:
    from environment.jackal_env import JackalEnv


class RewardManager:
    """Own reward-mode selection while the environment owns episode state."""

    def __init__(
        self,
        env: "JackalEnv",
        config: Dict[str, Dict[str, float]],
    ) -> None:
        self.env = env
        self.config = config

    def battle_stats(self) -> BattleStats:
        return collect_battle_stats(self.env)

    def calculate(
        self,
        before: BattleStats,
        after: BattleStats,
        actions: Sequence[int],
    ) -> Tuple[float, Dict[str, Any]]:
        if self.env.auto_aim:
            return calculate_auto_aim_reward(
                self.env,
                self.config["auto_aim"],
                before,
                after,
            )
        return calculate_manual_aim_reward(
            self.env,
            self.config["manual_aim"],
            before,
            after,
            actions,
        )
