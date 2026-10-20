"""Independent Q-Learning: off-policy tabular Q-learning per agent."""
from __future__ import annotations

from agents.tabular import TabularAgent


class QLearningAgent(TabularAgent):
    """Q-learning: target = R + gamma^k * max_a' Q(s', a')."""

    name = "iql"

    def target(self, ret: float, disc: float, s_next: int | None, a_next: int | None) -> float:
        """Off-policy target that bootstraps from the greedy value of the next state."""
        if s_next is None:
            return ret
        return ret + disc * float(self.q[s_next].max())
