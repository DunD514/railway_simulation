"""SARSA: on-policy tabular TD control per agent."""
from __future__ import annotations

from agents.tabular import TabularAgent


class SarsaAgent(TabularAgent):
    """SARSA: target = R + gamma^k * Q(s', a') with a' the action actually taken."""

    name = "sarsa"

    def target(self, ret: float, disc: float, s_next: int | None, a_next: int | None) -> float:
        """On-policy target that bootstraps from the behaviour policy's next action."""
        if s_next is None:
            return ret
        if a_next is None:
            raise ValueError("SARSA needs the next action")
        return ret + disc * float(self.q[s_next, a_next])
