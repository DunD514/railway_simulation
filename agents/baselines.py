"""Heuristic baselines and a greedy wrapper for trained Q-tables.

Every policy exposes ``actions(env, observations) -> {agent: action}``.
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from agents.discretize import Discretizer
from sim.env import HOLD, LOOP, PROCEED, CorridorEnv, Observation


class Policy(Protocol):
    """Anything that maps the current decision requests to actions."""

    name: str

    def actions(self, env: CorridorEnv, observations: Sequence[Observation]) -> dict[int, int]:
        """Return one action per requesting agent."""
        ...


class FCFSPolicy:
    """First-come-first-served: the freight already ahead keeps precedence (always PROCEED)."""

    name = "fcfs"

    def actions(self, env: CorridorEnv, observations: Sequence[Observation]) -> dict[int, int]:
        """PROCEED for every conflict."""
        return {o.agent: PROCEED for o in observations}


class PassengerFirstPolicy:
    """Loop the freight whenever a conflicting passenger is in the window and the loop is usable.

    A freight already in the loop stays there while any conflict remains (it is
    released automatically once the conflict clears).
    """

    name = "passenger_first"

    def actions(self, env: CorridorEnv, observations: Sequence[Observation]) -> dict[int, int]:
        """LOOP when admissible, else PROCEED."""
        return {o.agent: (LOOP if env.loop_valid(o) else PROCEED) for o in observations}


class RandomPolicy:
    """Uniformly random over PROCEED / LOOP / HOLD (sanity floor)."""

    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def actions(self, env: CorridorEnv, observations: Sequence[Observation]) -> dict[int, int]:
        """Random action per conflict."""
        return {o.agent: int(self.rng.integers(3)) for o in observations}


class GreedyQPolicy:
    """Greedy policy from per-agent Q-tables (ties -> lowest action index, i.e. PROCEED)."""

    def __init__(self, name: str, q_tables: Sequence[np.ndarray], discretizer: Discretizer) -> None:
        self.name = name
        self.q = [np.asarray(q) for q in q_tables]
        self.disc = discretizer

    def actions(self, env: CorridorEnv, observations: Sequence[Observation]) -> dict[int, int]:
        """argmax_a Q_i(s_i, a) for each requesting agent i."""
        return {o.agent: int(np.argmax(self.q[o.agent][self.disc.encode(o)])) for o in observations}


def run_policy_episode(env: CorridorEnv, policy: Policy) -> dict:
    """Roll out one episode under a fixed policy and return the env metrics."""
    while not env.done:
        obs = env.decision_requests()
        env.step(policy.actions(env, obs) if obs else {})
    return env.metrics()


__all__ = [
    "FCFSPolicy", "PassengerFirstPolicy", "RandomPolicy", "GreedyQPolicy", "Policy", "run_policy_episode",
    "PROCEED", "LOOP", "HOLD",
]
