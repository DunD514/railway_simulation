"""Tabular action-value learners and the independent multi-agent wrapper.

Agents act only at their own decision points, so each agent sees a semi-Markov
decision process. Between two consecutive decisions of agent i (k epochs apart)
the per-epoch rewards are accumulated as R = sum_{j<k} gamma^j r_{t+j} and the
bootstrap term is discounted by gamma^k.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from agents.discretize import Discretizer
from sim.env import N_ACTIONS, CorridorEnv, Observation, StepResult


class TabularAgent:
    """Base class: one Q-table, epsilon-greedy behaviour, subclass supplies the TD target."""

    name = "tabular"

    def __init__(
        self, n_states: int, alpha: float, gamma: float, q_init: float = 0.0, n_actions: int = N_ACTIONS
    ) -> None:
        self.q = np.full((n_states, n_actions), float(q_init), dtype=np.float64)
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def greedy(self, s: int, rng: np.random.Generator | None = None) -> int:
        """Greedy action; ties broken uniformly at random if ``rng`` is given, else lowest index."""
        row = self.q[s]
        best = np.flatnonzero(row == row.max())
        if rng is None or len(best) == 1:
            return int(best[0])
        return int(rng.choice(best))

    def act(self, s: int, epsilon: float, rng: np.random.Generator) -> int:
        """Epsilon-greedy action."""
        if rng.random() < epsilon:
            return int(rng.integers(self.q.shape[1]))
        return self.greedy(s, rng)

    def target(self, ret: float, disc: float, s_next: int | None, a_next: int | None) -> float:
        """TD target; implemented by subclasses."""
        raise NotImplementedError

    def update(self, s: int, a: int, ret: float, disc: float, s_next: int | None, a_next: int | None) -> float:
        """Apply one TD update to Q(s, a) and return the TD error.

        ``ret`` is the (discounted) reward accumulated since the decision, ``disc``
        the discount gamma^k for the bootstrap, ``s_next=None`` marks a terminal.
        """
        td = self.target(ret, disc, s_next, a_next) - self.q[s, a]
        self.q[s, a] += self.alpha * td
        self.visits[s, a] += 1
        return float(td)


@dataclass
class _Pending:
    s: int
    a: int
    ret: float = 0.0
    disc: float = 1.0


def linear_epsilon(episode: int, n_episodes: int, start: float, end: float, decay_frac: float) -> float:
    """Linear epsilon decay from ``start`` to ``end`` over ``decay_frac`` of the episodes."""
    horizon = max(1.0, decay_frac * n_episodes)
    frac = min(1.0, episode / horizon)
    return start + (end - start) * frac


class IndependentLearners:
    """N independent tabular learners (one per station) sharing an environment.

    reward_mode: "shared" (cooperative team reward) or "local" (own-section reward).
    """

    def __init__(
        self,
        agent_factory: Callable[[], TabularAgent],
        n_agents: int,
        discretizer: Discretizer,
        reward_mode: str = "shared",
    ) -> None:
        self.agents = [agent_factory() for _ in range(n_agents)]
        self.disc = discretizer
        self.reward_mode = reward_mode
        self._pending: list[_Pending | None] = [None] * n_agents

    @property
    def name(self) -> str:
        """Algorithm name of the underlying agents."""
        return self.agents[0].name

    def begin_episode(self) -> None:
        """Forget pending transitions."""
        self._pending = [None] * len(self.agents)

    def act(
        self, observations: Sequence[Observation], epsilon: float, rng: np.random.Generator, learn: bool = True
    ) -> dict[int, int]:
        """Choose actions for the agents that must decide, updating their previous transition first."""
        actions: dict[int, int] = {}
        for obs in observations:
            i = obs.agent
            ag = self.agents[i]
            s = self.disc.encode(obs)
            a = ag.act(s, epsilon, rng) if epsilon > 0 else ag.greedy(s)
            if learn:
                p = self._pending[i]
                if p is not None:
                    ag.update(p.s, p.a, p.ret, p.disc, s, a)
                self._pending[i] = _Pending(s, a)
            actions[i] = a
        return actions

    def observe(self, result: StepResult) -> None:
        """Accumulate this epoch's reward into every agent's open transition."""
        for i, p in enumerate(self._pending):
            if p is None:
                continue
            r = result.shared_reward if self.reward_mode == "shared" else result.local_rewards[i]
            p.ret += p.disc * r
            p.disc *= self.agents[i].gamma

    def end_episode(self) -> None:
        """Close open transitions with a terminal (non-bootstrapped) update."""
        for i, p in enumerate(self._pending):
            if p is not None:
                self.agents[i].update(p.s, p.a, p.ret, 0.0, None, None)
        self._pending = [None] * len(self.agents)

    def q_tables(self) -> list[np.ndarray]:
        """Copies of the Q-tables."""
        return [ag.q.copy() for ag in self.agents]


def run_learning_episode(
    env: CorridorEnv, learners: IndependentLearners, epsilon: float, rng: np.random.Generator, learn: bool = True
) -> dict:
    """Run one episode with the learners acting (and learning if ``learn``). Returns env metrics."""
    learners.begin_episode()
    while not env.done:
        obs = env.decision_requests()
        actions = learners.act(obs, epsilon, rng, learn=learn) if obs else {}
        result = env.step(actions)
        if learn:
            learners.observe(result)
    if learn:
        learners.end_episode()
    return env.metrics()
