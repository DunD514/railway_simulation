"""Q-update correctness (IQL vs SARSA) on a hand-computed 2-step example, and metric calculations."""
from __future__ import annotations

import numpy as np
import pytest

from agents.iql import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.tabular import linear_epsilon
from experiments.stats import bootstrap_ci, holm_correction, paired_comparison, paired_permutation_test


def _two_step(agent_cls):
    """Two decisions: s0 --a=1, r=-2 (1 epoch)--> s1 --a=2, r=-4, 2 epochs--> terminal.

    Q starts with Q[1] = [-1, -5, -3]; alpha=0.5, gamma=0.9.
    """
    ag = agent_cls(n_states=2, alpha=0.5, gamma=0.9)
    ag.q[1] = [-1.0, -5.0, -3.0]
    # step 1: transition (s0, a1) closed at s1 where the behaviour policy picks a'=2 (exploratory)
    ag.update(0, 1, ret=-2.0, disc=0.9, s_next=1, a_next=2)
    q01 = ag.q[0, 1]
    # step 2: (s1, a2) accumulates -4 over two epochs: R = -4 + 0.9*0 ... given as ret, terminal
    ag.update(1, 2, ret=-4.0 + 0.9 * -1.0, disc=0.0, s_next=None, a_next=None)
    return q01, ag.q[1, 2]


def test_iql_two_step_hand_computed():
    q01, q12 = _two_step(QLearningAgent)
    # IQL bootstraps from max Q[s1] = -1: target = -2 + 0.9*(-1) = -2.9 -> Q = 0 + 0.5*(-2.9) = -1.45
    assert q01 == pytest.approx(-1.45)
    # terminal: target = -4.9 -> Q = -3 + 0.5*(-4.9 + 3) = -3.95
    assert q12 == pytest.approx(-3.95)


def test_sarsa_two_step_hand_computed():
    q01, q12 = _two_step(SarsaAgent)
    # SARSA bootstraps from the action actually taken, Q[s1, 2] = -3: target = -2 - 2.7 = -4.7 -> -2.35
    assert q01 == pytest.approx(-2.35)
    assert q12 == pytest.approx(-3.95)


def test_iql_and_sarsa_agree_when_next_action_is_greedy():
    a, b = QLearningAgent(2, 0.1, 0.95), SarsaAgent(2, 0.1, 0.95)
    for ag in (a, b):
        ag.q[1] = [-1.0, -5.0, -3.0]
        ag.update(0, 0, ret=-10.0, disc=0.95 ** 3, s_next=1, a_next=0)
    assert a.q[0, 0] == pytest.approx(b.q[0, 0]) == pytest.approx(0.1 * (-10.0 - 0.95 ** 3))


def test_smdp_accumulation_through_learners(cfg):
    """IndependentLearners accumulates sum gamma^j r_j and bootstraps with gamma^k."""
    from agents.discretize import build_discretizer
    from agents.tabular import IndependentLearners
    from sim.env import Observation, StepResult

    disc = build_discretizer(cfg)
    L = IndependentLearners(lambda: QLearningAgent(disc.n_states, alpha=1.0, gamma=0.5), 6, disc)
    rng = np.random.default_rng(0)
    o1 = Observation(0, 1, 2, 5.0, 3.0, 60.0, 2.0, 0, 0.0, 0, 1, 0)
    o2 = Observation(0, 1, 2, 30.0, 3.0, 60.0, 2.0, 0, 0.0, 0, 1, 0)
    s1, s2 = disc.encode(o1), disc.encode(o2)
    assert s1 != s2
    L.agents[0].q[s2] = [-8.0, -6.0, -7.0]
    L.begin_episode()
    a = L.act([o1], epsilon=0.0, rng=rng)[0]
    for r in (-1.0, -2.0, -4.0):  # three epochs
        L.observe(StepResult(r, [r] * 6, [], 0.0, 0.0, False))
    L.act([o2], epsilon=0.0, rng=rng)
    # R = -1 + 0.5*-2 + 0.25*-4 = -3 ; bootstrap 0.5^3 * max(-8,-6,-7) = -0.75 ; alpha = 1
    assert L.agents[0].q[s1, a] == pytest.approx(-3.75)


def test_linear_epsilon_schedule():
    assert linear_epsilon(0, 100, 1.0, 0.05, 0.5) == pytest.approx(1.0)
    assert linear_epsilon(25, 100, 1.0, 0.05, 0.5) == pytest.approx(0.525)
    assert linear_epsilon(80, 100, 1.0, 0.05, 0.5) == pytest.approx(0.05)


def test_bootstrap_ci_contains_mean_and_shrinks():
    x = np.arange(10, dtype=float)
    m, lo, hi = bootstrap_ci(x, n_resamples=4000, seed=1)
    assert m == pytest.approx(4.5)
    assert lo < m < hi
    m2, lo2, hi2 = bootstrap_ci(np.tile(x, 10), n_resamples=4000, seed=1)
    assert (hi2 - lo2) < (hi - lo)
    assert bootstrap_ci([3.0]) == (3.0, 3.0, 3.0)


def test_paired_permutation_exact():
    # all 5 differences positive: only the all-plus and all-minus flips are as extreme -> p = 2/32
    assert paired_permutation_test([2, 3, 4, 5, 6], [1, 1, 1, 1, 1]) == pytest.approx(2 / 32)
    assert paired_permutation_test([1, 2, 3], [1, 2, 3]) == 1.0
    res = paired_comparison([1, 1, 1, 1], [2, 2, 2, 2])
    assert res["mean_diff"] == pytest.approx(-1.0) and res["a_lower_count"] == 4


def test_holm_correction():
    adj = holm_correction([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_env_metrics_aggregate_trains(cfg, corridor, classes):
    """Episode metrics equal the sums over the per-train records."""
    from agents.baselines import PassengerFirstPolicy, run_policy_episode
    from sim.env import CorridorEnv
    from sim.timetable import generate_timetable
    from sim.trains import EXITED

    env = CorridorEnv(cfg, generate_timetable(4, cfg, corridor, classes), corridor)
    m = run_policy_episode(env, PassengerFirstPolicy())
    p = [t for t in env.trains if t.cls.is_passenger]
    f = [t for t in env.trains if not t.cls.is_passenger]
    assert m["passenger_delay"] == pytest.approx(sum(t.accrued for t in p))
    assert m["freight_delay"] == pytest.approx(sum(t.accrued for t in f))
    assert m["trains_cleared"] == sum(t.status == EXITED for t in env.trains)
    assert sum(m["section_passenger_delay"]) == pytest.approx(m["passenger_delay"])
    expected_arr = sum(max(0.0, a - t.sched.sched_station_min[i]) for t in p for i, a in enumerate(t.arrivals) if a >= 0)
    assert m["station_arrival_delay"] == pytest.approx(expected_arr)
    assert m["loop_uses"] == sum(m["loop_uses_by_station"])
    # reward bookkeeping: total shared return = -(w_p * weighted passenger + w_f * freight) - penalties
    w_p, w_f = cfg["reward"]["w_p"], cfg["reward"]["w_f"]
    pen = cfg["control"]["invalid_penalty"] * m["invalid_actions"]
    assert m["shared_return"] == pytest.approx(-(w_p * m["passenger_delay_weighted"] + w_f * m["freight_delay"]) - pen)


def test_level_labels_match_encoding(cfg):
    """UI labels come from the discretiser; every feature has one label per level and level 0 of position is the loop."""
    from agents.discretize import build_discretizer
    from sim.config import with_overrides

    for c in (cfg, with_overrides(cfg, {"discretization.use_downstream_signal": False})):
        d = build_discretizer(c)
        labels = d.level_labels()
        assert list(labels) == list(d.feature_names)
        assert [len(labels[f]) for f in d.feature_names] == list(d.radices)
        assert labels["position"][0] == "in loop"
        for i in (0, d.n_states // 2, d.n_states - 1):
            lv = d.decode(i)
            assert all(0 <= lv[f] < len(labels[f]) for f in d.feature_names)
