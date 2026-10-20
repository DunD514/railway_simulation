"""Simulator tests: block exclusion, invalid-loop penalty, utilization band, reproducibility."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from agents.baselines import FCFSPolicy, PassengerFirstPolicy, RandomPolicy, run_policy_episode
from sim.env import HOLD, LOOP, CorridorEnv
from sim.timetable import Timetable, compute_utilization, generate_timetable, pair_headway
from sim.trains import EXITED, IN_LOOP, WAITING, ScheduledTrain


def _check_blocks(env: CorridorEnv) -> None:
    claimed: dict[int, int] = {}
    for tr in env.trains:
        if tr.status in (WAITING, EXITED):
            continue
        blocks = []
        if tr.status != IN_LOOP or env.occ[tr.block] == tr.tid:
            blocks.append(tr.block)
        if tr.tail_block >= 0:
            blocks.append(tr.tail_block)
        for b in blocks:
            assert b not in claimed, f"block {b} held by {claimed[b]} and {tr.tid}"
            claimed[b] = tr.tid
            assert env.occ[b] == tr.tid
    # every occupied block belongs to exactly one train
    for b, tid in enumerate(env.occ):
        if tid != -1:
            assert claimed.get(b) == tid, f"block {b} marked {tid} but not claimed"
    for st, occ in enumerate(env.loop_occ):
        assert len(occ) <= env.corridor.loop_capacity


@pytest.mark.parametrize("policy_name", ["fcfs", "pf", "random"])
def test_two_trains_never_share_a_block(cfg, corridor, classes, policy_name):
    rng = np.random.default_rng(3)
    policy = {"fcfs": FCFSPolicy(), "pf": PassengerFirstPolicy(), "random": RandomPolicy(rng)}[policy_name]
    for seed in (11, 12):
        env = CorridorEnv(cfg, generate_timetable(seed, cfg, corridor, classes), corridor)
        original = env._substep

        def checked() -> None:
            original()
            _check_blocks(env)

        env._substep = checked  # type: ignore[method-assign]
        while not env.done:
            obs = env.decision_requests()
            env.step(policy.actions(env, obs))
        assert env.metrics()["trains_cleared"] > 0


def test_head_positions_respect_block_order(cfg, corridor, classes):
    """Followers never overtake on the main line: their head stays behind the leader's block."""
    env = CorridorEnv(cfg, generate_timetable(5, cfg, corridor, classes), corridor)
    pol = PassengerFirstPolicy()
    while not env.done:
        env.step(pol.actions(env, env.decision_requests()))
        heads = [tr.block for tr in env.trains if tr.status not in (WAITING, EXITED, IN_LOOP)]
        assert len(heads) == len(set(heads))


def _find_busy_loop_decision(cfg, corridor, classes):
    pol = PassengerFirstPolicy()
    for seed in range(40):
        env = CorridorEnv(cfg, generate_timetable(seed, cfg, corridor, classes), corridor)
        while not env.done:
            obs = env.decision_requests()
            for o in obs:
                if not o.in_loop and o.loop_busy:
                    return env, o
            env.step(pol.actions(env, obs))
    raise AssertionError("no busy-loop decision found")


def test_invalid_loop_is_blocked_and_penalized(cfg, corridor, classes):
    env, o = _find_busy_loop_decision(cfg, corridor, classes)
    assert not env.loop_valid(o)
    others = {x.agent: HOLD for x in env.decision_requests() if x.agent != o.agent}
    env_loop = copy.deepcopy(env)
    env_hold = copy.deepcopy(env)
    r_loop = env_loop.step({**others, o.agent: LOOP})
    r_hold = env_hold.step({**others, o.agent: HOLD})
    penalty = cfg["control"]["invalid_penalty"]
    assert r_loop.invalid_agents == [o.agent]
    assert r_hold.invalid_agents == []
    # the refused LOOP behaves exactly like HOLD apart from the penalty
    assert r_loop.shared_reward == pytest.approx(r_hold.shared_reward - penalty)
    assert r_loop.local_rewards[o.agent] == pytest.approx(r_hold.local_rewards[o.agent] - penalty)
    f = env_loop.trains[o.freight_tid]
    assert f.route_loop == -1 and f.tid not in env_loop.loop_occ[o.agent]
    assert env_loop.metrics()["invalid_actions"] == env.metrics()["invalid_actions"] + 1


def test_utilization_within_target_band(cfg, corridor, classes):
    lo, hi = cfg["timetable"]["target_utilization"]
    for seed in range(60):
        tt = generate_timetable(seed, cfg, corridor, classes)
        assert lo <= tt.utilization <= hi
        # recompute independently from the class order
        u, _ = compute_utilization([t.cls for t in tt.trains], corridor, tt.window_min, cfg["corridor"]["block_km"])
        assert u == pytest.approx(tt.utilization)
        ratio = tt.n_passenger / max(1, tt.n_freight)
        assert 1.2 <= ratio <= 1.8  # roughly 3:2


def test_pair_headway_hand_computed(classes):
    frt, vb = classes["FRT"], classes["VB"]
    # freight leader, Vande Bharat follower over 200 km with 10 km blocks:
    # 200/1 - 200/2.5 = 120 min running-time gap + (10+1.2)/1 + 10/2.5 = 15.2 min clearance
    assert pair_headway(frt, vb, 200.0, 10.0) == pytest.approx(135.2)
    # faster leader: clearance only, (10+0.6)/2.5 + 10/1 = 14.24
    assert pair_headway(vb, frt, 200.0, 10.0) == pytest.approx(14.24)


def test_identical_seeds_reproduce_identical_episodes(cfg, corridor, classes):
    def run(seed: int, pol_seed: int):
        env = CorridorEnv(cfg, generate_timetable(seed, cfg, corridor, classes), corridor, record=True)
        m = run_policy_episode(env, RandomPolicy(np.random.default_rng(pol_seed)))
        return m, env.log.frames, env.log.decisions

    a = run(21, 7)
    b = run(21, 7)
    assert a[0] == b[0]
    assert a[1] == b[1]
    assert a[2] == b[2]
    c = run(22, 7)
    assert c[1] != a[1]


def test_single_train_runs_on_schedule(cfg, corridor, classes):
    """A lone train meets no conflicts: zero accrued delay, arrivals exactly on the free-run schedule."""
    from sim.timetable import free_run_station_times

    raj = classes["RAJ"]
    st_times = tuple(free_run_station_times(raj, corridor))
    tt = Timetable(seed=0, trains=(ScheduledTrain(0, raj, 0.0, 0.0, st_times),), utilization=0.0,
                   section_utilization=(), window_min=720.0)
    env = CorridorEnv(cfg, tt, corridor)
    m = run_policy_episode(env, FCFSPolicy())
    assert m["passenger_delay"] == pytest.approx(0.0, abs=1e-6)
    assert m["station_arrival_delay"] == pytest.approx(0.0, abs=1e-6)
    assert m["trains_cleared"] == 1
    expected_exit = st_times[-1] + raj.dwell_min + corridor_exit_km(corridor) / raj.speed_kpm
    assert env.trains[0].exit_min == pytest.approx(expected_exit, abs=1e-6)


def corridor_exit_km(corridor) -> float:
    return corridor.length_km - corridor.stations[-1].km
