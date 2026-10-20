"""Simulator answers checked against hand calculations, and live-simulation output checked for consistency."""
from __future__ import annotations

import numpy as np
import pytest

from agents.baselines import FCFSPolicy, GreedyQPolicy, PassengerFirstPolicy, run_policy_episode
from experiments.common import Context, model_path
from sim.env import HOLD, PROCEED, CorridorEnv
from sim.timetable import Timetable, free_run_station_times
from sim.trains import ScheduledTrain


@pytest.fixture(scope="module")
def ctx(cfg):
    return Context(cfg)


def _tt(ctx, *specs):
    k, c = ctx.classes, ctx.corridor
    trains = tuple(ScheduledTrain(tid=i, cls=k[code], sched_entry_min=t0, ready_min=t0,
                                  sched_station_min=tuple(t0 + x for x in free_run_station_times(k[code], c)))
                   for i, (code, t0) in enumerate(specs))
    return Timetable(seed=0, trains=trains, utilization=0.0, section_utilization=(), window_min=720.0)


def _run(cfg, ctx, tt, policy):
    env = CorridorEnv(cfg, tt, ctx.corridor)
    return env, run_policy_episode(env, policy)


def test_lone_trains_run_exactly_to_the_hand_computed_time(cfg, ctx):
    env, _ = _run(cfg, ctx, _tt(ctx, ("VB", 0.0)), FCFSPolicy())
    vb = env.trains[0]
    assert vb.exit_min == pytest.approx(870 / 150 * 60 + 6 * 2, abs=1e-6)       # 348 min running + 6 x 2 min dwell
    assert vb.arrivals[0] == pytest.approx(30 / 150 * 60, abs=1e-6)              # CNB at km 30
    assert vb.accrued == pytest.approx(0.0, abs=1e-9)
    env, _ = _run(cfg, ctx, _tt(ctx, ("FRT", 60.0)), FCFSPolicy())
    assert env.trains[0].exit_min == pytest.approx(60 + 870, abs=1e-6)          # 870 km at 60 km/h
    assert env.trains[0].accrued == pytest.approx(0.0, abs=1e-9)


def test_passenger_trapped_behind_freight_without_loops(cfg, ctx):
    env, _ = _run(cfg, ctx, _tt(ctx, ("FRT", 0.0), ("EXP", 20.0)), FCFSPolicy())
    f, p = env.trains
    assert p.exit_min > f.exit_min                                    # cannot overtake on the main line
    assert p.exit_min - f.exit_min < 10                               # follows at about one block
    ideal = 870 / 110 * 60 + 6 * 3
    assert p.accrued == pytest.approx(p.exit_min - 20 - ideal, abs=1.0)
    assert f.accrued == pytest.approx(0.0, abs=1e-9)


def test_looping_moves_delay_from_passenger_to_freight(cfg, ctx):
    tt = _tt(ctx, ("FRT", 0.0), ("EXP", 20.0))
    _, fcfs = _run(cfg, ctx, tt, FCFSPolicy())
    env, pf = _run(cfg, ctx, tt, PassengerFirstPolicy())
    f, p = env.trains
    assert pf["loop_uses"] >= 1 and p.exit_min < f.exit_min          # overtook in a loop
    assert pf["passenger_delay"] < 0.05 * fcfs["passenger_delay"]
    assert pf["freight_delay"] > fcfs["freight_delay"]


@pytest.mark.skipif(not model_path("main", "iql", 0).is_file(), reason="trained models missing")
@pytest.mark.parametrize("sc_kwargs", [{}, {"load": 0.7, "passenger_share": 0.3}, {"load": 1.85, "late_prob": 1.0},
                                       {"mix_vb": 1, "mix_raj": 0, "mix_exp": 0, "w_p": 3, "w_f": 7, "model_run": 4}])
def test_live_simulation_output_is_consistent(sc_kwargs):
    from api.scenario import base, make_scenario, scenario_config
    from api.simulate import POLICIES, _q_tables, simulate
    from experiments.common import MODELS

    sc = make_scenario(**sc_kwargs)
    r = simulate(sc, str(MODELS))
    _, ctx = base()
    for name in POLICIES:
        rp, mt = r["replays"][name], r["replays"][name]["metrics"]
        ts, dec = rp["trains"], rp["decisions"]
        assert abs(sum(t["delay"] for t in ts if t["kind"] == "passenger") - mt["passenger_delay"]) <= 0.05 * len(ts) + 0.01
        assert abs(sum(t["delay"] for t in ts if t["kind"] == "freight") - mt["freight_delay"]) <= 0.05 * len(ts) + 0.01
        assert sum(t["status"] == "exited" for t in ts) == mt["trains_cleared"]
        assert sum(e[3] == "loop_in" for e in rp["events"]) == mt["loop_uses"]
        assert sum(not d["valid"] for d in dec) == mt["invalid_actions"]
        score = -(sc.w_p * mt["passenger_delay_weighted"] + sc.w_f * mt["freight_delay"]) - 100.0 * mt["invalid_actions"]
        assert abs(score - mt["shared_return"]) <= 0.005 * (sc.w_p + sc.w_f + 1) + 1e-6
        if name == "fcfs":
            assert mt["loop_uses"] == 0 and all(d["action"] == PROCEED for d in dec)
        if name == "passenger_first":
            assert all(d["action"] != HOLD and d["valid"] for d in dec)
    # trained agents act exactly as argmax of the selected run's Q-table
    cfg = scenario_config(sc)
    from sim.timetable import generate_timetable
    tt = generate_timetable(sc.seed, cfg, ctx.corridor, ctx.classes)
    for algo in ("iql", "sarsa"):
        q = _q_tables(algo, sc.model_run, str(MODELS))
        pol = GreedyQPolicy(algo, list(q), ctx.disc)
        env = CorridorEnv(cfg, tt, ctx.corridor)
        while not env.done:
            obs = env.decision_requests()
            acts = pol.actions(env, obs) if obs else {}
            for o in obs:
                assert acts[o.agent] == int(np.argmax(q[o.agent][ctx.disc.encode(o)]))
            env.step(acts)
        assert env.metrics()["passenger_delay"] == pytest.approx(r["replays"][algo]["metrics"]["passenger_delay"], abs=0.01)
