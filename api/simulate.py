"""Live simulation of one day of a user scenario (GET /api/simulate).

The day's timetable is generated from the scenario exactly as in training and evaluation, then every policy
(FCFS, passenger-first and the trained IQL / SARSA agents of one training run) dispatches it. Everything is
seeded, so identical inputs give identical outputs.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from agents.baselines import FCFSPolicy, GreedyQPolicy, PassengerFirstPolicy, run_policy_episode
from api.scenario import BAND_HALF_WIDTH, Scenario, ScenarioError, base, describe, scenario_config  # noqa: F401
from experiments.common import MODELS, model_path
from experiments.evaluate import POLICY_LABELS, available_runs, episode_payload
from sim.env import CorridorEnv
from sim.timetable import generate_timetable

POLICIES = ("fcfs", "passenger_first", "iql", "sarsa")


@lru_cache(maxsize=64)
def _q_tables(algo: str, run: int, models_dir: str) -> tuple[np.ndarray, ...]:
    path = model_path("main", algo, run, Path(models_dir))
    if not path.is_file():
        raise ScenarioError(f"no trained {algo.upper()} model for run {run}")
    return tuple(np.load(path)["q"])


def _policy(name: str, sc: Scenario, models_dir: Path):
    if name == "fcfs":
        return FCFSPolicy()
    if name == "passenger_first":
        return PassengerFirstPolicy()
    _, ctx = base()
    return GreedyQPolicy(name, list(_q_tables(name, sc.model_run, str(models_dir))), ctx.disc)


@lru_cache(maxsize=64)
def simulate(sc: Scenario, models_dir: str | None = None, agents: str = "default") -> dict[str, Any]:
    """Run every policy on the scenario's day and return metrics plus a replay per policy."""
    cfg_base, ctx = base()
    mdir = Path(models_dir) if models_dir else MODELS
    n_runs = available_runs(mdir, "main")
    if sc.model_run >= n_runs:
        raise ScenarioError(f"these agents have {n_runs} trained run(s); choose a model run below {n_runs}")
    cfg = scenario_config(sc)
    try:
        tt = generate_timetable(sc.seed, cfg, ctx.corridor, ctx.classes)
    except RuntimeError as e:
        raise ScenarioError(f"could not build a timetable at load {sc.load:.2f} with these settings ({e})") from e
    replays: dict[str, Any] = {}
    for name in POLICIES:
        env = CorridorEnv(cfg, tt, ctx.corridor, record=True)
        m = run_policy_episode(env, _policy(name, sc, mdir))
        replays[name] = episode_payload(env, tt, name, sc.seed, m)
    lo, hi = cfg_base["timetable"]["target_utilization"]
    return {
        "scenario": {**describe(sc)["all"]},
        "scenario_key": describe(sc)["key"],
        "agents": agents,
        "n_model_runs": n_runs,
        "timetable": {
            "seed": sc.seed,
            "n_trains": len(tt.trains),
            "n_passenger": tt.n_passenger,
            "n_freight": tt.n_freight,
            "utilization": tt.utilization,
            "late_trains": sum(1 for t in tt.trains if t.ready_min - t.sched_entry_min > sc.jitter_min + 1e-9),
        },
        "trained_on": {"load": [lo, hi], "passenger_share": cfg_base["timetable"]["passenger_share"]},
        "policies": list(POLICIES),
        "labels": {k: POLICY_LABELS[k] for k in POLICIES},
        "replays": replays,
    }
