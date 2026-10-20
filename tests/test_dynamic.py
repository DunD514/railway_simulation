"""Scenario-dependent pages: keys, evaluation and training per scenario, and the API around them."""
from __future__ import annotations

import json
import shutil
from multiprocessing import get_context

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.scenario import is_default, make_scenario, scenario_config, traffic_key
from experiments.common import MODELS, RESULTS, model_path
from experiments.evaluate import evaluate_all
from experiments.train import train_all

pytestmark = pytest.mark.skipif(not model_path("main", "iql", 1).is_file(), reason="trained models missing")


@pytest.fixture(scope="module")
def pool():
    with get_context("spawn").Pool(4) as p:
        yield p


@pytest.fixture(scope="module")
def two_run_models(tmp_path_factory):
    """A copy of the first two trained main runs (keeps evaluation fast)."""
    d = tmp_path_factory.mktemp("models")
    for algo in ("iql", "sarsa"):
        for run in (0, 1):
            dst = model_path("main", algo, run, d)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(model_path("main", algo, run, MODELS), dst)
    return d


def _small(cfg):
    cfg["experiment"]["bootstrap_resamples"] = 500
    cfg["experiment"]["eval_timetables_per_seed"] = 2
    return cfg


def test_default_scenario_is_config_and_keys_follow_traffic():
    sc = make_scenario()
    assert is_default(sc) and traffic_key(sc) == "default"
    assert scenario_config(sc)["timetable"]["target_utilization"] == [1.2, 1.5]
    assert traffic_key(make_scenario(seed=77, model_run=3)) == "default"  # day choice is not traffic
    keys = {traffic_key(make_scenario(**{k: v})) for k, v in
            [("load", 1.4), ("passenger_share", 0.5), ("mix_vb", 0.5), ("late_prob", 0.1), ("late_mean_min", 20),
             ("jitter_min", 8), ("w_p", 5), ("w_f", 2)]}
    assert len(keys) == 8 and "default" not in keys


def test_scenario_evaluation_is_deterministic_and_follows_traffic(tmp_path, pool, two_run_models):
    heavy = _small(scenario_config(make_scenario(load=1.75, passenger_share=0.8)))
    light = _small(scenario_config(make_scenario(load=0.8)))
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for out, cfg in ((a, heavy), (b, heavy), (c, light)):
        evaluate_all(cfg, out, two_run_models, pool, log=lambda m: None)
    for f in ("eval_main.json", "behavior.json", "overview.json", "replays/index.json"):
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    ea, ec = (json.loads((d / "eval_main.json").read_text()) for d in (a, c))
    assert ea["protocol"]["n_runs"] == 2
    assert ea["policies"]["fcfs"]["passenger_delay"]["mean"] > ec["policies"]["fcfs"]["passenger_delay"]["mean"]
    ov = json.loads((a / "overview.json").read_text())
    assert abs(ov["timetable"]["utilization_mean"] - 1.75) <= 0.15
    # models without ablation / sweep variants are reported as unavailable, not silently mixed in
    assert json.loads((a / "ablations.json").read_text())["available"] is False
    assert json.loads((a / "sensitivity.json").read_text())["available"] is False


def test_training_on_a_scenario_is_deterministic_and_uses_its_traffic(tmp_path, pool):
    def cfg_for(**kw):
        cfg = scenario_config(make_scenario(**kw))
        cfg["experiment"]["train_episodes"] = 20
        cfg["experiment"]["n_seeds"] = 1
        return cfg

    for name, cfg in (("x", cfg_for(load=1.7)), ("y", cfg_for(load=1.7)), ("z", cfg_for(load=0.8))):
        train_all(cfg, tmp_path / name, pool, ablations=False, sweep=False, log=lambda m: None)
    q = {n: np.load(model_path("main", "iql", 0, tmp_path / n / "models"))["q"] for n in "xyz"}
    assert np.array_equal(q["x"], q["y"])
    assert not np.array_equal(q["x"], q["z"])
    cs = json.loads((tmp_path / "x" / "curves_summary.json").read_text())
    assert cs["episodes"] == 20 and set(cs["variants"]) == {"main"}


def test_api_scenario_states():
    from api.main import app

    client = TestClient(app)
    # default scenario: precomputed files, served directly
    r = client.get("/api/page/eval")
    assert r.status_code == 200 and r.json() == json.loads((RESULTS / "eval_main.json").read_text())
    # untrained scenario asks for training instead of failing
    q = "load=1.83&passenger_share=0.42"
    body = client.get(f"/api/page/eval?{q}&agents=trained").json()
    assert body["status"] == "needs_training"
    assert client.get(f"/api/simulate?{q}&agents=trained").json()["status"] == "needs_training"
    st = client.get(f"/api/scenario?{q}").json()
    assert st["scenario"]["key"] != "default" and st["trained"] is None
    # training: default scenario refused, budget validated
    assert client.post("/api/train").status_code == 409
    assert client.post(f"/api/train?{q}&episodes=5000").status_code == 422
    assert client.post(f"/api/train?{q}&seeds=0").status_code == 422
    est = client.get(f"/api/train/estimate?{q}&episodes=50&seeds=2").json()
    assert est["episodes"] == 50 * (2 * 2 + 2 * 2) and est["seconds"] > 0
