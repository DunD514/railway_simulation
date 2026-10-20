"""Live simulation endpoint: determinism, input validation, and that inputs actually change the scenario."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api import simulate as sim_mod
from api.main import app

client = TestClient(app)
pytestmark = pytest.mark.skipif(not sim_mod.model_path("main", "iql", 0).is_file(), reason="trained models missing")


def _get(**params):
    return client.get("/api/simulate", params=params)


def test_same_inputs_same_output_even_without_cache():
    a = _get(seed=11, load=1.3).json()
    sim_mod.simulate.cache_clear()
    b = _get(seed=11, load=1.3).json()
    assert a == b


def test_every_policy_runs_the_same_timetable():
    r = _get(seed=3).json()
    tts = [[(t["label"], t["ready"]) for t in r["replays"][p]["trains"]] for p in r["policies"]]
    assert all(tt == tts[0] for tt in tts)
    assert r["timetable"]["n_trains"] == len(tts[0])


def test_inputs_change_the_scenario():
    base = _get(seed=5, load=1.0, passenger_share=0.5).json()["timetable"]
    busier = _get(seed=5, load=1.7, passenger_share=0.5).json()["timetable"]
    more_pax = _get(seed=5, load=1.0, passenger_share=0.9).json()["timetable"]
    assert busier["utilization"] > base["utilization"] + 0.4
    assert more_pax["n_passenger"] / more_pax["n_trains"] > base["n_passenger"] / base["n_trains"]
    assert _get(seed=6).json()["timetable"] != _get(seed=7).json()["timetable"]


def test_scoring_weights_only_change_the_score():
    a = _get(seed=2, w_p=10, w_f=1).json()["replays"]["iql"]["metrics"]
    b = _get(seed=2, w_p=1, w_f=5).json()["replays"]["iql"]["metrics"]
    assert a["passenger_delay"] == b["passenger_delay"] and a["freight_delay"] == b["freight_delay"]
    assert a["shared_return"] != b["shared_return"]


def test_utilisation_lands_in_requested_band():
    for load in (0.7, 1.35, 1.8):
        u = _get(seed=4, load=load).json()["timetable"]["utilization"]
        assert abs(u - load) <= sim_mod.BAND_HALF_WIDTH + 1e-9


@pytest.mark.parametrize("params", [{"load": 3.0}, {"passenger_share": 0.0}, {"model_run": 10}, {"seed": -1},
                                    {"late_prob": 1.5}])
def test_out_of_range_inputs_rejected(params):
    assert _get(**params).status_code == 422


def test_empty_passenger_mix_rejected():
    r = _get(mix_vb=0, mix_raj=0, mix_exp=0)
    assert r.status_code == 422 and "mix" in r.json()["detail"]
