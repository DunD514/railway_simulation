import json
import urllib.request
import pytest

BASE_URL = "http://localhost:8000"

def test_live_simulation_real_physics():
    """Verify simulation results against real corridor physics and safety rules."""
    req = urllib.request.urlopen(f"{BASE_URL}/api/simulate?seed=900000000")
    assert req.status == 200
    data = json.loads(req.read())

    tt = data["timetable"]
    assert tt["seed"] == 900000000
    assert tt["n_trains"] == tt["n_passenger"] + tt["n_freight"]
    assert tt["n_trains"] > 0
    print(f"\n[PASS] Timetable Seed 900000000: {tt['n_trains']} Trains ({tt['n_passenger']} Pax, {tt['n_freight']} Frt)")

    # Verify every policy output
    assert set(data["policies"]) == {"fcfs", "passenger_first", "iql", "sarsa"}
    for pol in data["policies"]:
        rp = data["replays"][pol]
        m = rp["metrics"]

        # Physics integrity
        assert m["trains_cleared"] <= m["n_trains"]
        assert m["passenger_delay"] >= 0
        assert m["freight_delay"] >= 0
        assert "shared_return" in m

        # Passenger Safety: Passenger trains NEVER enter a loop
        pax_trains = [t for t in rp["trains"] if t["kind"] == "passenger"]
        for t in pax_trains:
            assert t["status"] != "loop", f"Safety violation: Passenger train {t['tid']} in loop under {pol}!"

        # Interlocking integrity
        for d in rp["decisions"]:
            assert d["action"] in (0, 1, 2)  # PROCEED, LOOP, HOLD
            assert isinstance(d["valid"], bool)

        print(f"[PASS] Policy {pol.upper()}: Pax Delay={m['passenger_delay']:.1f}m, Frt Delay={m['freight_delay']:.1f}m, Cleared={m['trains_cleared']}/{m['n_trains']}")

def test_dynamic_scenario_inputs():
    """Verify that changing inputs (load, passenger share) changes simulator outputs dynamically."""
    # Run 1: High load (1.60), high passenger share (0.80)
    req1 = urllib.request.urlopen(f"{BASE_URL}/api/simulate?seed=500&load=1.60&passenger_share=0.80")
    data1 = json.loads(req1.read())
    tt1 = data1["timetable"]

    # Run 2: Low load (0.70), low passenger share (0.20)
    req2 = urllib.request.urlopen(f"{BASE_URL}/api/simulate?seed=500&load=0.70&passenger_share=0.20")
    data2 = json.loads(req2.read())
    tt2 = data2["timetable"]

    # Dynamic reaction checks
    assert tt1["n_trains"] > tt2["n_trains"], "Higher load must yield more trains!"
    assert tt1["n_passenger"] > tt1["n_freight"], "80% pax share must yield more pax than freight!"
    assert tt2["n_passenger"] < tt2["n_freight"], "20% pax share must yield more freight than pax!"
    print(f"\n[PASS] Dynamic Load & Mix: High Load ({tt1['n_trains']} trains) vs Low Load ({tt2['n_trains']} trains)")

def test_real_evaluation_ground_truth():
    """Verify that the evaluation endpoint returns real policy distributions."""
    req = urllib.request.urlopen(f"{BASE_URL}/api/page/eval?agents=default")
    assert req.status == 200
    data = json.loads(req.read())

    policies = data["policies"]
    assert "iql" in policies
    assert "fcfs" in policies
    assert "passenger_first" in policies

    # IQL should significantly outperform FCFS on return
    iql_ret = policies["iql"]["shared_return"]["mean"]
    fcfs_ret = policies["fcfs"]["shared_return"]["mean"]
    assert iql_ret > fcfs_ret, f"IQL ({iql_ret}) must beat FCFS ({fcfs_ret})!"
    print(f"\n[PASS] RL Outperformed Baseline: IQL Return ({iql_ret:.0f}) > FCFS Return ({fcfs_ret:.0f})")
