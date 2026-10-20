"""User scenarios: the inputs a visitor can change, and how they map onto config.yaml.

A scenario has two parts:
- traffic and scoring (load, passenger share and mix, lateness, jitter, reward weights). These define which days
  the corridor sees and how they are scored, so every page of the app is computed per traffic key;
- day selection (seed, model run), which only the Simulate page uses.

The default scenario reproduces config.yaml exactly, so its pages are the precomputed results in results/.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from typing import Any

from experiments.common import Context
from sim.config import load_config, with_overrides

# Timetables are calibrated into [load - w, load + w]; 0.15 makes the default load (1.35) reproduce the
# configured training band [1.20, 1.50].
BAND_HALF_WIDTH = 0.15

TRAFFIC_FIELDS = ("load", "passenger_share", "mix_vb", "mix_raj", "mix_exp", "late_prob", "late_mean_min",
                  "jitter_min", "w_p", "w_f")

# name: (default, min, max, round-to digits). Defaults mirror config.yaml.
LIMITS: dict[str, tuple[float, float, float, int]] = {
    "load": (1.35, 0.6, 1.9, 3),
    "passenger_share": (0.6, 0.1, 0.9, 3),
    "mix_vb": (0.2, 0.0, 1.0, 3),
    "mix_raj": (0.3, 0.0, 1.0, 3),
    "mix_exp": (0.5, 0.0, 1.0, 3),
    "late_prob": (0.3, 0.0, 1.0, 3),
    "late_mean_min": (12.0, 0.0, 60.0, 2),
    "jitter_min": (4.0, 0.0, 20.0, 2),
    "w_p": (10.0, 0.0, 50.0, 3),
    "w_f": (1.0, 0.0, 10.0, 3),
}


class ScenarioError(ValueError):
    """The inputs describe something the generator or the trained models cannot provide."""


@dataclass(frozen=True)
class Scenario:
    """All user inputs. Use :func:`make_scenario` to build one with validation and canonical rounding."""

    seed: int = 1
    load: float = 1.35
    passenger_share: float = 0.6
    mix_vb: float = 0.2
    mix_raj: float = 0.3
    mix_exp: float = 0.5
    late_prob: float = 0.3
    late_mean_min: float = 12.0
    jitter_min: float = 4.0
    model_run: int = 0
    w_p: float = 10.0
    w_f: float = 1.0

    def traffic(self) -> dict[str, float]:
        """The part of the scenario that defines the days and their scoring."""
        return {k: getattr(self, k) for k in TRAFFIC_FIELDS}


def make_scenario(**kw: Any) -> Scenario:
    """Validate ranges and round floats so equal inputs always produce the same key."""
    out: dict[str, Any] = {}
    for f in fields(Scenario):
        if f.name not in kw or kw[f.name] is None:
            continue
        v = kw[f.name]
        if f.name in LIMITS:
            _, lo, hi, nd = LIMITS[f.name]
            v = float(v)
            if not lo <= v <= hi:
                raise ScenarioError(f"{f.name} must be between {lo} and {hi}")
            v = round(v, nd)
        else:
            v = int(v)
        out[f.name] = v
    sc = Scenario(**out)
    if sc.mix_vb + sc.mix_raj + sc.mix_exp <= 0:
        raise ScenarioError("the passenger mix needs at least one class with a positive share")
    if not 0 <= sc.seed <= 2**31 - 1:
        raise ScenarioError("seed must be between 0 and 2147483647")
    return sc


@lru_cache(maxsize=1)
def base() -> tuple[dict[str, Any], Context]:
    """config.yaml and its context (corridor, classes, discretiser)."""
    cfg = load_config()
    return cfg, Context(cfg)


def scenario_config(sc: Scenario) -> dict[str, Any]:
    """config.yaml with the scenario's traffic and scoring. Learning and state settings stay as configured."""
    cfg, _ = base()
    w = BAND_HALF_WIDTH
    return with_overrides(cfg, {
        "timetable.target_utilization": [round(sc.load - w, 6), round(sc.load + w, 6)],
        "timetable.passenger_share": sc.passenger_share,
        "timetable.passenger_mix": {"VB": sc.mix_vb, "RAJ": sc.mix_raj, "EXP": sc.mix_exp},
        "timetable.initial_delay_prob": sc.late_prob,
        "timetable.initial_delay_mean_min": sc.late_mean_min,
        "timetable.departure_jitter_min": sc.jitter_min,
        "reward.w_p": sc.w_p,
        "reward.w_f": sc.w_f,
    })


def is_default(sc: Scenario) -> bool:
    """True when the scenario's traffic and scoring are exactly config.yaml (precomputed results apply)."""
    cfg, _ = base()
    c = scenario_config(sc)
    return c["timetable"] == cfg["timetable"] and c["reward"] == cfg["reward"]


def traffic_key(sc: Scenario) -> str:
    """Stable identifier of the scenario's traffic and scoring ("default" for config.yaml)."""
    if is_default(sc):
        return "default"
    blob = json.dumps(sc.traffic(), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def describe(sc: Scenario) -> dict[str, Any]:
    """JSON form returned with every scenario-dependent response."""
    return {"key": traffic_key(sc), "is_default": is_default(sc), "traffic": sc.traffic(), "all": asdict(sc)}
