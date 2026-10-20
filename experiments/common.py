"""Shared experiment plumbing: seeds, variants, learner construction, result paths."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from agents.discretize import build_discretizer
from agents.iql import QLearningAgent
from agents.sarsa import SarsaAgent
from agents.tabular import IndependentLearners
from sim.config import ROOT, with_overrides
from sim.corridor import build_corridor
from sim.timetable import Timetable, generate_timetable
from sim.trains import load_train_classes

RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
ALGOS = {"iql": QLearningAgent, "sarsa": SarsaAgent}


@dataclass(frozen=True)
class Variant:
    """A named training configuration (config overrides on top of config.yaml)."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)
    label: str = ""


def main_variants() -> list[Variant]:
    """Main experiment and the two ablations."""
    return [
        Variant("main", {}, "shared reward, with downstream signal"),
        Variant("local", {"reward.mode": "local"}, "local-only reward (own-section delay)"),
        Variant("nosignal", {"discretization.use_downstream_signal": False}, "no downstream congestion signal"),
    ]


def sweep_variants(cfg: dict[str, Any]) -> list[Variant]:
    """Hyperparameter sweep cells: alpha x gamma grid plus epsilon schedules."""
    sw = cfg["experiment"]["sweep"]
    out: list[Variant] = []
    for a in sw["alphas"]:
        for g in sw["gammas"]:
            out.append(Variant(f"sweep_a{a}_g{g}", {"learning.alpha": a, "learning.gamma": g}, f"alpha={a}, gamma={g}"))
    for name, sched in sw["epsilon_schedules"].items():
        out.append(Variant(f"eps_{name}", {"learning.epsilon": sched}, f"epsilon schedule '{name}'"))
    return out


def is_default_cell(cfg: dict[str, Any], v: Variant) -> bool:
    """True if a sweep cell reproduces the main configuration exactly (so main runs can be reused)."""
    eff = with_overrides(cfg, v.overrides)
    return eff["learning"] == cfg["learning"] and eff["reward"] == cfg["reward"] and \
        eff["discretization"] == cfg["discretization"]


class Context:
    """Per-process cache of corridor, classes and discretizer for a given config."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.corridor = build_corridor(cfg)
        self.classes = load_train_classes(cfg)
        self.disc = build_discretizer(cfg)

    def timetable(self, seed: int) -> Timetable:
        """Generate the timetable for a seed."""
        return generate_timetable(seed, self.cfg, self.corridor, self.classes)


def train_timetable_seed(cfg: dict[str, Any], run: int, episode: int) -> int:
    """Training timetable seed for run ``run`` and episode ``episode`` (never overlaps eval seeds)."""
    return int(cfg["experiment"]["train_timetable_base"]) * (run + 1) + episode


def eval_timetable_seeds(cfg: dict[str, Any], run: int) -> list[int]:
    """Held-out timetable seeds assigned to run ``run`` (disjoint between runs and from training)."""
    base = int(cfg["experiment"]["eval_timetable_base"])
    k = int(cfg["experiment"]["eval_timetables_per_seed"])
    return [base + run * 1000 + j for j in range(k)]


def run_rng_seed(cfg: dict[str, Any], run: int, salt: int = 0) -> int:
    """Seed for the learner / policy RNG of a run."""
    return int(np.random.SeedSequence([int(cfg["seed"]), run, salt]).generate_state(1)[0])


def make_learners(algo: str, cfg: dict[str, Any], ctx: Context) -> IndependentLearners:
    """Build N independent tabular learners for the given algorithm and config."""
    lc = cfg["learning"]
    cls = ALGOS[algo]
    n_states = ctx.disc.n_states
    q0 = q_init_value(cfg, ctx.corridor.n_stations)
    return IndependentLearners(
        lambda: cls(n_states, lc["alpha"], lc["gamma"], q0),
        ctx.corridor.n_stations,
        ctx.disc,
        reward_mode=cfg["reward"]["mode"],
    )


def q_init_value(cfg: dict[str, Any], n_agents: int) -> float:
    """Neutral Q initialisation r_bar / (1 - gamma); r_bar is scaled down per agent for local rewards."""
    lc = cfg["learning"]
    r_bar = float(lc["q_init_reward_per_epoch"])
    if cfg["reward"]["mode"] == "local":
        r_bar /= n_agents
    return r_bar / (1.0 - float(lc["gamma"]))


def model_path(variant: str, algo: str, run: int, models_dir: Path | None = None) -> Path:
    """Where the Q-tables of one trained run are stored (default: results/models)."""
    return (models_dir or MODELS) / variant / f"{algo}_seed{run}.npz"


# ---- optional progress reporting from worker processes (used by the API's background jobs)
_PROGRESS = None


def init_progress(queue: Any) -> None:
    """Pool initializer: workers put (key, done, total) tuples on ``queue``."""
    global _PROGRESS
    _PROGRESS = queue


def report_progress(key: str, done: int, total: int) -> None:
    """Report progress of one unit of work if a progress queue is installed."""
    if _PROGRESS is not None:
        try:
            _PROGRESS.put_nowait((key, done, total))
        except Exception:  # noqa: BLE001 - progress is best effort
            pass


def n_workers(cfg: dict[str, Any]) -> int:
    """Worker processes for parallel jobs."""
    w = int(cfg["experiment"].get("workers", 0))
    if w <= 0:
        w = max(1, (os.cpu_count() or 2) - 1)
    return w


def write_json(path: Path, obj: Any) -> None:
    """Write compact JSON (floats rounded by the caller where it matters)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, separators=(",", ":"))


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rnd(x: float, nd: int = 2) -> float:
    """Round for JSON export."""
    return float(round(float(x), nd))
