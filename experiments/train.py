"""Train IQL and SARSA (main experiment, ablations, hyperparameter sweep) in parallel.

Usage: python -m experiments.train [--episodes N] [--seeds K] [--no-sweep]

Writes Q-tables to results/models/<variant>/<algo>_seed<k>.npz and learning
curves to results/curves/<variant>.json plus results/curves_summary.json.
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.baselines import FCFSPolicy, PassengerFirstPolicy, run_policy_episode
from agents.tabular import linear_epsilon, run_learning_episode
from experiments.common import (
    RESULTS, Context, Variant, is_default_cell, main_variants, make_learners, model_path, n_workers, report_progress,
    rnd, run_rng_seed, sweep_variants, train_timetable_seed, write_json,
)
from sim.config import load_config, with_overrides
from sim.env import CorridorEnv

CURVE_KEYS = ("shared_return", "passenger_delay", "freight_delay", "invalid_rate", "loop_uses", "trains_cleared")
BAND_KEYS = ("shared_return", "passenger_delay", "freight_delay")


def train_job(job: dict[str, Any]) -> dict[str, Any]:
    """Train one (variant, algo, run) and save its Q-tables. Returns the per-episode curve."""
    cfg = with_overrides(job["cfg"], job["overrides"])
    ctx = Context(cfg)
    run, algo = job["run"], job["algo"]
    learners = make_learners(algo, cfg, ctx)
    rng = np.random.default_rng(run_rng_seed(cfg, run, salt=1))
    n_ep = int(cfg["experiment"]["train_episodes"])
    eps = cfg["learning"]["epsilon"]
    curve: dict[str, list[float]] = {k: [] for k in CURVE_KEYS}
    curve["epsilon"] = []
    t0 = time.time()
    for ep in range(n_ep):
        e = linear_epsilon(ep, n_ep, eps["start"], eps["end"], eps["decay_frac"])
        env = CorridorEnv(cfg, ctx.timetable(train_timetable_seed(cfg, run, ep)), ctx.corridor)
        m = run_learning_episode(env, learners, e, rng)
        for k in CURVE_KEYS:
            curve[k].append(rnd(m[k], 3))
        curve["epsilon"].append(rnd(e, 4))
        if (ep + 1) % 10 == 0 or ep + 1 == n_ep:
            report_progress(job["key"], ep + 1, n_ep)
    path = model_path(job["variant"], algo, run, Path(job["models_dir"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        q=np.stack(learners.q_tables()),
        visits=np.stack([ag.visits for ag in learners.agents]),
    )
    visited = [int((ag.visits.sum(axis=1) > 0).sum()) for ag in learners.agents]
    return {
        "variant": job["variant"], "algo": algo, "run": run, "curve": curve,
        "seconds": time.time() - t0, "visited_states": visited,
    }


def reference_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run a heuristic baseline on the same training timetables (reference line for learning curves)."""
    cfg = job["cfg"]
    ctx = Context(cfg)
    policy = {"fcfs": FCFSPolicy(), "passenger_first": PassengerFirstPolicy()}[job["policy"]]
    curve: dict[str, list[float]] = {k: [] for k in CURVE_KEYS}
    for ep in range(int(cfg["experiment"]["train_episodes"])):
        env = CorridorEnv(cfg, ctx.timetable(train_timetable_seed(cfg, job["run"], ep)), ctx.corridor)
        m = run_policy_episode(env, policy)
        for k in CURVE_KEYS:
            curve[k].append(rnd(m[k], 3))
        if (ep + 1) % 25 == 0 or ep + 1 == int(cfg["experiment"]["train_episodes"]):
            report_progress(job["key"], ep + 1, int(cfg["experiment"]["train_episodes"]))
    return {"policy": job["policy"], "run": job["run"], "curve": curve}


def _dispatch(job: dict[str, Any]) -> dict[str, Any]:
    return reference_job(job) if job["kind"] == "ref" else train_job(job)


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing moving average along the last axis (window shrinks at the start)."""
    c = np.cumsum(np.pad(x, [(0, 0)] * (x.ndim - 1) + [(1, 0)]), axis=-1)
    idx = np.arange(1, x.shape[-1] + 1)
    lo = np.maximum(0, idx - w)
    return (c[..., idx] - c[..., lo]) / (idx - lo)


def band(per_seed: list[list[float]], w: int, n_boot: int = 2000, seed: int = 0) -> dict[str, list[float]]:
    """Smoothed mean curve with a 95% bootstrap band over seeds (vectorised)."""
    x = smooth(np.asarray(per_seed, dtype=float), w)
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n > 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        means = x[idx].mean(axis=1)
        lo, hi = np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)
    else:
        lo = hi = x[0]
    return {
        "mean": [rnd(v, 1) for v in x.mean(axis=0)],
        "lo": [rnd(v, 1) for v in lo],
        "hi": [rnd(v, 1) for v in hi],
    }


def plan_jobs(cfg: dict[str, Any], models_dir: Path, ablations: bool, sweep: bool) -> tuple[list[dict[str, Any]], list[Variant]]:
    """All training and reference jobs for a configuration (deterministic order)."""
    n_seeds = int(cfg["experiment"]["n_seeds"])
    variants = main_variants() if ablations else main_variants()[:1]
    jobs: list[dict[str, Any]] = []
    for v in variants:
        for algo in ("iql", "sarsa"):
            for run in range(n_seeds):
                jobs.append({"kind": "train", "variant": v.name, "overrides": v.overrides, "algo": algo,
                             "run": run, "cfg": cfg, "models_dir": str(models_dir)})
    sweep_vs: list[Variant] = sweep_variants(cfg) if sweep else []
    sw_seeds = int(cfg["experiment"]["sweep"]["n_seeds"])
    for v in sweep_vs:
        if is_default_cell(cfg, v):
            continue  # identical to the main runs: reuse them
        for algo in cfg["experiment"]["sweep"]["algos"]:
            for run in range(sw_seeds):
                jobs.append({"kind": "train", "variant": v.name, "overrides": v.overrides, "algo": algo,
                             "run": run, "cfg": cfg, "models_dir": str(models_dir)})
    for pol in ("fcfs", "passenger_first"):
        for run in range(n_seeds):
            jobs.append({"kind": "ref", "policy": pol, "run": run, "cfg": cfg})
    for j in jobs:
        j["key"] = f"{j['kind']}/{j.get('variant', j.get('policy'))}/{j.get('algo', '')}/{j['run']}"
    return jobs, sweep_vs


def train_all(
    cfg: dict[str, Any],
    out_dir: Path,
    pool: Any,
    ablations: bool = True,
    sweep: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Train every planned job on ``pool``; write models, curves and curves_summary.json under ``out_dir``."""
    ctx = Context(cfg)
    n_seeds = int(cfg["experiment"]["n_seeds"])
    n_ep = int(cfg["experiment"]["train_episodes"])
    log(f"corridor: {ctx.corridor.n_blocks} blocks, {ctx.corridor.n_stations} stations/agents, "
        f"{ctx.corridor.length_km:.0f} km modelled")
    for v in main_variants():
        d = Context(with_overrides(cfg, v.overrides)).disc
        log(f"Q-table size per agent [{v.name}]: {d.n_states} states x 3 actions = {d.n_states * 3} entries "
            f"(radices {d.radices}); 6 agents")
    log(f"training: {n_ep} episodes x {n_seeds} seeds")
    jobs, sweep_vs = plan_jobs(cfg, out_dir / "models", ablations, sweep)
    sw_seeds = int(cfg["experiment"]["sweep"]["n_seeds"])
    log(f"jobs: {len(jobs)}")

    t0 = time.time()
    results: list[dict[str, Any]] = []
    for k, res in enumerate(pool.imap_unordered(_dispatch, jobs), 1):
        results.append(res)
        if res.get("algo"):
            last = np.mean(res["curve"]["shared_return"][-max(1, n_ep // 10):])
            log(f"[{k}/{len(jobs)}] {res['variant']:<22} {res['algo']:<5} seed {res['run']:<2} "
                f"{res['seconds']:6.1f}s  final-10% return {last:9.0f}  visited states/agent {res['visited_states']}")
        else:
            log(f"[{k}/{len(jobs)}] reference {res['policy']} seed {res['run']}")
    log(f"training wall time: {time.time() - t0:.1f}s")

    # ---- assemble curves
    w = int(cfg["experiment"]["curve_smooth"])
    refs: dict[str, dict[str, list[list[float]]]] = {}
    for pol in ("fcfs", "passenger_first"):
        rs = sorted((r for r in results if r.get("policy") == pol), key=lambda r: r["run"])
        refs[pol] = {k: [r["curve"][k] for r in rs] for k in CURVE_KEYS}
    trained = main_variants() if ablations else main_variants()[:1]
    variants = {v.name: v for v in trained + sweep_vs}
    summary: dict[str, Any] = {"episodes": n_ep, "smooth": w, "variants": {}, "reference": {}}
    for pol, data in refs.items():
        summary["reference"][pol] = {k: band(data[k], w) for k in BAND_KEYS}
    sweep_names = {s.name for s in sweep_vs}
    for name, v in variants.items():
        src = "main" if (name != "main" and name in sweep_names and is_default_cell(cfg, v)) else name
        per_algo: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        for algo in ("iql", "sarsa"):
            rs = sorted((r for r in results if r.get("variant") == src and r.get("algo") == algo),
                        key=lambda r: r["run"])
            if name in sweep_names:
                rs = rs[:sw_seeds]
            if not rs:
                continue
            raw[algo] = {k: [r["curve"][k] for r in rs] for k in list(CURVE_KEYS) + ["epsilon"]}
            per_algo[algo] = {k: band(raw[algo][k], w) for k in BAND_KEYS}
            per_algo[algo]["n_seeds"] = len(rs)
            per_algo[algo]["epsilon"] = rs[0]["curve"]["epsilon"]
        summary["variants"][name] = {"label": v.label, "overrides": v.overrides, "reused_from": src, "algos": per_algo}
        write_json(out_dir / "curves" / f"{name}.json",
                   {"variant": name, "label": v.label, "episodes": n_ep, "per_seed": raw,
                    "reference": refs if name == "main" else None})
    write_json(out_dir / "curves_summary.json", summary)
    return summary


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    if args.episodes:
        cfg["experiment"]["train_episodes"] = args.episodes
    if args.seeds:
        cfg["experiment"]["n_seeds"] = args.seeds
        cfg["experiment"]["sweep"]["n_seeds"] = min(args.seeds, cfg["experiment"]["sweep"]["n_seeds"])
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"workers={n_workers(cfg)}")
    with get_context("spawn").Pool(n_workers(cfg)) as pool:
        train_all(cfg, RESULTS, pool, ablations=True, sweep=not args.no_sweep, log=log)
    (RESULTS / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    log("wrote results/curves_summary.json and results/curves/*.json")


if __name__ == "__main__":
    main()
