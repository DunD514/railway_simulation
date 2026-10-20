"""Greedy evaluation on held-out timetables, statistics, behaviour analysis and replay export.

Usage: python -m experiments.evaluate

Protocol: run k (k = 0..n_seeds-1) owns ``eval_timetables_per_seed`` held-out
timetable seeds that no learner trained on and no other run uses. Every policy
(learned agents of run k, FCFS, passenger-first, random) is evaluated on run
k's timetables; the run-level mean is the unit of analysis. CIs are percentile
bootstrap over runs; comparisons are paired sign-flip permutation tests over
runs (same timetables), Holm-corrected within each metric family.
"""
from __future__ import annotations

import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agents.baselines import FCFSPolicy, GreedyQPolicy, PassengerFirstPolicy, RandomPolicy, run_policy_episode
from experiments.ablations import METRICS, ablation_table, compare, decision_stats, sensitivity_grid, summarize
from experiments.common import (
    RESULTS, Context, eval_timetable_seeds, is_default_cell, main_variants, model_path, n_workers, read_json, rnd,
    run_rng_seed, sweep_variants, write_json,
)
from experiments.stats import holm_correction
from sim.config import load_config, with_overrides
from sim.env import ACTION_NAMES, CorridorEnv
from sim.timetable import overtaking_sections
from sim.trains import STATUS_NAMES

BASELINES = ("fcfs", "passenger_first", "random")
POLICY_LABELS = {
    "iql": "IQL", "sarsa": "SARSA", "fcfs": "FCFS", "passenger_first": "Passenger-first", "random": "Random",
}


def _make_policy(job: dict[str, Any], cfg: dict[str, Any], ctx: Context):
    kind = job["policy"]
    if kind == "fcfs":
        return FCFSPolicy()
    if kind == "passenger_first":
        return PassengerFirstPolicy()
    if kind == "random":
        return RandomPolicy(np.random.default_rng(run_rng_seed(cfg, job["run"], salt=2)))
    data = np.load(model_path(job["variant"], kind, job["run"], Path(job["models_dir"])))
    return GreedyQPolicy(kind, list(data["q"]), ctx.disc)


def eval_job(job: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one policy of one run on that run's held-out timetables."""
    cfg = with_overrides(job["cfg"], job.get("overrides", {}))
    ctx = Context(cfg)
    policy = _make_policy(job, cfg, ctx)
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for seed in eval_timetable_seeds(cfg, job["run"]):
        env = CorridorEnv(cfg, ctx.timetable(seed), ctx.corridor, log_decisions=job.get("log", False))
        m = run_policy_episode(env, policy)
        m["timetable_seed"] = seed
        episodes.append(m)
        if job.get("log"):
            n_st = ctx.corridor.n_stations
            for d in env.log.decisions:
                p = env.trains[d["passenger"]]
                nxt = d["agent"] + 1
                d["downstream_pdelay"] = p.accrued_by_section[nxt] if nxt <= n_st else None
                d["timetable_seed"] = seed
                decisions.append(d)
    return {"key": job["key"], "run": job["run"], "episodes": episodes, "decisions": decisions}


def replay_job(job: dict[str, Any]) -> dict[str, Any]:
    """Record a full episode (positions per epoch, station events, agent actions) for the UI."""
    cfg = job["cfg"]
    ctx = Context(cfg)
    policy = _make_policy(job, cfg, ctx)
    tt = ctx.timetable(job["seed"])
    env = CorridorEnv(cfg, tt, ctx.corridor, record=True)
    m = run_policy_episode(env, policy)
    return episode_payload(env, tt, job["policy"], job["seed"], m)


def episode_payload(env: CorridorEnv, tt: Any, policy: str, seed: int, m: dict[str, Any]) -> dict[str, Any]:
    """JSON replay of a recorded episode (shared by offline evaluation and the live /api/simulate endpoint)."""
    trains = []
    for tr in env.trains:
        trains.append({
            "tid": tr.tid, "label": tr.sched.label, "cls": tr.cls.code, "kind": tr.cls.kind,
            "speed_kmh": tr.cls.speed_kmh, "sched_entry": rnd(tr.sched.sched_entry_min),
            "ready": rnd(tr.sched.ready_min), "sched_station": [rnd(x) for x in tr.sched.sched_station_min],
            "arrivals": [rnd(x) for x in tr.arrivals], "departures": [rnd(x) for x in tr.departures],
            "entered": rnd(tr.entered_min), "exit": rnd(tr.exit_min), "delay": rnd(tr.accrued, 1),
            "status": STATUS_NAMES[tr.status],
        })
    return {
        "policy": policy, "label": POLICY_LABELS[policy], "seed": seed,
        "utilization": rnd(tt.utilization, 3), "epoch_min": env.epoch_min,
        "trains": trains, "frames": env.log.frames, "events": env.log.events,
        "decisions": env.log.decisions,
        "metrics": {k: (rnd(v, 2) if isinstance(v, float) else v) for k, v in m.items()},
    }


def _dispatch(job: dict[str, Any]) -> dict[str, Any]:
    return replay_job(job) if job["kind"] == "replay" else eval_job(job)


def behaviour(decisions: list[dict[str, Any]], episodes: list[dict[str, Any]], n_st: int, window: float) -> dict[str, Any]:
    """Where/when a policy loops: per-agent actions, loop rate vs ETA margin and vs hour of day."""
    eta_edges = list(range(0, int(window) + 10, 10))
    per_agent = []
    for i in range(n_st):
        ds = [d for d in decisions if d["agent"] == i]
        appr = [d for d in ds if not d["in_loop"]]
        inl = [d for d in ds if d["in_loop"]]
        counts = [sum(1 for d in appr if d["action"] == a) for a in range(3)]
        per_agent.append({
            "agent": i,
            "approach_counts": counts,
            "in_loop_stay": sum(1 for d in inl if d["action"] != 0),
            "in_loop_release": sum(1 for d in inl if d["action"] == 0),
            "invalid": sum(1 for d in ds if not d["valid"]),
            "loop_uses_per_episode": float(np.mean([e["loop_uses_by_station"][i] for e in episodes])),
        })
    eta_rows = []
    for lo, hi in zip(eta_edges[:-1], eta_edges[1:]):
        appr = [d for d in decisions if not d["in_loop"] and lo <= d["eta"] < hi]
        inl = [d for d in decisions if d["in_loop"] and lo <= d["eta"] < hi]
        eta_rows.append({
            "lo": lo, "hi": hi, "n": len(appr),
            "loop_rate": (sum(1 for d in appr if d["action"] == 1) / len(appr)) if appr else None,
            "hold_rate": (sum(1 for d in appr if d["action"] == 2) / len(appr)) if appr else None,
            "n_in_loop": len(inl),
            "stay_rate": (sum(1 for d in inl if d["action"] != 0) / len(inl)) if inl else None,
        })
    hour_rows = []
    for h in range(24):
        appr = [d for d in decisions if not d["in_loop"] and h <= d["t"] / 60.0 < h + 1]
        hour_rows.append({
            "hour": h, "n": len(appr),
            "loop_rate": (sum(1 for d in appr if d["action"] == 1) / len(appr)) if appr else None,
        })
    return {"per_agent": per_agent, "eta": eta_rows, "hour": hour_rows, "summary": decision_stats(decisions)}


def policy_summary(
    variant: str, algo: str, n_runs: int, ctx: Context, top: int = 6, models_dir: Path | None = None
) -> list[dict[str, Any]]:
    """Per agent: the most visited states and the action most runs' greedy policy picks there."""
    qs, vs = [], []
    for r in range(n_runs):
        data = np.load(model_path(variant, algo, r, models_dir))
        qs.append(data["q"])
        vs.append(data["visits"])
    q = np.stack(qs)          # runs x agents x states x actions
    v = np.stack(vs).sum(axis=(0, 3))  # agents x states
    out = []
    for i in range(q.shape[1]):
        order = np.argsort(-v[i])[:top]
        states = []
        for s in order:
            if v[i, s] == 0:
                continue
            greedy = np.argmax(q[:, i, s, :], axis=1)
            votes = np.bincount(greedy, minlength=3)
            states.append({
                "state": int(s), "features": ctx.disc.decode(int(s)), "visits": int(v[i, s]),
                "votes": votes.tolist(), "action": ACTION_NAMES[int(np.argmax(votes))],
                "q_mean": [rnd(x, 1) for x in q[:, i, s, :].mean(axis=0)],
            })
        visited = int((v[i] > 0).sum())
        out.append({"agent": i, "visited_states": visited, "top_states": states})
    return out


def overview(cfg: dict[str, Any], ctx: Context) -> dict[str, Any]:
    """Problem figures, corridor geometry and timetable statistics for the Overview page."""
    seeds = [s for r in range(int(cfg["experiment"]["n_seeds"])) for s in eval_timetable_seeds(cfg, r)]
    tts = [ctx.timetable(s) for s in seeds]
    utils = [t.utilization for t in tts]
    sec_u = np.array([t.section_utilization for t in tts]).mean(axis=0)
    c = ctx.corridor
    pts = ["entry"] + [s.code for s in c.stations] + ["exit"]
    secs = overtaking_sections(c)
    return {
        "stations": [{"index": s.index, "code": s.code, "name": s.name, "km": s.km} for s in c.stations],
        "length_km": c.length_km,
        "n_blocks": c.n_blocks,
        "block_bounds": list(c.bounds),
        "block_km": cfg["corridor"]["block_km"],
        "entry_approach_km": cfg["corridor"]["entry_approach_km"],
        "exit_km": cfg["corridor"]["exit_km"],
        "approach_zone_km": c.approach_zone_km,
        "classes": [
            {"code": k, **{kk: vv for kk, vv in v.items()}} for k, v in cfg["trains"]["classes"].items()
        ],
        "time": cfg["time"],
        "timetable": {
            "n_eval_timetables": len(tts),
            "utilization_mean": float(np.mean(utils)),
            "utilization_min": float(np.min(utils)),
            "utilization_max": float(np.max(utils)),
            "target": cfg["timetable"]["target_utilization"],
            "trains_mean": float(np.mean([len(t.trains) for t in tts])),
            "passengers_mean": float(np.mean([t.n_passenger for t in tts])),
            "freights_mean": float(np.mean([t.n_freight for t in tts])),
            "window_min": cfg["timetable"]["injection_window_min"],
            "sections": [
                {"from": pts[k], "to": pts[k + 1], "km": secs[k], "utilization": float(sec_u[k])}
                for k in range(len(secs))
            ],
        },
        "control": cfg["control"],
        "reward": cfg["reward"],
        "learning": cfg["learning"],
        "discretization": ctx.disc.describe(),
        "discretization_nosignal": Context(with_overrides(cfg, {"discretization.use_downstream_signal": False})).disc.describe(),
        "experiment": {k: v for k, v in cfg["experiment"].items() if k != "sweep"},
        "sweep": cfg["experiment"]["sweep"],
    }


def verdicts(main: dict[str, Any]) -> list[str]:
    """Plain-language statements derived from the significance tests (no hand-written numbers)."""
    out = []
    pf = main["policies"]["passenger_first"]
    for algo in ("iql", "sarsa"):
        comp = main["comparisons"][f"{algo}_vs_passenger_first"]
        name = POLICY_LABELS[algo]
        for key, word in (("shared_return", "team return"), ("passenger_delay", "passenger delay")):
            c = comp[key]
            better = c["mean_diff"] > 0 if key == "shared_return" else c["mean_diff"] < 0
            sig = c["p_holm"] < 0.05
            a = main["policies"][algo][key]["mean"]
            b = pf[key]["mean"]
            if better and sig:
                verdict = "beats"
            elif better:
                verdict = "is ahead of, but not significantly different from,"
            elif sig:
                verdict = "is significantly worse than"
            else:
                verdict = "is behind, but not significantly different from,"
            out.append(f"{name} {verdict} the passenger-first baseline on {word}: "
                       f"{a:,.0f} vs {b:,.0f} (paired diff {c['mean_diff']:+,.0f}, "
                       f"95% CI [{c['ci_lo']:+,.0f}, {c['ci_hi']:+,.0f}], Holm p={c['p_holm']:.3f}).")
    c = main["comparisons"]["iql_vs_sarsa"]["shared_return"]
    out.append(f"IQL vs SARSA team return: paired diff {c['mean_diff']:+,.0f} "
               f"(95% CI [{c['ci_lo']:+,.0f}, {c['ci_hi']:+,.0f}], Holm p={c['p_holm']:.3f}).")
    return out


def available_runs(models_dir: Path, variant: str, algos: tuple[str, ...] = ("iql", "sarsa")) -> int:
    """Number of consecutive runs 0..k-1 whose Q-tables exist for every algorithm of ``variant``."""
    k = 0
    while all(model_path(variant, a, k, models_dir).is_file() for a in algos):
        k += 1
    return k


def evaluate_all(
    cfg: dict[str, Any],
    out_dir: Path,
    models_dir: Path,
    pool: Any,
    curves_path: Path | None = None,
    log: Callable[[str], None] = print,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate every available trained model on ``cfg``'s held-out timetables; write all result JSON to ``out_dir``.

    Works with whatever ``models_dir`` contains: the main runs are required, ablation and sweep tables are
    evaluated when present (their pages report them as unavailable otherwise).
    """
    ctx = Context(cfg)
    n_runs = available_runs(models_dir, "main")
    if n_runs == 0:
        raise FileNotFoundError(f"no trained main models in {models_dir}")
    n_boot = int(cfg["experiment"]["bootstrap_resamples"])
    sw = cfg["experiment"]["sweep"]
    curves = read_json(curves_path) if curves_path and curves_path.is_file() else None
    variants = {v.name: v for v in main_variants() if available_runs(models_dir, v.name) >= n_runs}
    has_ablations = {"local", "nosignal"} <= set(variants)
    sweep = sweep_variants(cfg)
    sw_runs = min(int(sw["n_seeds"]), n_runs)
    sweep_avail = all(is_default_cell(cfg, v) or available_runs(models_dir, v.name, tuple(sw["algos"])) >= sw_runs
                      for v in sweep)
    base = {"cfg": cfg, "models_dir": str(models_dir)}
    jobs: list[dict[str, Any]] = []
    for run in range(n_runs):
        for pol in BASELINES:
            jobs.append({**base, "kind": "eval", "key": pol, "policy": pol, "run": run,
                         "log": pol == "passenger_first"})
        for v in variants.values():
            for algo in ("iql", "sarsa"):
                jobs.append({**base, "kind": "eval", "key": f"{v.name}/{algo}", "policy": algo, "variant": v.name,
                             "overrides": v.overrides, "run": run, "log": True})
    if sweep_avail:
        for v in sweep:
            if is_default_cell(cfg, v):
                continue
            for algo in sw["algos"]:
                for run in range(sw_runs):
                    jobs.append({**base, "kind": "eval", "key": f"{v.name}/{algo}", "policy": algo,
                                 "variant": v.name, "overrides": v.overrides, "run": run})
    replay_seeds = eval_timetable_seeds(cfg, 0)[: int(cfg["experiment"]["replay_seeds"])]
    for seed in replay_seeds:
        for pol in ("fcfs", "passenger_first", "iql", "sarsa"):
            jobs.append({**base, "kind": "replay", "policy": pol, "variant": "main", "run": 0, "seed": seed})
    log(f"evaluation jobs: {len(jobs)} ({n_runs} runs; ablations {'yes' if has_ablations else 'no'}; "
        f"sweep {'yes' if sweep_avail else 'no'})")
    t0 = time.time()
    out = []
    for i, res in enumerate(pool.imap(_dispatch, jobs, chunksize=1), 1):
        out.append(res)
        if progress:
            progress(i, len(jobs))
    log(f"evaluation wall time: {time.time() - t0:.1f}s")

    results: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    decisions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    replays = []
    for res in out:
        if "frames" in res:
            replays.append(res)
            continue
        results[res["key"]][res["run"]] = res["episodes"]
        decisions[res["key"]].extend(res["decisions"])

    # ---------------- main table
    main_keys = {"iql": "main/iql", "sarsa": "main/sarsa", "fcfs": "fcfs",
                 "passenger_first": "passenger_first", "random": "random"}
    table = {name: summarize(results[k], n_boot) for name, k in main_keys.items()}
    comp_keys = ("passenger_delay", "freight_delay", "shared_return", "trains_cleared")
    pairs = {"iql_vs_sarsa": ("main/iql", "main/sarsa"),
             "iql_vs_passenger_first": ("main/iql", "passenger_first"),
             "sarsa_vs_passenger_first": ("main/sarsa", "passenger_first"),
             "iql_vs_fcfs": ("main/iql", "fcfs"),
             "sarsa_vs_fcfs": ("main/sarsa", "fcfs")}
    comparisons = {name: compare(results[a], results[b], comp_keys, n_boot) for name, (a, b) in pairs.items()}
    for k in comp_keys:  # Holm correction over the five comparisons for each metric
        adj = holm_correction([comparisons[n][k]["p_value"] for n in pairs])
        for n, p in zip(pairs, adj):
            comparisons[n][k]["p_holm"] = p
    main_out = {
        "protocol": {
            "n_runs": n_runs,
            "eval_timetables_per_run": int(cfg["experiment"]["eval_timetables_per_seed"]),
            "train_episodes": int(curves["episodes"]) if curves else int(cfg["experiment"]["train_episodes"]),
            "ci": "95% percentile bootstrap over run-level means",
            "test": "two-sided paired sign-flip permutation test over runs (exact), Holm-corrected per metric",
            "bootstrap_resamples": n_boot,
        },
        "metrics": [{"key": k, "label": lab, "unit": u} for k, lab, u in METRICS],
        "order": list(main_keys),
        "labels": POLICY_LABELS,
        "policies": table,
        "comparisons": comparisons,
        "comparison_labels": {n: f"{POLICY_LABELS[n.split('_vs_')[0]]} vs {POLICY_LABELS[n.split('_vs_')[1]]}"
                              for n in pairs},
    }
    main_out["verdicts"] = verdicts(main_out)
    write_json(out_dir / "eval_main.json", main_out)

    # ---------------- ablations
    if has_ablations:
        abl_a = ablation_table("A", "shared reward", "local-only reward", results, decisions,
                               {"iql": ("main/iql", "local/iql"), "sarsa": ("main/sarsa", "local/sarsa")}, n_boot)
        abl_b = ablation_table("B", "with downstream signal", "without signal", results, decisions,
                               {"iql": ("main/iql", "nosignal/iql"), "sarsa": ("main/sarsa", "nosignal/sarsa")}, n_boot)
        write_json(out_dir / "ablations.json", {
            "available": True, "stations": [s.code for s in ctx.corridor.stations] + ["exit"], "A": abl_a, "B": abl_b,
        })
    else:
        write_json(out_dir / "ablations.json", {"available": False,
                                                "reason": "these agents were trained without the ablation variants"})

    # ---------------- sensitivity
    if sweep_avail:
        def key_of(variant: str, algo: str) -> str:
            v = next(x for x in sweep if x.name == variant)
            if is_default_cell(cfg, v):
                k = f"main_first{sw_runs}/{algo}"
                if k not in results:
                    results[k] = {r: results[f"main/{algo}"][r] for r in range(sw_runs)}
                return k
            return f"{variant}/{algo}"

        sens = sensitivity_grid(cfg, results, key_of, n_boot)
        sens["n_seeds"] = sw_runs
        sens["available"] = True
        sens["curves"] = {k: v for k, v in curves["variants"].items()
                          if k.startswith("sweep_") or k.startswith("eps_")} if curves else {}
        write_json(out_dir / "sensitivity.json", sens)
    else:
        write_json(out_dir / "sensitivity.json", {"available": False,
                                                  "reason": "these agents were trained without the hyperparameter sweep"})

    # ---------------- behaviour
    window = float(cfg["control"]["window_min"])
    n_st = ctx.corridor.n_stations
    beh = {"stations": [s.code for s in ctx.corridor.stations], "policies": {}, "labels": POLICY_LABELS}
    for name, k in (("iql", "main/iql"), ("sarsa", "main/sarsa"), ("passenger_first", "passenger_first")):
        eps_all = [m for r in results[k] for m in results[k][r]]
        beh["policies"][name] = behaviour(decisions[k], eps_all, n_st, window)
    beh["policy_summary"] = {algo: policy_summary("main", algo, n_runs, ctx, models_dir=models_dir)
                             for algo in ("iql", "sarsa")}
    beh["feature_names"] = list(ctx.disc.feature_names)
    beh["level_labels"] = ctx.disc.level_labels()
    write_json(out_dir / "behavior.json", beh)

    # ---------------- replays + overview
    index = []
    for rp in replays:
        name = f"{rp['policy']}_{rp['seed']}"
        write_json(out_dir / "replays" / f"{name}.json", rp)
        index.append({"policy": rp["policy"], "label": rp["label"], "seed": rp["seed"], "file": name,
                      "passenger_delay": rp["metrics"]["passenger_delay"], "freight_delay": rp["metrics"]["freight_delay"]})
    write_json(out_dir / "replays" / "index.json", {"seeds": replay_seeds, "policies": ["fcfs", "passenger_first", "iql", "sarsa"],
                                                    "labels": POLICY_LABELS, "items": index})
    write_json(out_dir / "overview.json", overview(cfg, ctx))
    return {"main": main_out, "table": table, "main_keys": main_keys}


def main() -> None:
    """Entry point: evaluate results/models on the configured held-out timetables into results/."""
    cfg = load_config()
    with get_context("spawn").Pool(n_workers(cfg)) as pool:
        r = evaluate_all(cfg, RESULTS, RESULTS / "models", pool, curves_path=RESULTS / "curves_summary.json")
    main_out, table, main_keys = r["main"], r["table"], r["main_keys"]

    # ---------------- console report
    print("\nMain comparison (held-out, mean [95% CI] over runs):")
    hdr = f"{'policy':<16}" + "".join(f"{lab[:22]:>34}" for _, lab, _ in METRICS[:5])
    print(hdr)
    for name in main_keys:
        row = f"{POLICY_LABELS[name]:<16}"
        for k, _, _ in METRICS[:5]:
            s = table[name][k]
            row += f"{s['mean']:>12.1f} [{s['lo']:>8.1f},{s['hi']:>9.1f}]"
        print(row)
    print()
    for v in main_out["verdicts"]:
        print(" -", v)


if __name__ == "__main__":
    main()
