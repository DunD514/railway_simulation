"""Ablation and sensitivity tables built from per-run evaluation results."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from experiments.stats import bootstrap_ci, holm_correction, paired_comparison

METRICS = (
    ("passenger_delay", "Passenger delay", "min"),
    ("freight_delay", "Freight delay", "min"),
    ("shared_return", "Team return", "reward"),
    ("trains_cleared", "Trains cleared within 24 h", "trains"),
    ("loop_uses", "Loop uses", "count"),
    ("invalid_rate", "Invalid-action rate", "fraction"),
    ("station_arrival_delay", "Station arrival delay (sum)", "min"),
)


def run_means(episodes_by_run: dict[int, list[dict[str, Any]]], key: str) -> list[float]:
    """Per-run mean of a metric (runs sorted by index)."""
    return [float(np.mean([m[key] for m in episodes_by_run[r]])) for r in sorted(episodes_by_run)]


def summarize(episodes_by_run: dict[int, list[dict[str, Any]]], n_boot: int) -> dict[str, Any]:
    """Mean and bootstrap 95% CI over runs for every reported metric."""
    out: dict[str, Any] = {"n_runs": len(episodes_by_run),
                           "n_episodes": sum(len(v) for v in episodes_by_run.values())}
    for key, _, _ in METRICS:
        vals = run_means(episodes_by_run, key)
        m, lo, hi = bootstrap_ci(vals, n_resamples=n_boot)
        out[key] = {"mean": m, "lo": lo, "hi": hi, "per_run": vals}
    return out


def compare(
    a: dict[int, list[dict[str, Any]]],
    b: dict[int, list[dict[str, Any]]],
    keys: Sequence[str],
    n_boot: int,
) -> dict[str, Any]:
    """Paired comparison a-b on runs present in both (same held-out timetables)."""
    runs = sorted(set(a) & set(b))
    out: dict[str, Any] = {}
    for k in keys:
        va = [float(np.mean([m[k] for m in a[r]])) for r in runs]
        vb = [float(np.mean([m[k] for m in b[r]])) for r in runs]
        out[k] = paired_comparison(va, vb, n_resamples=n_boot)
    return out


def section_profile(episodes_by_run: dict[int, list[dict[str, Any]]], key: str) -> list[float]:
    """Mean per-section delay profile (sections 0..n_stations)."""
    arr = np.array([m[key] for r in episodes_by_run for m in episodes_by_run[r]], dtype=float)
    return [float(x) for x in arr.mean(axis=0)]


def decision_stats(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Action shares for approach decisions and the 'downstream dumping' measure."""
    appr = [d for d in decisions if not d["in_loop"]]
    n = max(1, len(appr))
    proceed = [d for d in appr if d["action"] == 0]
    loop = [d for d in appr if d["action"] == 1]
    hold = [d for d in appr if d["action"] == 2]
    in_loop = [d for d in decisions if d["in_loop"]]

    def mean_ds(ds: list[dict[str, Any]]) -> float | None:
        vals = [d["downstream_pdelay"] for d in ds if d.get("downstream_pdelay") is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "approach_decisions": len(appr),
        "in_loop_decisions": len(in_loop),
        "proceed_share": len(proceed) / n,
        "loop_share": len(loop) / n,
        "hold_share": len(hold) / n,
        "release_share": (sum(1 for d in in_loop if d["action"] == 0) / len(in_loop)) if in_loop else 0.0,
        # delay the conflicting passenger later accrued in the next section, by the action taken
        "downstream_pdelay_after_proceed": mean_ds(proceed),
        "downstream_pdelay_after_loop": mean_ds(loop),
        "downstream_pdelay_all": mean_ds(appr),
    }


def ablation_table(
    name: str,
    label_a: str,
    label_b: str,
    results: dict[str, dict[int, list[dict[str, Any]]]],
    decisions: dict[str, list[dict[str, Any]]],
    pairs: dict[str, tuple[str, str]],
    n_boot: int,
) -> dict[str, Any]:
    """Build one ablation table: for each algorithm, variant A vs variant B with paired tests."""
    rows: dict[str, Any] = {}
    keys = [k for k, _, _ in METRICS]
    for algo, (ka, kb) in pairs.items():
        comp = compare(results[kb], results[ka], keys, n_boot)
        pv = holm_correction([comp[k]["p_value"] for k in ("passenger_delay", "freight_delay", "shared_return")])
        for k, p in zip(("passenger_delay", "freight_delay", "shared_return"), pv):
            comp[k]["p_holm"] = p
        rows[algo] = {
            "a": {"key": ka, "label": label_a, "summary": summarize(results[ka], n_boot),
                  "sections_p": section_profile(results[ka], "section_passenger_delay"),
                  "sections_f": section_profile(results[ka], "section_freight_delay"),
                  "decisions": decision_stats(decisions.get(ka, []))},
            "b": {"key": kb, "label": label_b, "summary": summarize(results[kb], n_boot),
                  "sections_p": section_profile(results[kb], "section_passenger_delay"),
                  "sections_f": section_profile(results[kb], "section_freight_delay"),
                  "decisions": decision_stats(decisions.get(kb, []))},
            "b_minus_a": comp,
        }
    return {"name": name, "label_a": label_a, "label_b": label_b, "rows": rows}


def sensitivity_grid(
    cfg: dict[str, Any],
    results: dict[str, dict[int, list[dict[str, Any]]]],
    key_of: Any,
    n_boot: int,
) -> dict[str, Any]:
    """alpha x gamma grid and epsilon-schedule table of greedy held-out performance per algorithm."""
    sw = cfg["experiment"]["sweep"]
    out: dict[str, Any] = {"alphas": sw["alphas"], "gammas": sw["gammas"], "algos": {}, "n_seeds": sw["n_seeds"]}
    for algo in sw["algos"]:
        grid = []
        for a in sw["alphas"]:
            row = []
            for g in sw["gammas"]:
                res = results[key_of(f"sweep_a{a}_g{g}", algo)]
                s = summarize(res, n_boot)
                row.append({"alpha": a, "gamma": g,
                            "shared_return": s["shared_return"], "passenger_delay": s["passenger_delay"],
                            "freight_delay": s["freight_delay"]})
            grid.append(row)
        eps_rows = []
        for name, sched in sw["epsilon_schedules"].items():
            s = summarize(results[key_of(f"eps_{name}", algo)], n_boot)
            eps_rows.append({"name": name, "schedule": sched, "shared_return": s["shared_return"],
                             "passenger_delay": s["passenger_delay"], "freight_delay": s["freight_delay"]})
        out["algos"][algo] = {"grid": grid, "epsilon": eps_rows}
    return out
