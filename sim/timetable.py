"""Synthetic timetable generator calibrated to a utilization band.

Utilization
-----------
We use a compression-style headway method (in the spirit of UIC 406). For each
overtaking section s (the stretch between two consecutive loops, length D_s) and
for each consecutive pair (leader l, follower f) in scheduled entry order, the
minimum headway the pair consumes at the section entrance is

    h_s(l, f) = max(0, D_s / v_l - D_s / v_f) + t_clear(l, f)
    t_clear(l, f) = (B + o_l) / v_l + B / v_f

where B is the block length and o_l the leader's overlap (the freight braking
distance for freight). The first term is the "speed differential" running-time
gap that prevents a faster follower from catching a slower leader before the
next loop; the second is the fixed-block clearance (leader clears block plus
overlap, follower needs one clear block to see a green). Utilization of section
s over the injection window T is

    U_s = sum_pairs h_s(l, f) / T

and the corridor utilization is max_s U_s (the bottleneck). U > 1 means the
timetable is infeasible without overtaking at loops, i.e. the corridor is
saturated and precedence decisions are unavoidable. The generator draws a class
sequence and entry times from the seed and adjusts the number of trains until
the bottleneck utilization lands inside the configured band.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sim.corridor import Corridor
from sim.trains import ScheduledTrain, TrainClass


@dataclass(frozen=True)
class Timetable:
    """A generated timetable plus its utilization diagnostics."""

    seed: int
    trains: tuple[ScheduledTrain, ...]
    utilization: float
    section_utilization: tuple[float, ...]
    window_min: float

    @property
    def n_passenger(self) -> int:
        """Number of passenger trains."""
        return sum(1 for t in self.trains if t.cls.is_passenger)

    @property
    def n_freight(self) -> int:
        """Number of freight trains."""
        return sum(1 for t in self.trains if not t.cls.is_passenger)


def free_run_station_times(cls: TrainClass, corridor: Corridor) -> list[float]:
    """Unobstructed minutes from corridor entry to arrival at each station (dwells included)."""
    out: list[float] = []
    t = 0.0
    prev_km = 0.0
    for i, st in enumerate(corridor.stations):
        t += (st.km - prev_km) / cls.speed_kpm
        out.append(t)
        if cls.is_passenger:
            t += cls.dwell_min
        prev_km = st.km
    return out


def overtaking_sections(corridor: Corridor) -> list[float]:
    """Lengths of the stretches between consecutive overtaking points (entry, loops, exit)."""
    pts = [0.0] + [s.km for s in corridor.stations] + [corridor.length_km]
    return [pts[k + 1] - pts[k] for k in range(len(pts) - 1)]


def pair_headway(lead: TrainClass, follow: TrainClass, section_km: float, block_km: float) -> float:
    """Minimum headway (min) consumed by a (leader, follower) pair on one overtaking section."""
    diff = max(0.0, section_km / lead.speed_kpm - section_km / follow.speed_kpm)
    clear = (block_km + lead.overlap_km) / lead.speed_kpm + block_km / follow.speed_kpm
    return diff + clear


def compute_utilization(
    classes_in_order: list[TrainClass], corridor: Corridor, window_min: float, block_km: float
) -> tuple[float, list[float]]:
    """Return (bottleneck utilization, per-section utilization) for an entry-ordered class sequence."""
    per_section: list[float] = []
    for d in overtaking_sections(corridor):
        total = 0.0
        for a, b in zip(classes_in_order[:-1], classes_in_order[1:]):
            total += pair_headway(a, b, d, block_km)
        per_section.append(total / window_min)
    return max(per_section), per_section


def _draw(
    rng: np.random.Generator, n: int, cfg: dict[str, Any], classes: dict[str, TrainClass]
) -> list[TrainClass]:
    tt = cfg["timetable"]
    mix = tt["passenger_mix"]
    p_codes = list(mix.keys())
    p_probs = np.array([mix[c] for c in p_codes], dtype=float)
    p_probs /= p_probs.sum()
    n_pass = int(round(n * float(tt["passenger_share"])))
    kinds = ["P"] * n_pass + ["F"] * (n - n_pass)
    rng.shuffle(kinds)
    out: list[TrainClass] = []
    for k in kinds:
        if k == "F":
            out.append(classes["FRT"])
        else:
            out.append(classes[p_codes[int(rng.choice(len(p_codes), p=p_probs))]])
    return out


def generate_timetable(seed: int, cfg: dict[str, Any], corridor: Corridor, classes: dict[str, TrainClass]) -> Timetable:
    """Generate a seeded timetable whose bottleneck utilization lies in the target band.

    Entry times are drawn uniformly over the injection window and sorted, then
    pushed apart so consecutive trains respect the plain block clearance time
    (the timetable is feasible at the entry signal, but not without overtaking).
    """
    tt = cfg["timetable"]
    window = float(tt["injection_window_min"])
    lo, hi = (float(x) for x in tt["target_utilization"])
    block_km = float(cfg["corridor"]["block_km"])
    target = 0.5 * (lo + hi)
    rng = np.random.default_rng(seed)
    n = 20
    seq: list[TrainClass] = []
    util = 0.0
    per: list[float] = []
    for _ in range(int(tt["max_calibration_iters"])):
        seq = _draw(rng, n, cfg, classes)
        util, per = compute_utilization(seq, corridor, window, block_km)
        if lo <= util <= hi:
            break
        # proportional correction of the train count, at least one train per step
        n_new = int(round(n * target / max(util, 1e-6)))
        if n_new == n:
            n_new = n + (1 if util < lo else -1)
        n = max(4, n_new)
    else:
        raise RuntimeError(f"timetable calibration failed for seed {seed} (last U={util:.3f})")

    # entry times: sorted uniform draws, then enforce minimum block clearance between consecutive trains
    times = np.sort(rng.uniform(0.0, window, size=len(seq)))
    for k in range(1, len(seq)):
        lead, follow = seq[k - 1], seq[k]
        gap = (block_km + lead.overlap_km) / lead.speed_kpm + block_km / follow.speed_kpm
        times[k] = max(times[k], times[k - 1] + gap)
    jitter = float(tt["departure_jitter_min"])
    p_late = float(tt["initial_delay_prob"])
    mean_late = float(tt["initial_delay_mean_min"])
    trains: list[ScheduledTrain] = []
    for tid, (cls, t0) in enumerate(zip(seq, times)):
        j = float(rng.uniform(0.0, jitter))
        late = float(rng.exponential(mean_late)) if rng.random() < p_late else 0.0
        st_times = tuple(round(float(t0) + x, 4) for x in free_run_station_times(cls, corridor))
        trains.append(
            ScheduledTrain(
                tid=tid,
                cls=cls,
                sched_entry_min=round(float(t0), 4),
                ready_min=round(float(t0) + j + late, 4),
                sched_station_min=st_times,
            )
        )
    return Timetable(
        seed=seed, trains=tuple(trains), utilization=float(util), section_utilization=tuple(per), window_min=window
    )
