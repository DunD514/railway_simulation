"""Static corridor geometry: stations, fixed blocks, sections and loops.

Positions are kilometres measured along the UP line from the start of the entry
approach (so CNB sits at ``entry_approach_km``). Block boundaries always include
every station, so a station is a block boundary and its loop/siding is entered
at that boundary.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Station:
    """A station with one loop/siding."""

    index: int
    code: str
    name: str
    km: float
    boundary: int  # index into Corridor.bounds


@dataclass(frozen=True)
class Corridor:
    """Fixed-block double-track corridor (one direction simulated)."""

    stations: tuple[Station, ...]
    bounds: tuple[float, ...]          # block boundaries, len = n_blocks + 1
    loop_capacity: int
    approach_zone_km: float
    turnout_speed_kmh: float
    turnout_zone_km: float
    block_section: tuple[int, ...] = field(default=())  # section index per block (0..n_stations)

    @property
    def n_blocks(self) -> int:
        """Number of fixed blocks."""
        return len(self.bounds) - 1

    @property
    def n_stations(self) -> int:
        """Number of stations (= number of agents)."""
        return len(self.stations)

    @property
    def length_km(self) -> float:
        """Total modelled length including entry approach and exit stretch."""
        return self.bounds[-1]

    def block_of(self, pos_km: float) -> int:
        """Return the block index containing ``pos_km`` (right-open blocks)."""
        b = bisect_right(self.bounds, pos_km) - 1
        return min(max(b, 0), self.n_blocks - 1)

    def section_of_block(self, block: int) -> int:
        """Section index of a block: k means the stretch ending at station k; n_stations = exit."""
        return self.block_section[block]

    def next_station_after(self, pos_km: float) -> int:
        """Index of the first station at or beyond ``pos_km`` (n_stations if none)."""
        for st in self.stations:
            if st.km >= pos_km:
                return st.index
        return self.n_stations

    def next_loop_km(self, station_index: int) -> float:
        """Km of the next overtaking opportunity after station ``station_index`` (or line end)."""
        if station_index + 1 < self.n_stations:
            return self.stations[station_index + 1].km
        return self.length_km

    def section_blocks(self, section: int) -> list[int]:
        """Blocks belonging to a section."""
        return [b for b, s in enumerate(self.block_section) if s == section]


def build_corridor(cfg: dict[str, Any]) -> Corridor:
    """Build the corridor from the ``corridor`` block of the config."""
    c = cfg["corridor"]
    block_km = float(c["block_km"])
    seg_lengths = [float(c["entry_approach_km"])] + [float(d) for d in c["distances_km"]] + [float(c["exit_km"])]
    bounds: list[float] = [0.0]
    station_boundaries: list[int] = []
    block_section: list[int] = []
    for seg_idx, length in enumerate(seg_lengths):
        n = max(1, int(round(length / block_km)))
        start = bounds[-1]
        for k in range(1, n + 1):
            bounds.append(round(start + length * k / n, 6))
            block_section.append(seg_idx)  # seg 0 -> ends at station 0, ..., last seg -> exit section
        if seg_idx < len(seg_lengths) - 1:
            station_boundaries.append(len(bounds) - 1)
    stations = tuple(
        Station(index=i, code=s["code"], name=s["name"], km=bounds[station_boundaries[i]], boundary=station_boundaries[i])
        for i, s in enumerate(c["stations"])
    )
    if len(stations) != len(c["distances_km"]) + 1:
        raise ValueError("need exactly one more station than inter-station distances")
    return Corridor(
        stations=stations,
        bounds=tuple(bounds),
        loop_capacity=int(c["loop_capacity"]),
        approach_zone_km=float(c["approach_zone_km"]),
        turnout_speed_kmh=float(c["turnout_speed_kmh"]),
        turnout_zone_km=float(c["turnout_zone_km"]),
        block_section=tuple(block_section),
    )
